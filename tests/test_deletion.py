"""文档删除状态机测试：原始文件删除失败必须保留元数据（可恢复），批量删除走同一服务。

用假 pipeline/repo/object_store，不依赖 PG、对象存储或向量库。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.errors import ObjectStoreError
from app.domain.kb_config import KBConfig
from app.orchestration.deletion import DocumentDeletionService


class _FakePipeline:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.deleted: list[str] = []
        # 记录缓存失效调用：DELETING 落库后必须立即失效检索缓存（v2.1 P0-3）
        self.cache_invalidated: list[str] = []

    def invalidate_retrieval_cache(self, kb: KBConfig) -> None:
        self.cache_invalidated.append(kb.kb_id)

    def delete_document(self, kb: KBConfig, doc_id: str) -> Any:
        if self.fail:
            raise ObjectStoreError("索引删除失败")
        self.deleted.append(doc_id)
        return SimpleNamespace(doc_id=doc_id, deleted_vectors=2, deleted_keywords=3)


class _FakeObjectStore:
    def __init__(
        self, *, fail: bool = False, fail_keys: set[str] | None = None
    ) -> None:
        self.fail = fail
        self.fail_keys = fail_keys or set()
        self.deleted: list[str] = []

    def delete(self, key: str) -> None:
        if self.fail or key in self.fail_keys:
            raise ObjectStoreError("对象存储不可用")
        self.deleted.append(key)


class _FakeRepo:
    def __init__(self, docs: dict[str, Any] | None = None) -> None:
        self.docs = docs or {}
        self.deleting: list[str] = []
        self.deleted: list[str] = []
        self.audits: list[dict[str, Any]] = []

    async def get_document(self, doc_id: str) -> Any:
        return self.docs.get(doc_id)

    async def mark_document_deleting(self, doc_id: str) -> None:
        self.deleting.append(doc_id)

    async def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        self.docs.pop(doc_id, None)

    async def list_document_ids_by_owner(
        self, *, kb_id: str, tenant_id: str, owner: str
    ) -> list[str]:
        return [doc_id for doc_id, info in self.docs.items() if info.owner == owner]

    async def list_deleting_document_ids(
        self, *, kb_id: str, tenant_id: str, limit: int = 100
    ) -> list[str]:
        return list(self.deleting)[:limit]

    async def write_audit(self, **kwargs: Any) -> None:
        self.audits.append(kwargs)


def _kb() -> KBConfig:
    return KBConfig(kb_id="kb-1", kb_code="demo", name="演示库")


def _doc(doc_id: str, *, owner: str = "alice") -> Any:
    return SimpleNamespace(
        doc_id=doc_id,
        kb_id="kb-1",
        tenant_id="default",
        raw_object_key=f"raw/{doc_id}.md",
        owner=owner,
    )


def _service(
    *, pipeline: _FakePipeline | None = None, object_store: _FakeObjectStore | None = None
) -> tuple[DocumentDeletionService, _FakePipeline, _FakeObjectStore, _FakeRepo]:
    repo = _FakeRepo({"doc-1": _doc("doc-1"), "doc-2": _doc("doc-2")})
    pipe = pipeline or _FakePipeline()
    store = object_store or _FakeObjectStore()
    return (
        DocumentDeletionService(pipeline=pipe, repo=repo, object_store=store),
        pipe,
        store,
        repo,
    )


def test_delete_runs_state_machine_in_order() -> None:
    service, pipe, store, repo = _service()

    outcome = asyncio.run(service.delete(_kb(), "doc-1", user_id="alice"))

    assert repo.deleting == ["doc-1"]
    # DELETING 落库后立即失效检索缓存（v2.1 P0-3：删除开始即对检索不可见）
    assert pipe.cache_invalidated == [_kb().kb_id]
    assert pipe.deleted == ["doc-1"]
    assert store.deleted == ["raw/doc-1.md"]
    assert repo.deleted == ["doc-1"]
    assert outcome.metadata_deleted is True
    assert len(repo.audits) == 1
    assert repo.audits[0]["action"] == "document.delete"


def test_object_delete_failure_keeps_metadata_for_retry() -> None:
    """原始文件删不掉时绝不能删 PG 元数据，否则原件会失去追踪。"""
    service, pipe, _store, repo = _service(object_store=_FakeObjectStore(fail=True))

    with pytest.raises(ObjectStoreError):
        asyncio.run(service.delete(_kb(), "doc-1", user_id="alice"))

    assert pipe.deleted == ["doc-1"]
    # 关键断言：元数据仍在（未调用 delete_document），供 resume_pending 重试
    assert repo.deleted == []
    assert repo.audits[-1]["action"] == "document.delete_failed"
    assert "doc-1" in repo.docs


def test_index_delete_failure_keeps_metadata() -> None:
    service, _pipe, _store, repo = _service(pipeline=_FakePipeline(fail=True))

    with pytest.raises(ObjectStoreError):
        asyncio.run(service.delete(_kb(), "doc-1", user_id="alice"))

    assert repo.deleted == []
    assert repo.audits[-1]["detail"]["stage"] == "index"


def test_batch_delete_uses_same_service_and_continues_after_failure() -> None:
    # doc-2 的原始文件删除失败，不应阻断 doc-1 的删除
    service, pipe, _store, _repo = _service(
        object_store=_FakeObjectStore(fail_keys={"raw/doc-2.md"})
    )

    outcome = asyncio.run(service.purge_owner(_kb(), "alice", user_id="alice"))

    assert outcome.deleted_documents == 1
    assert outcome.failures == {"doc-2": "对象存储不可用"}
    assert pipe.deleted == ["doc-1", "doc-2"]


def test_delete_carrier_migrates_duplicate_and_repoints_lineage() -> None:
    """回归（用户复现）：A 承载索引、B 是重复来源，删除 A 后 B 不能只剩元数据。

    删除承载者时必须把 B 重灌为新的物理承载者，并把其余 duplicate 改指 B，
    否则 B 的元数据还在、实际索引已经消失。
    """
    content_hash = "hash-x"
    carrier = SimpleNamespace(
        doc_id="doc-a", kb_id="kb-1", tenant_id="default", owner="alice",
        raw_object_key="raw/doc-a.md", content_hash=content_hash,
        duplicate_of=None, chunk_count=3, acl_groups=["finance"],
        source_uri="kb://kb-1/a.txt", title="A", version=1,
        source_id="src-a", source_type="file", source_version="1",
        authority_level=5, valid_from=None, valid_to=None,
    )
    duplicate = SimpleNamespace(
        doc_id="doc-b", kb_id="kb-1", tenant_id="default", owner="bob",
        raw_object_key=None, content_hash=content_hash,
        duplicate_of="doc-a", chunk_count=0, acl_groups=["hr"],
        source_uri="kb://kb-1/b.txt", title="B", version=1,
        source_id="src-b", source_type="file", source_version="1",
        authority_level=3, valid_from=None, valid_to=None,
    )
    lineage = [carrier, duplicate]

    class Repo(_FakeRepo):
        def __init__(self) -> None:
            super().__init__({doc.doc_id: doc for doc in lineage})
            self.ready: list[dict[str, Any]] = []
            self.repointed: list[tuple[str, str]] = []

        async def list_lineage_documents(
            self, *, kb_id: str, tenant_id: str, content_hashes: list[str]
        ) -> dict[str, list[Any]]:
            return {content_hash: list(lineage)}

        async def get_chunk_manifest(self, doc_id: str) -> list[dict[str, Any]]:
            return []

        async def mark_document_ready(self, doc_id: str, **kwargs: Any) -> None:
            self.ready.append({"doc_id": doc_id, **kwargs})

        async def repoint_duplicates(
            self, *, old_carrier_doc_id: str, new_carrier_doc_id: str
        ) -> int:
            self.repointed.append((old_carrier_doc_id, new_carrier_doc_id))
            return 1

    class Pipe(_FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.ingested: list[dict[str, Any]] = []
            self.refreshed: list[tuple[str, list[str]]] = []

        def ingest_bytes(self, kb: KBConfig, **kwargs: Any) -> Any:
            self.ingested.append(kwargs)
            return SimpleNamespace(
                chunk_count=3, collection="kb-1", index="kb-1_chunks",
                raw_object_key="raw/doc-b.md", version=1, content_hash=content_hash,
                quality={}, collected_at=None,
                chunk_manifest=[{"chunk_id": "c1"}, {"chunk_id": "c2"}],
            )

        def refresh_lineage_acl(
            self, kb: KBConfig, *, carrier_doc_id: str, merged_acl: list[str]
        ) -> int:
            self.refreshed.append((carrier_doc_id, merged_acl))
            return 0

    repo, pipe = Repo(), Pipe()
    store = _FakeObjectStore()
    store.get_bytes = lambda key: b"same-content"  # type: ignore[method-assign]
    service = DocumentDeletionService(pipeline=pipe, repo=repo, object_store=store)

    # 承载者原文尚在：B 没有自己的归档时退回使用承载者归档（内容指纹相同）
    asyncio.run(service.delete(_kb(), "doc-a", user_id="alice"))

    assert pipe.ingested and pipe.ingested[0]["source_uri"] == "kb://kb-1/b.txt"
    assert pipe.ingested[0]["acl_groups"] == ["hr"]
    assert repo.ready[0]["doc_id"] == "doc-b"
    assert repo.ready[0]["clear_duplicate"] is True
    assert repo.repointed == [("doc-a", "doc-b")]
    assert pipe.refreshed[0][0] == "doc-b"


def test_delete_duplicate_shrinks_carrier_acl() -> None:
    """回归（用户复现）：A 私有、B 公开，合并后索引公开；删除 B 必须收缩回私有。"""
    content_hash = "hash-y"
    carrier = SimpleNamespace(
        doc_id="doc-a", kb_id="kb-1", tenant_id="default", owner="alice",
        raw_object_key="raw/doc-a.md", content_hash=content_hash,
        duplicate_of=None, chunk_count=3, acl_groups=["finance"],
        source_uri="kb://kb-1/a.txt", title="A", version=1,
        source_id="src-a", source_type="file", source_version="1",
        authority_level=5, valid_from=None, valid_to=None,
    )
    duplicate = SimpleNamespace(
        doc_id="doc-b", kb_id="kb-1", tenant_id="default", owner="bob",
        raw_object_key="raw/doc-b.md", content_hash=content_hash,
        duplicate_of="doc-a", chunk_count=0, acl_groups=None,
        source_uri="kb://kb-1/b.txt", title="B", version=1,
        source_id="src-b", source_type="file", source_version="1",
        authority_level=3, valid_from=None, valid_to=None,
    )
    lineage = [carrier, duplicate]

    class Repo(_FakeRepo):
        def __init__(self) -> None:
            super().__init__({doc.doc_id: doc for doc in lineage})

        async def list_lineage_documents(
            self, *, kb_id: str, tenant_id: str, content_hashes: list[str]
        ) -> dict[str, list[Any]]:
            return {content_hash: list(lineage)}

    class Pipe(_FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.refreshed: list[tuple[str, list[str]]] = []

        def refresh_lineage_acl(
            self, kb: KBConfig, *, carrier_doc_id: str, merged_acl: list[str]
        ) -> int:
            self.refreshed.append((carrier_doc_id, merged_acl))
            return 1

    repo, pipe = Repo(), Pipe()
    service = DocumentDeletionService(pipeline=pipe, repo=repo, object_store=_FakeObjectStore())

    asyncio.run(service.delete(_kb(), "doc-b", user_id="bob"))

    # 承载者 A 仍在，但并集 ACL 只剩 A 的私有组 → 索引可见性收缩
    assert pipe.refreshed == [("doc-a", ["finance"])]
