"""保留期清理任务测试：用假资源替换 worker 单例，不依赖 PG/Milvus/ES。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.tasks import maintenance


class _FakeRepo:
    """记录过期文档查询与删除调用；返回按知识库预置的过期 doc_id。"""

    def __init__(self, expired: dict[str, list[str]]) -> None:
        self._expired = expired
        self.calls: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.deleting: list[str] = []
        self.audits: list[dict[str, Any]] = []

    async def list_expired_document_ids(
        self, *, kb_id: str, tenant_id: str, cutoff: datetime, limit: int = 500
    ) -> list[str]:
        self.calls.append({"kb_id": kb_id, "tenant_id": tenant_id, "cutoff": cutoff,
                           "limit": limit})
        return self._expired.get(kb_id, [])[:limit]

    async def get_document(self, doc_id: str) -> Any:
        return SimpleNamespace(
            doc_id=doc_id,
            kb_id="kb-a",
            tenant_id="default",
            raw_object_key=f"raw/{doc_id}.md",
        )

    async def mark_document_deleting(self, doc_id: str) -> None:
        self.deleting.append(doc_id)

    async def list_document_ids_by_owner(
        self, *, kb_id: str, tenant_id: str, owner: str
    ) -> list[str]:
        return []

    async def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)

    async def write_audit(self, **kwargs: Any) -> None:
        self.audits.append(kwargs)


class _FakeObjectStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, key: str) -> None:
        self.deleted.append(key)


class _FakePipeline:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    # v2.1 P0-3：删除服务在 DELETING 落库后会调用缓存失效
    def invalidate_retrieval_cache(self, kb: Any) -> None:
        return None

    def delete_document(self, kb: Any, doc_id: str) -> Any:
        self.deleted.append((kb.kb_code, doc_id))
        return SimpleNamespace(doc_id=doc_id, deleted_vectors=1, deleted_keywords=1)


class _FakeResources:
    def __init__(self, kbs: list[Any], expired: dict[str, list[str]]) -> None:
        self.kb_registry = SimpleNamespace(list_enabled=lambda: list(kbs))
        self.pipeline = _FakePipeline()
        self.repo = _FakeRepo(expired)
        self.object_store = _FakeObjectStore()

    def run(self, coro: Any) -> Any:
        return asyncio.run(coro)


@pytest.fixture()
def resources(monkeypatch: pytest.MonkeyPatch) -> _FakeResources:
    kbs = [
        SimpleNamespace(kb_id="kb-a", tenant_id="default", kb_code="a"),
        SimpleNamespace(kb_id="kb-b", tenant_id="default", kb_code="b"),
    ]
    fake = _FakeResources(kbs, {"kb-a": ["doc-old-1", "doc-old-2"]})
    monkeypatch.setattr(maintenance, "_get_resources", lambda: fake)
    return fake


def test_retention_disabled_skips_without_touching_repo(resources: _FakeResources) -> None:
    """保留期为 0/负数时直接跳过，不扫描也不删除任何文档。"""
    result = maintenance.purge_expired_documents_task.apply(
        kwargs={"retention_days": 0}
    ).get()

    assert result["skipped"] is True
    assert result["deleted"] == 0
    assert resources.repo.calls == []


def test_retention_purges_each_enabled_kb(resources: _FakeResources) -> None:
    """按知识库逐库扫描过期文档，先删索引再删元数据，并回传 cutoff。"""
    result = maintenance.purge_expired_documents_task.apply(
        kwargs={"retention_days": 30, "limit": 10}
    ).get()

    assert result["skipped"] is False
    assert result["deleted"] == 2
    cutoff = datetime.fromisoformat(str(result["cutoff"]))
    assert cutoff <= datetime.now(UTC) - timedelta(days=29)

    # 两个知识库都会被扫描，但只有 kb-a 有过期文档
    assert [call["kb_id"] for call in resources.repo.calls] == ["kb-a", "kb-b"]
    assert all(call["limit"] == 10 for call in resources.repo.calls)

    # 状态机顺序：先落 DELETING，再清索引/原文，最后删元数据
    assert resources.repo.deleting == ["doc-old-1", "doc-old-2"]
    assert [doc_id for _, doc_id in resources.pipeline.deleted] == ["doc-old-1", "doc-old-2"]
    assert resources.object_store.deleted == ["raw/doc-old-1.md", "raw/doc-old-2.md"]
    assert resources.repo.deleted == ["doc-old-1", "doc-old-2"]
    assert len(resources.repo.audits) == 2
