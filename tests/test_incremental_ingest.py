"""S2 增量索引：Manifest Diff 向量复用、先写后删、版本去重、索引对账。

全部用内存 fake（不连 Milvus/ES/PG），验证编排逻辑本身。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Collection
from typing import Any

import pytest

from app.core.settings import Settings
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import ChunkingConfig, KBConfig
from app.ingestion.loader import LoadedDocument
from app.orchestration.pipeline import RAGPipeline
from app.orchestration.registry import KBRuntime
from app.retrieval.hybrid import _dedupe_to_latest_version
from app.storage.keyword_base import KeywordHit, KeywordRecord
from app.storage.vector_base import VectorHit, VectorRecord
from app.tasks.maintenance import _reconcile_kb

# ------------------------------------------------------------------ fakes


class _FakeEmbedding:
    dim = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # 向量值与文本绑定，便于断言"复用的是旧向量"
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.records: dict[str, VectorRecord] = {}
        self.calls: list[str] = []

    def ensure_collection(self, dim: int) -> None:
        return None

    def upsert(self, records: list[VectorRecord], *, dim: int) -> int:
        self.calls.append("upsert")
        for record in records:
            self.vectors[record.chunk_id] = list(record.vector)
            self.records[record.chunk_id] = record
        return len(records)

    def get_records_by_doc(self, doc_id: str) -> list[VectorRecord]:
        """整篇快照：补偿逻辑用它区分"新建 vs 覆盖"。"""
        return [record for record in self.records.values() if record.doc_id == doc_id]

    def reconcile_index_params(self) -> int:
        return 0

    def delete_by_doc(self, doc_id: str) -> int:
        self.calls.append("delete_by_doc")
        return 0

    def delete_by_ids(self, chunk_ids: list[str]) -> int:
        self.calls.append("delete_by_ids")
        deleted = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.vectors:
                del self.vectors[chunk_id]
                deleted += 1
        return deleted

    def get_vectors_by_ids(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        return {cid: self.vectors[cid] for cid in chunk_ids if cid in self.vectors}

    def list_chunk_ids(self, *, batch_size: int = 1000) -> Any:
        yield sorted(self.vectors)

    def search(
        self, vector: list[float], *, top_k: int, expr: str | None = None
    ) -> list[VectorHit]:
        return []


class _FakeKeywordStore:
    def __init__(self, *, fail_upsert: bool = False) -> None:
        self.records: dict[str, KeywordRecord] = {}
        self.fail_upsert = fail_upsert
        self.calls: list[str] = []

    def ensure_index(self) -> None:
        return None

    def upsert(self, records: list[KeywordRecord]) -> int:
        self.calls.append("upsert")
        if self.fail_upsert:
            raise RuntimeError("ES 写入失败（模拟）")
        for record in records:
            self.records[record.chunk_id] = record
        return len(records)

    def delete_by_doc(self, doc_id: str) -> int:
        self.calls.append("delete_by_doc")
        return 0

    def delete_by_ids(self, chunk_ids: list[str]) -> int:
        self.calls.append("delete_by_ids")
        deleted = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.records:
                del self.records[chunk_id]
                deleted += 1
        return deleted

    def get_by_ids(self, chunk_ids: list[str]) -> list[KeywordHit]:
        return [
            KeywordHit(chunk_id=cid, doc_id=self.records[cid].doc_id, score=1.0,
                       content=self.records[cid].content,
                       source_uri=self.records[cid].source_uri,
                       metadata=dict(self.records[cid].metadata))
            for cid in chunk_ids
            if cid in self.records
        ]

    def get_records_by_doc(self, doc_id: str) -> list[KeywordRecord]:
        """整篇快照：覆盖写入失败时用它恢复原记录（含父块）。"""
        return [
            record for record in self.records.values() if record.doc_id == doc_id
        ]

    def list_chunk_ids(self, *, batch_size: int = 1000) -> Any:
        yield sorted(self.records)


class _FakeRegistry:
    def __init__(self, runtime: KBRuntime) -> None:
        self._runtime = runtime

    def runtime(self, kb: KBConfig) -> KBRuntime:
        return self._runtime


# ------------------------------------------------------------------ helpers


def _kb() -> KBConfig:
    return KBConfig(
        kb_id="kb-inc-001",
        kb_code="inc",
        name="增量测试库",
        chunking=ChunkingConfig(
            size=20,  # 每段约 10~16 字：强制一段一块，便于断言 Diff 粒度
            overlap=0,
            separators=["\n\n"],
            min_chunk_len=1,
            structure_aware=False,
            parent_child=False,
        ),
    )


def _doc(content: str, doc_id: str = "doc-1") -> LoadedDocument:
    return LoadedDocument(
        doc_id=doc_id,
        source_uri="kb://inc/a.txt",
        title="a",
        content=content,
        mime_type="text/plain",
        content_hash="h-" + str(len(content)),
    )


def _pipeline() -> tuple[RAGPipeline, _FakeEmbedding, _FakeVectorStore, _FakeKeywordStore]:
    embedding = _FakeEmbedding()
    vector_store = _FakeVectorStore()
    keyword_store = _FakeKeywordStore()
    runtime = KBRuntime(
        config=_kb(),
        embedding=embedding,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
        keyword_store=keyword_store,  # type: ignore[arg-type]
    )
    pipeline = RAGPipeline(
        settings=Settings(),
        registry=_FakeRegistry(runtime),  # type: ignore[arg-type]
    )
    return pipeline, embedding, vector_store, keyword_store


V1 = "第一段讲苹果种植技术与修剪要点\n\n第二段讲柑橘病虫害防治方法\n\n第三段讲水稻育种流程"
V2 = (
    "第一段讲苹果种植技术与修剪要点\n\n第二段讲柑橘病虫害防治方法\n\n"
    "第三段讲小麦育种与田间管理新方法"
)


# ------------------------------------------------------------------ 用例


def test_manifest_diff_reuses_unchanged_vectors() -> None:
    """内容小改升版本：未变段落不重算 Embedding，旧向量按 id 取回复用。"""
    pipeline, embedding, vector_store, _ = _pipeline()
    kb = _kb()

    first = pipeline.ingest_document(kb, _doc(V1), raw_bytes=None, version=1)
    assert first.chunk_count == 3
    assert len(embedding.calls) == 1 and len(embedding.calls[0]) == 3
    old_vectors = dict(vector_store.vectors)

    embedding.calls.clear()
    second = pipeline.ingest_document(
        kb, _doc(V2), raw_bytes=None, version=2, old_manifest=first.chunk_manifest
    )

    assert second.chunk_count == 3
    # 只有变化的第 3 段重算了 Embedding
    assert len(embedding.calls) == 1
    assert len(embedding.calls[0]) == 1
    assert "小麦" in embedding.calls[0][0]

    # 未变的两段复用了旧向量（fake 向量值与文本长度绑定，可跨 chunk_id 比对）
    old_vector_values = {tuple(vector) for vector in old_vectors.values()}
    new_ids = {entry["chunk_id"] for entry in second.chunk_manifest}
    reused = [
        cid for cid in new_ids if tuple(vector_store.vectors[cid]) in old_vector_values
    ]
    assert len(reused) == 2

    # 旧版 chunk 全部被精确删除（不是 delete_by_doc），且先写后删
    assert "delete_by_doc" not in vector_store.calls
    assert vector_store.calls.index("upsert") < vector_store.calls.index("delete_by_ids")
    assert not (set(old_vectors) & set(vector_store.vectors))

    # 新清单带 embedding_version，供下次 Diff 与对账使用
    assert {entry["embedding_version"] for entry in second.chunk_manifest} == {"v1"}


def test_same_version_rerun_embeds_nothing() -> None:
    """同版本重跑（任务重试）：chunk_id 稳定，全部复用，无失效 chunk 可删。"""
    pipeline, embedding, vector_store, _ = _pipeline()
    kb = _kb()
    first = pipeline.ingest_document(kb, _doc(V1), raw_bytes=None, version=1)

    embedding.calls.clear()
    vector_store.calls.clear()
    second = pipeline.ingest_document(
        kb, _doc(V1), raw_bytes=None, version=1, old_manifest=first.chunk_manifest
    )

    assert embedding.calls == []  # 零重算
    assert "delete_by_ids" not in vector_store.calls  # 无失效 id
    assert {entry["chunk_id"] for entry in second.chunk_manifest} == {
        entry["chunk_id"] for entry in first.chunk_manifest
    }


def test_embedding_version_mismatch_forces_recompute() -> None:
    """旧清单来自别的 embedding 版本（如换模型后）：不允许复用，全部重算。"""
    pipeline, embedding, _, _ = _pipeline()
    kb = _kb()
    first = pipeline.ingest_document(kb, _doc(V1), raw_bytes=None, version=1)
    stale_manifest = [
        {**entry, "embedding_version": "v0"} for entry in first.chunk_manifest
    ]

    embedding.calls.clear()
    pipeline.ingest_document(
        kb, _doc(V1), raw_bytes=None, version=2, old_manifest=stale_manifest
    )
    assert len(embedding.calls) == 1 and len(embedding.calls[0]) == 3


def test_write_failure_rolls_back_only_new_chunks() -> None:
    """ES 写失败：按 id 精确回滚 Milvus 本次新写入，旧版本 chunk 不动。"""
    pipeline, _, vector_store, keyword_store = _pipeline()
    kb = _kb()
    first = pipeline.ingest_document(kb, _doc(V1), raw_bytes=None, version=1)
    old_ids = set(vector_store.vectors)

    keyword_store.fail_upsert = True
    with pytest.raises(RuntimeError):
        pipeline.ingest_document(
            kb, _doc(V2), raw_bytes=None, version=2, old_manifest=first.chunk_manifest
        )

    # 新版 chunk 已回滚，旧版完整保留（先写后删：旧版删除在写入成功后才发生）
    assert set(vector_store.vectors) == old_ids


def test_same_version_rebuild_failure_keeps_existing_index() -> None:
    """回归（用户故障注入）：同版本仅改 ACL 重建失败，不能把原有索引删光。

    内容与版本不变 → chunk_id 不变，upsert 是"覆盖"而非"新建"；旧实现按 id
    全量回滚，导致 ES 写失败后原有 3 条向量 + 3 条关键词全部变成 0。
    """
    pipeline, _, vector_store, keyword_store = _pipeline()
    kb = _kb()
    first = pipeline.ingest_document(
        kb, _doc(V1), raw_bytes=None, version=1, acl_groups=["finance"]
    )
    old_ids = set(vector_store.vectors)

    keyword_store.fail_upsert = True
    with pytest.raises(RuntimeError):
        pipeline.ingest_document(
            kb,
            _doc(V1),
            raw_bytes=None,
            version=1,
            acl_groups=["finance", "hr"],
            old_manifest=first.chunk_manifest,
        )

    # 原有索引完整保留：新建 id 为 0，回滚只恢复被覆盖记录，不做删除
    assert set(vector_store.vectors) == old_ids
    assert len(vector_store.vectors) == 3
    assert len(keyword_store.records) == 3


def test_dedupe_to_latest_version() -> None:
    """检索融合后同一文档只保留最高版本，缺版本的历史数据原样保留。"""
    chunks = [
        RetrievedChunk(chunk_id="a2", doc_id="A", score=0.9, content="新",
                       source_uri="u", metadata={"version": 2}),
        RetrievedChunk(chunk_id="a1", doc_id="A", score=0.8, content="旧",
                       source_uri="u", metadata={"version": 1}),
        RetrievedChunk(chunk_id="b1", doc_id="B", score=0.7, content="历史",
                       source_uri="u", metadata={}),
    ]
    kept = _dedupe_to_latest_version(chunks)
    assert [c.chunk_id for c in kept] == ["a2", "b1"]


# ------------------------------------------------------------------ 对账


class _FakeRepo:
    def __init__(
        self, manifests: dict[str, list[dict]], *, inflight: list[str] | None = None
    ) -> None:
        self._manifests = manifests
        self._inflight = inflight or []
        self.ready_calls: list[dict[str, Any]] = []
        self.index_status_calls: list[tuple[list[str], str]] = []

    async def list_inflight_document_ids(
        self, *, kb_id: str, tenant_id: str
    ) -> list[str]:
        return list(self._inflight)

    async def scan_kb_chunks(
        self, *, kb_id: str, batch_size: int = 1000
    ) -> AsyncIterator[list[tuple[str, str, str]]]:
        batch = [
            (doc_id, str(entry.get("chunk_id", "")), str(entry.get("role") or "child"))
            for doc_id, entries in self._manifests.items()
            for entry in entries
        ]
        yield batch

    async def filter_existing_chunk_ids(
        self, *, kb_id: str, chunk_ids: Collection[str]
    ) -> set[str]:
        existing = {
            str(entry.get("chunk_id", ""))
            for entries in self._manifests.values()
            for entry in entries
        }
        return {chunk_id for chunk_id in chunk_ids if chunk_id in existing}

    async def mark_chunks_index_status(
        self, chunk_ids: Collection[str], *, status: str
    ) -> int:
        self.index_status_calls.append((list(chunk_ids), status))
        return len(chunk_ids)


class _FakeResources:
    def __init__(self, runtime: KBRuntime, repo: _FakeRepo) -> None:
        self.runtime_registry = _FakeRegistry(runtime)
        self.repo = repo
        self._loop = asyncio.new_event_loop()

    def run(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)


def _manifest_of(result: Any) -> list[dict]:
    return [dict(entry) for entry in result.chunk_manifest]


def test_reconcile_repair_false_is_read_only() -> None:
    """repair=False 只能检测不能改数据（用户复现过只检模式仍删 Milvus/ES）。

    旧断言把"repair=False 也删孤儿"当作预期，那正是本次要修的缺陷；
    只读模式必须只报告 detected / pending 计数，不产生任何删除与状态写入。
    """
    pipeline, _, vector_store, keyword_store = _pipeline()
    kb = _kb()
    result = pipeline.ingest_document(kb, _doc(V1), raw_bytes=None, version=1)
    manifest = _manifest_of(result)

    # 制造孤儿 + 缺失
    vector_store.vectors["orphan-v"] = [0.0, 0.0, 0.0, 0.0]
    keyword_store.records["orphan-k"] = KeywordRecord(
        chunk_id="orphan-k", doc_id="ghost", kb_id=kb.kb_id, tenant_id=kb.tenant_id,
        content="孤儿关键词记录", source_uri="kb://inc/ghost.txt", version=1, chunk_index=0,
    )
    lost_id = manifest[0]["chunk_id"]
    del vector_store.vectors[lost_id]

    repo = _FakeRepo({result.doc_id: manifest})
    resources = _FakeResources(
        KBRuntime(config=kb, embedding=_FakeEmbedding(),  # type: ignore[arg-type]
                  vector_store=vector_store, keyword_store=keyword_store),  # type: ignore[arg-type]
        repo,
    )
    stats = _reconcile_kb(resources, kb, repair=False)  # type: ignore[arg-type]

    assert stats["orphan_vectors_detected"] == 1
    assert stats["orphan_keywords_detected"] == 1
    assert stats["orphan_vectors_deleted"] == 0
    assert stats["orphan_keywords_deleted"] == 0
    # 只读模式下索引内容不变
    assert "orphan-v" in vector_store.vectors
    assert "orphan-k" in keyword_store.records
    assert stats["missing_vectors"] == 1
    assert stats["docs_pending_repair"] == 1
    assert repo.index_status_calls == []


def test_reconcile_defers_orphans_when_ingest_inflight() -> None:
    """在途入库保护：ingesting 文档存在时，先写索引后提交清单的窗口不能被误删。"""
    pipeline, _, vector_store, keyword_store = _pipeline()
    kb = _kb()
    result = pipeline.ingest_document(kb, _doc(V1), raw_bytes=None, version=1)
    manifest = _manifest_of(result)

    vector_store.vectors["in-flight-v"] = [0.0, 0.0, 0.0, 0.0]
    repo = _FakeRepo({result.doc_id: manifest}, inflight=["doc-inflight"])
    resources = _FakeResources(
        KBRuntime(config=kb, embedding=_FakeEmbedding(),  # type: ignore[arg-type]
                  vector_store=vector_store, keyword_store=keyword_store),  # type: ignore[arg-type]
        repo,
    )
    stats = _reconcile_kb(resources, kb, repair=True)  # type: ignore[arg-type]

    assert stats["orphan_vectors_detected"] == 1
    assert stats["orphans_deferred"] == 1
    assert stats["orphan_vectors_deleted"] == 0
    assert "in-flight-v" in vector_store.vectors
