"""检索治理测试：排除被替代文档（两路过滤下推）+ 权威度/时效加权排序。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from app.domain.dto import RetrievedChunk
from app.domain.kb_config import RetrievalConfig
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ranking import apply_governance_weighting, governance_weight
from app.storage.keyword_base import KeywordHit
from app.storage.vector_base import VectorHit, scope_expression

NOW = datetime(2026, 11, 3, tzinfo=UTC)


def _chunk(
    chunk_id: str,
    *,
    authority_level: int = 0,
    collected_at: datetime | None = None,
) -> RetrievedChunk:
    metadata: dict[str, Any] = {"authority_level": authority_level}
    if collected_at is not None:
        metadata["collected_at"] = collected_at.isoformat()
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"内容-{chunk_id}",
        score=0.5,
        doc_id=f"doc-{chunk_id}",
        source_uri=f"kb://gov/{chunk_id}.md",
        metadata=metadata,
    )


# ---------------------------------------------------------------- 过滤下推
def test_scope_expression_excludes_superseded_documents() -> None:
    """被替代文档的排除条件与 ACL 一起下推到向量库表达式里。"""
    expr = scope_expression(
        "default", "kb-hr", frozenset({"hr"}), ["doc-old", "doc-old"]
    )
    assert 'tenant_id == "default"' in expr
    assert 'kb_id == "kb-hr"' in expr
    assert "json_contains_any" in expr
    # 去重且保持 JSON 数组语法，可被 Milvus 直接解析
    assert 'doc_id not in ["doc-old"]' in expr

    # 没有待排除文档时不产生多余条件
    assert "doc_id not in" not in scope_expression("default", "kb-hr", frozenset({"hr"}))


def test_governance_weight_is_multiplicative_and_neutral_without_metadata() -> None:
    """缺少治理字段的片段权重恒为 1；有字段时权威度与新鲜度相乘。"""
    plain = governance_weight(
        {}, authority_weight=0.2, recency_weight=0.1, half_life_days=180, now=NOW
    )
    assert plain == 1.0

    fresh = governance_weight(
        {"authority_level": 100, "collected_at": NOW.isoformat()},
        authority_weight=0.2,
        recency_weight=0.1,
        half_life_days=180,
        now=NOW,
    )
    assert fresh == (1 + 0.2) * (1 + 0.1)

    stale = governance_weight(
        {"authority_level": 0, "collected_at": (NOW - timedelta(days=180)).isoformat()},
        authority_weight=0.2,
        recency_weight=0.1,
        half_life_days=180,
        now=NOW,
    )
    # 半个半衰期后新鲜度衰减到 0.5
    assert stale == 1 + 0.1 * 0.5


def test_governance_weight_reorders_rrf_candidates() -> None:
    """权威度更高、采集时间更新的片段应排到旧的低权威片段前面。"""
    now = NOW
    stale = _chunk("stale", authority_level=0, collected_at=now - timedelta(days=720))
    authoritative = _chunk("authoritative", authority_level=90, collected_at=now)

    ranked = apply_governance_weighting(
        [stale, authoritative],
        authority_weight=0.2,
        recency_weight=0.1,
        half_life_days=180,
        now=now,
    )
    assert [chunk.chunk_id for chunk in ranked] == ["authoritative", "stale"]
    assert ranked[0].metadata["governance_weight"] > ranked[1].metadata["governance_weight"]


def test_governance_weight_disabled_keeps_order() -> None:
    """两个权重都置 0 时不改变顺序，也不写入 governance_weight。"""
    chunks = [_chunk("a"), _chunk("b")]
    ranked = apply_governance_weighting(
        chunks, authority_weight=0, recency_weight=0, half_life_days=180, now=NOW
    )
    assert [chunk.chunk_id for chunk in ranked] == ["a", "b"]
    assert "governance_weight" not in ranked[0].metadata


# ---------------------------------------------------------------- 混合检索接线
class _FakeVectorStore:
    def __init__(self, hits: Sequence[VectorHit]) -> None:
        self._hits = list(hits)
        self.expressions: list[str | None] = []

    def search(self, vector: Sequence[float], *, top_k: int, expr: str | None = None):
        self.expressions.append(expr)
        return self._hits[:top_k]


class _FakeKeywordStore:
    def __init__(self, hits: Sequence[KeywordHit]) -> None:
        self._hits = list(hits)
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any):
        self.calls.append(kwargs)
        # 与真实 ES 实现一致：检索只返回子块，父块仅通过 get_by_ids 回取
        hits = [hit for hit in self._hits if hit.metadata.get("chunk_role") != "parent"]
        return hits[: kwargs.get("top_k", len(hits))]

    def get_by_ids(self, chunk_ids: Sequence[str]) -> list[KeywordHit]:
        wanted = set(chunk_ids)
        return [hit for hit in self._hits if hit.chunk_id in wanted]


class _FakeEmbedding:
    dim = 8

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


class _FakeReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], *, top_n: int = 5):
        return chunks[:top_n]


class _FakeRewriter:
    def rewrite(self, question: str) -> list[str]:
        return [question]


def _retriever(
    vector_store: _FakeVectorStore, keyword_store: _FakeKeywordStore
) -> HybridRetriever:
    return HybridRetriever(
        vector_store=vector_store,  # type: ignore[arg-type]
        keyword_store=keyword_store,  # type: ignore[arg-type]
        embedding=_FakeEmbedding(),  # type: ignore[arg-type]
        reranker=_FakeReranker(),  # type: ignore[arg-type]
        rewriter=_FakeRewriter(),  # type: ignore[arg-type]
        tenant_id="default",
        kb_id="kb-hr",
        config=RetrievalConfig(use_query_rewrite=False, use_rerank=False, top_k=5),
        authority_weight=0.2,
        recency_weight=0.1,
    )


def test_hybrid_retriever_pushes_exclusions_to_both_channels() -> None:
    """排除集合必须同时下推到向量表达式与 ES 查询，而不是召回后再丢弃。"""
    vector_hit = VectorHit(
        chunk_id="c1", score=0.9, content="正文", doc_id="doc-1", source_uri="kb://a"
    )
    keyword_hit = KeywordHit(
        chunk_id="c2", score=3.2, content="正文", doc_id="doc-2", source_uri="kb://b"
    )
    vector_store = _FakeVectorStore([vector_hit])
    keyword_store = _FakeKeywordStore([keyword_hit])
    retriever = _retriever(vector_store, keyword_store)

    retriever.retrieve("报销流程", excluded_doc_ids=frozenset({"doc-old"}))

    assert vector_store.expressions
    assert 'doc_id not in ["doc-old"]' in str(vector_store.expressions[0])
    assert keyword_store.calls[0]["excluded_doc_ids"] == frozenset({"doc-old"})

    # 不传排除集合时两路都不应出现排除条件
    vector_store.expressions.clear()
    keyword_store.calls.clear()
    retriever.retrieve("报销流程")
    assert "doc_id not in" not in str(vector_store.expressions[0])
    assert keyword_store.calls[0]["excluded_doc_ids"] is None


def test_hybrid_retriever_expands_parent_and_dedupes_by_parent() -> None:
    """父子切片：命中子块后回取父块；同一父块下的多个子块只保留排名靠前的一个。"""
    child_a = _chunk("child-a")
    child_b = _chunk("child-b")
    child_a.metadata["parent_chunk_id"] = "parent-x"
    child_b.metadata["parent_chunk_id"] = "parent-x"
    parent = KeywordHit(
        chunk_id="parent-x",
        score=0.0,
        content="整节内容：报销时限与住宿标准",
        doc_id="doc-a",
        source_uri="kb://a",
        metadata={"chunk_role": "parent"},
    )
    vector_store = _FakeVectorStore(
        [
            VectorHit(
                chunk_id="child-a",
                score=0.9,
                content="子块A",
                doc_id="doc-a",
                source_uri="kb://a",
                metadata=child_a.metadata,
            ),
            VectorHit(
                chunk_id="child-b",
                score=0.8,
                content="子块B",
                doc_id="doc-a",
                source_uri="kb://a",
                metadata=child_b.metadata,
            ),
        ]
    )
    keyword_store = _FakeKeywordStore([parent])
    retriever = _retriever(vector_store, keyword_store)

    chunks = retriever.retrieve("报销时限")

    assert len(chunks) == 1, "同一父块的两个子块应去重，避免重复上下文"
    assert chunks[0].chunk_id == "child-a"
    assert chunks[0].metadata["parent_content"] == parent.content
