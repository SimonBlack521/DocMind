"""混合检索接线：自适应 K、Embedding 缓存、精排上限与按需跳过（9.3 / 9.7）。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.domain.dto import RetrievedChunk
from app.domain.kb_config import RetrievalConfig
from app.retrieval.embedding_cache import QueryEmbeddingCache
from app.retrieval.hybrid import HybridRetriever
from app.storage.keyword_base import KeywordHit
from app.storage.vector_base import VectorHit


class _VectorStore:
    def __init__(self, hits: Sequence[VectorHit]) -> None:
        self._hits = list(hits)
        self.top_ks: list[int] = []

    def search(self, vector: Sequence[float], *, top_k: int, expr: str | None = None):
        self.top_ks.append(top_k)
        return self._hits[:top_k]


class _KeywordStore:
    def search(self, query: str, **kwargs: Any):
        return []

    def get_by_ids(self, chunk_ids: Sequence[str]) -> list[KeywordHit]:
        return []


class _Embedding:
    model = "fake-embedding"
    dim = 4

    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [0.0] * self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [[0.0] * self.dim for _ in texts]


class _Reranker:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def rerank(self, query: str, chunks: list[RetrievedChunk], *, top_n: int = 5):
        self.calls.append(len(chunks))
        return chunks[:top_n]


class _Rewriter:
    def rewrite(self, question: str) -> list[str]:
        return [question]


def _hits(count: int, *, score: float = 0.9) -> list[VectorHit]:
    return [
        VectorHit(
            chunk_id=f"c{index}",
            score=score - index * 0.01,
            content=f"内容-{index}",
            doc_id=f"doc-{index}",
            source_uri=f"kb://{index}",
        )
        for index in range(count)
    ]


def _retriever(
    *,
    config: RetrievalConfig,
    hits: Sequence[VectorHit],
    embedding: _Embedding | None = None,
    embedding_cache: QueryEmbeddingCache | None = None,
    rerank_max_candidates: int = 0,
    rerank_skip_confidence: float = 0.0,
) -> tuple[HybridRetriever, _VectorStore, _Reranker, _Embedding]:
    vector_store = _VectorStore(hits)
    reranker = _Reranker()
    embedder = embedding or _Embedding()
    retriever = HybridRetriever(
        vector_store=vector_store,  # type: ignore[arg-type]
        keyword_store=_KeywordStore(),  # type: ignore[arg-type]
        embedding=embedder,  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
        rewriter=_Rewriter(),  # type: ignore[arg-type]
        tenant_id="default",
        kb_id="kb-1",
        config=config,
        embedding_cache=embedding_cache,
        embedding_version="v1",
        rerank_max_candidates=rerank_max_candidates,
        rerank_skip_confidence=rerank_skip_confidence,
    )
    return retriever, vector_store, reranker, embedder


def test_adaptive_top_k_applies_only_when_caller_omits_top_k() -> None:
    """未显式传 top_k 时按问题复杂度自适应；显式传值时尊重调用方。"""
    retriever, store, _, _ = _retriever(
        config=RetrievalConfig(
            use_query_rewrite=False, use_rerank=False, adaptive_top_k=True
        ),
        hits=_hits(30),
    )
    retriever.retrieve("年假几天")
    retriever.retrieve("年假几天", top_k=5)
    assert store.top_ks == [10, 5]


def test_query_embedding_cache_avoids_repeated_model_calls() -> None:
    cache = QueryEmbeddingCache(ttl_s=3600)
    retriever, _, _, embedder = _retriever(
        config=RetrievalConfig(use_query_rewrite=False, use_rerank=False),
        hits=_hits(3),
        embedding_cache=cache,
    )
    retriever.retrieve("报销流程")
    retriever.retrieve("报销流程")
    assert embedder.calls == 1


def test_high_confidence_simple_query_skips_rerank() -> None:
    retriever, _, reranker, _ = _retriever(
        config=RetrievalConfig(use_query_rewrite=False, use_rerank=True, rerank_top_n=2),
        hits=_hits(5, score=0.95),
        rerank_skip_confidence=0.8,
    )
    retriever.retrieve("年假几天")
    assert reranker.calls == []
    # 复杂问题不跳过
    retriever.retrieve("对比新旧制度，分别说明申请条件、流程以及时间限制的差异")
    assert reranker.calls


def test_rerank_failure_degrades_to_rrf_order() -> None:
    """精排服务故障 / 推理异常不应让请求失败，应降级为 RRF 排序继续返回结果。"""
    from app.core.errors import ModelUnavailableError

    class _BrokenReranker(_Reranker):
        def rerank(self, query: str, chunks: list[RetrievedChunk], *, top_n: int = 5):
            raise ModelUnavailableError("rerank service down")

    retriever, _, _, _ = _retriever(
        config=RetrievalConfig(use_query_rewrite=False, use_rerank=True, rerank_top_n=2),
        hits=_hits(5, score=0.6),
        rerank_skip_confidence=0,
    )
    retriever._reranker = _BrokenReranker()  # type: ignore[assignment]
    chunks = retriever.retrieve("年假几天")
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "c0"  # RRF 原顺序


def test_rerank_candidate_pool_is_capped() -> None:
    retriever, _, reranker, _ = _retriever(
        config=RetrievalConfig(use_query_rewrite=False, use_rerank=True, rerank_top_n=2, top_k=5),
        hits=_hits(8),
        rerank_max_candidates=3,
    )
    retriever.retrieve("年假几天")
    assert reranker.calls == [3]
