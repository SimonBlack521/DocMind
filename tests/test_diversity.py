"""候选去重与 MMR 多样性（9.5）。"""

from __future__ import annotations

from app.domain.dto import RetrievedChunk
from app.retrieval.diversity import dedupe_chunks, mmr_select


def _chunk(
    chunk_id: str,
    content: str,
    *,
    doc_id: str = "doc-1",
    score: float = 0.8,
    char_start: int | None = None,
    char_end: int | None = None,
) -> RetrievedChunk:
    metadata: dict[str, object] = {}
    if char_start is not None and char_end is not None:
        metadata["char_start"] = char_start
        metadata["char_end"] = char_end
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=score,
        doc_id=doc_id,
        source_uri=f"kb://{doc_id}",
        metadata=metadata,
    )


def test_dedupe_by_chunk_id_span_and_content() -> None:
    """chunk_id、同文档原文区间、正文指纹三级去重，保留排名靠前者。"""
    chunks = [
        _chunk("a", "报销流程", doc_id="doc-1", char_start=0, char_end=10),
        _chunk("a", "报销流程", doc_id="doc-1", char_start=0, char_end=10),
        _chunk("b", "报销流程", doc_id="doc-1", char_start=0, char_end=10),
        _chunk("c", "报销流程", doc_id="doc-2"),
        _chunk("d", "完全不同的内容", doc_id="doc-2"),
    ]
    kept = dedupe_chunks(chunks)
    assert [chunk.chunk_id for chunk in kept] == ["a", "d"]


def test_mmr_prefers_diverse_candidate() -> None:
    """两个近乎重复的报销片段 + 一个请假片段，top_n=2 时应选一个报销 + 一个请假。"""
    chunks = [
        _chunk("a", "报销流程 报销审批 报销材料", score=0.9),
        _chunk("b", "报销流程 报销审批 报销材料 补充说明", score=0.89),
        _chunk("c", "请假流程 年假 事假 审批", score=0.88),
    ]
    selected = mmr_select("报销和请假流程", chunks, top_n=2, lambda_=0.5)
    ids = [chunk.chunk_id for chunk in selected]
    assert "a" in ids
    assert "c" in ids


def test_mmr_lambda_one_is_pure_relevance() -> None:
    """λ=1 退化为纯相关性截断，不改变原排序。"""
    chunks = [_chunk("a", "重复内容", score=0.9), _chunk("b", "重复内容", score=0.8)]
    selected = mmr_select("问题", chunks, top_n=1, lambda_=1.0)
    assert [chunk.chunk_id for chunk in selected] == ["a"]
