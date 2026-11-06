"""评测指标测试：检索/证据/冗余/成本/拒答五组口径的纯函数校验。"""

from __future__ import annotations

import pytest

from app.domain.dto import RetrievedChunk
from app.evaluation.metrics import (
    cost_metrics,
    estimate_tokens,
    evidence_metrics,
    ndcg_at_k,
    p95,
    recall_at_k,
    reciprocal_rank,
    redundancy_metrics,
    refusal_quality,
    summarize_refusal,
)


def _chunk(
    chunk_id: str,
    content: str,
    *,
    doc_id: str = "doc-1",
    start: int | None = None,
    end: int | None = None,
    content_hash: str = "",
    section_path: str = "制度 > 小节",
) -> RetrievedChunk:
    metadata: dict[str, object] = {"section_path": section_path}
    if start is not None:
        metadata["char_start"] = start
    if end is not None:
        metadata["char_end"] = end
    if content_hash:
        metadata["content_hash"] = content_hash
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=0.5,
        doc_id=doc_id,
        source_uri=f"kb://{doc_id}",
        metadata=metadata,
    )


def test_recall_mrr_and_ndcg() -> None:
    """检索指标：命中位置越靠前，MRR/nDCG 越高；K 截断生效。"""
    flags = [False, False, True, False]
    assert recall_at_k(flags, 1) == 0.0
    assert recall_at_k(flags, 5) == 1.0
    assert reciprocal_rank(flags) == 1 / 3
    assert ndcg_at_k(flags, 1) < ndcg_at_k(flags, 5)
    # 相关项在理想排序的第 1 位时 nDCG=1
    assert ndcg_at_k([True, False], 2) == 1.0


def test_recall_dedups_repeated_chunks_of_same_document() -> None:
    """同一文档的多个 chunk 进前 K 时只能算一次命中，否则 recall 会超过 1。

    真实评测（1000 篇/9000 chunk）中出现过 recall@5=1.145：单篇文档 9 个 chunk
    同时进入前 5，chunk 级计数把命中数放大到分母的 1.145 倍。
    """
    flags = [True, True, True, False, True]
    ids = ["doc-a", "doc-a", "doc-a", "doc-b", "doc-b"]
    # 前 5 命中 doc-a（3 个 chunk）与 doc-b（1 个 chunk），去重后为 2/2
    assert recall_at_k(flags, 5, golden_total=2, relevance_ids=ids) == 1.0
    # 前 3 只有 doc-a 的 chunk，去重后为 1/2
    assert recall_at_k(flags, 3, golden_total=2, relevance_ids=ids) == 0.5
    # 去重后不会超过 1
    assert recall_at_k(flags, 5, golden_total=1, relevance_ids=ids) == 1.0

    # nDCG 同样去重：重复 chunk 不再叠加收益
    dedup = ndcg_at_k(flags, 5, golden_total=2, relevance_ids=ids)
    assert dedup <= 1.0
    assert dedup == ndcg_at_k([True, False, False, False, True], 5, golden_total=2)


def test_evidence_metrics_detects_complete_and_truncated_evidence() -> None:
    """证据完整率与边界截断率：完整证据与"只剩半句"的证据要能区分。"""
    chunks = [
        _chunk("c1", "一线城市住宿每晚不超过500元，二线城市不超过400元。"),
        _chunk("c2", "报销需在出差结束后10个工作日内提交，", start=0, end=20),
        _chunk("c3", "逾期需部门负责人签字确认。", start=20, end=40),
    ]
    metrics = evidence_metrics(
        chunks,
        ["一线城市住宿每晚不超过500元", "报销需在出差结束后10个工作日内提交报销单"],
    )
    assert metrics["evidence_completeness"] == 0.5
    assert metrics["boundary_truncation_rate"] == 0.5
    assert metrics["citation_locatable_rate"] == 1.0
    # c2/c3 区间连续（end=20, start=20），存在相邻依赖
    assert metrics["adjacent_dependency_rate"] > 0


def test_redundancy_metrics_three_directions() -> None:
    """冗余三项：重复内容、同文档集中度、重复 token。"""
    chunks = [
        _chunk("c1", "报销时限 10 个工作日", content_hash="same"),
        _chunk("c2", "报销时限 10 个工作日", content_hash="same"),
        _chunk("c3", "住宿标准每人每晚 500 元", doc_id="doc-2", content_hash="other"),
    ]
    metrics = redundancy_metrics(chunks)
    assert metrics["duplicate_chunk_ratio"] == pytest.approx(1 / 3)
    assert metrics["same_doc_candidate_ratio"] == pytest.approx(2 / 3)
    assert 0 < metrics["duplicate_token_ratio"] < 1


def test_cost_metrics_and_p95() -> None:
    """成本指标：chunk 数、向量字节、上下文 token、P95 耗时。"""
    chunks = [_chunk("c1", "甲" * 150), _chunk("c2", "乙" * 300)]
    metrics = cost_metrics(chunks, vector_dim=1024, latencies_ms=[1, 2, 3])
    assert metrics["chunk_count"] == 2.0
    assert metrics["vector_storage_bytes_per_chunk"] == 4096.0
    expected_tokens = estimate_tokens("甲" * 150) + estimate_tokens("乙" * 300)
    assert metrics["context_token_estimate"] == expected_tokens
    assert metrics["retrieval_p95_ms"] == 3.0
    assert estimate_tokens("") == 0
    assert p95([]) == 0.0


def test_refusal_quality_four_quadrants() -> None:
    """拒答质量四分类覆盖：该答且答/该拒且拒/误拒/幻觉。"""
    assert refusal_quality(predicted_refusal=False, expected_no_answer=False) == "correct_answer"
    assert refusal_quality(predicted_refusal=True, expected_no_answer=True) == "correct_refusal"
    assert refusal_quality(predicted_refusal=True, expected_no_answer=False) == "false_refusal"
    assert refusal_quality(predicted_refusal=False, expected_no_answer=True) == "hallucination"
    assert summarize_refusal(["correct_answer", "hallucination"]) == {
        "correct_answer": 1,
        "correct_refusal": 0,
        "false_refusal": 0,
        "hallucination": 1,
    }
