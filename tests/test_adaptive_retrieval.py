"""自适应召回与分数断层早停（9.3 / 9.7）。"""

from __future__ import annotations

from app.domain.dto import RetrievedChunk
from app.retrieval.adaptive import (
    estimate_query_complexity,
    resolve_dynamic_top_k,
    truncate_at_score_cliff,
)


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"内容-{chunk_id}",
        score=score,
        doc_id=f"doc-{chunk_id}",
        source_uri=f"kb://{chunk_id}",
    )


def test_complexity_rewards_multi_hop_markers() -> None:
    """多跳/流程类问题复杂度更高，短 FAQ 问题复杂度更低。"""
    simple = estimate_query_complexity("年假多少天")
    complex_query = estimate_query_complexity(
        "对比新旧报销制度，分别说明申请条件、流程以及时间限制的差异"
    )
    assert simple < complex_query
    assert 0.0 <= simple <= 1.0
    assert 0.0 <= complex_query <= 1.0


def test_dynamic_top_k_small_for_simple_and_large_for_complex() -> None:
    """简单问题取 base_k/2，复杂问题放大到 base_k×2，中间保持基准值。"""
    simple = resolve_dynamic_top_k("年假几天", base_k=20, min_k=5, max_k=40)
    complex_query = resolve_dynamic_top_k(
        "对比新旧报销制度，分别说明申请条件、流程以及时间限制的差异",
        base_k=20,
        min_k=5,
        max_k=40,
    )
    assert simple == 10
    assert complex_query == 40


def test_dynamic_top_k_respects_bounds() -> None:
    """上下限参与夹取；未配置边界时保持 base_k。"""
    assert resolve_dynamic_top_k("年假几天", base_k=20, min_k=12, max_k=40) == 12
    assert resolve_dynamic_top_k("问题", base_k=20) == 20
    assert resolve_dynamic_top_k("问题", base_k=0) == 0


def test_score_cliff_stops_before_noise() -> None:
    """前两名分数很高、第三名断崖下跌时，只保留断层之前的候选。"""
    hits = [_chunk("a", 0.9), _chunk("b", 0.86), _chunk("c", 0.2), _chunk("d", 0.18)]
    kept = truncate_at_score_cliff(hits, min_keep=2, ratio=0.5)
    assert [chunk.chunk_id for chunk in kept] == ["a", "b"]


def test_score_cliff_keeps_flat_results_and_disabled_mode() -> None:
    """分数平缓时不做截断；ratio=0 时完全关闭早停。"""
    hits = [_chunk("a", 0.9), _chunk("b", 0.88), _chunk("c", 0.86)]
    assert [chunk.chunk_id for chunk in truncate_at_score_cliff(hits, min_keep=1, ratio=0.5)] == [
        "a",
        "b",
        "c",
    ]
    assert truncate_at_score_cliff(hits, min_keep=1, ratio=0.0) == hits
