"""RRF 融合测试：chunk_id 去重、权重、稳定性。"""

from __future__ import annotations

import pytest

from app.core.errors import ConfigurationError
from app.domain.dto import RetrievedChunk
from app.retrieval.rrf import reciprocal_rank_fusion


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"内容-{chunk_id}",
        score=1.0,
        doc_id="doc-1",
        source_uri="kb://finance/a.txt",
    )


def test_fusion_dedupes_by_chunk_id_and_accumulates_score() -> None:
    vector_hits = [_chunk("a"), _chunk("b"), _chunk("c")]
    keyword_hits = [_chunk("b"), _chunk("d")]

    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], k=60)

    assert [chunk.chunk_id for chunk in fused] == ["b", "a", "d", "c"]
    # b 在向量路 rank=1、BM25 路 rank=0，两路得分累加
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)


def test_duplicate_in_same_channel_counts_once() -> None:
    fused = reciprocal_rank_fusion([[_chunk("a"), _chunk("a"), _chunk("b")]])

    assert len(fused) == 2
    assert fused[0].chunk_id == "a"
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 63)


def test_weights_change_ranking() -> None:
    vector_hits = [_chunk("a"), _chunk("b")]
    keyword_hits = [_chunk("b"), _chunk("a")]

    vector_only = reciprocal_rank_fusion([vector_hits, keyword_hits], weights=[1.0, 0.0])
    keyword_only = reciprocal_rank_fusion([vector_hits, keyword_hits], weights=[0.0, 1.0])

    assert [chunk.chunk_id for chunk in vector_only] == ["a", "b"]
    assert [chunk.chunk_id for chunk in keyword_only] == ["b", "a"]


def test_invalid_arguments_raise_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        reciprocal_rank_fusion([[_chunk("a")]], weights=[1.0, 1.0])
    with pytest.raises(ConfigurationError):
        reciprocal_rank_fusion([[_chunk("a")]], k=0)
