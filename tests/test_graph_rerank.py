"""图增强重排测试（v2.0）：默认关闭、图中心度提权、分数不被修改。"""

from __future__ import annotations

import pytest

from app.domain.dto import RetrievedChunk
from app.domain.kb_config import RetrievalConfig
from app.retrieval.graph import graph_enhanced_rerank


def _chunk(chunk_id: str, content: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=score,
        doc_id=f"doc-{chunk_id}",
        source_uri=f"s3://docs/{chunk_id}",
    )


def test_graph_rerank_returns_input_for_single_chunk() -> None:
    chunk = _chunk("a", "预算")
    assert graph_enhanced_rerank("预算", [chunk]) == [chunk]


def test_graph_rerank_lifts_mutually_supported_chunk() -> None:
    a = _chunk("a", "预算")
    b = _chunk("b", "预算 流程 审批")
    c = _chunk("c", "流程 审批 制度")

    ranked = graph_enhanced_rerank("预算", [a, b, c], boost=1.0)

    # b 与 c 共享"流程/审批"两个词元，图中心度更高，应排到 a 之前
    assert [chunk.chunk_id for chunk in ranked] == ["b", "a", "c"]


def test_graph_rerank_preserves_scores() -> None:
    a = _chunk("a", "预算", score=0.8)
    b = _chunk("b", "预算 流程 审批", score=0.7)
    c = _chunk("c", "流程 审批 制度", score=0.6)

    ranked = graph_enhanced_rerank("预算", [a, b, c], boost=1.0)

    scores = {chunk.chunk_id: chunk.score for chunk in ranked}
    assert scores == {"a": 0.8, "b": 0.7, "c": 0.6}


def test_graph_rerank_without_shared_terms_keeps_order() -> None:
    a = _chunk("a", "预算")
    b = _chunk("b", "天气")
    c = _chunk("c", "旅行")

    assert graph_enhanced_rerank("预算", [a, b, c], boost=0.0) == [a, b, c]


def test_graph_config_defaults_off_and_validates_boost() -> None:
    config = RetrievalConfig()
    assert config.graph_rerank_enabled is False
    assert config.graph_rerank_boost == 0.5

    with pytest.raises(ValueError):
        RetrievalConfig(graph_rerank_boost=1.5)
