"""Agentic RAG 测试（v2.0）：规划解析、工具选择、反思补检索与安全回退。

LLM 用按序返回预置文本的假实现；检索/结构化/API 用回调桩，
不依赖真实模型与存储。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from app.core.settings import Settings
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import AgenticConfig, KBConfig
from app.orchestration.agentic import AgenticRAG, StructuredTerminal
from app.orchestration.pipeline import RAGPipeline
from app.providers.llm import ChatMessage, LLMClient, LLMResponse
from app.retrieval.query_router import QueryRouter


class FakeLLM(LLMClient):
    """按顺序返回预置响应；用尽后返回空字符串（模拟模型无输出）。"""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-agentic"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResponse(text=text, model=self.model)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        return iter([self.chat(messages).text])


def _chunk(chunk_id: str, content: str = "证据内容") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=0.9,
        doc_id=f"doc-{chunk_id}",
        source_uri=f"s3://docs/{chunk_id}",
        title=f"文档 {chunk_id}",
    )


def _build(
    llm: LLMClient | None,
    *,
    search_results: dict[str, list[RetrievedChunk]] | None = None,
    structured_result: StructuredTerminal | None = None,
    api_result: StructuredTerminal | None = None,
    structured_available: bool = False,
    api_available: bool = False,
    config: AgenticConfig | None = None,
) -> tuple[AgenticRAG, list[str]]:
    calls: list[str] = []

    def search(question: str) -> list[RetrievedChunk]:
        calls.append(question)
        return (search_results or {}).get(question, [])

    def structured(question: str) -> StructuredTerminal | None:
        calls.append(f"sql:{question}")
        return structured_result

    def api(question: str) -> StructuredTerminal | None:
        calls.append(f"api:{question}")
        return api_result

    agent = AgenticRAG(
        llm=llm,
        router=QueryRouter(),
        search=search,
        structured=structured,
        api=api,
        config=config or AgenticConfig(enabled=True, max_rounds=2, min_evidence_chunks=1),
        structured_available=structured_available,
        api_available=api_available,
    )
    return agent, calls


def test_agentic_plans_multiple_sub_queries_and_merges_dedup() -> None:
    llm = FakeLLM(
        [
            '[{"tool":"document","query":"问题一"},{"tool":"document","query":"问题二"}]',
        ]
    )
    results = {"问题一": [_chunk("a"), _chunk("b")], "问题二": [_chunk("b"), _chunk("c")]}
    agent, calls = _build(llm, search_results=results, config=AgenticConfig(
        enabled=True, max_rounds=1, min_evidence_chunks=1
    ))

    outcome = agent.run("复合问题", [])

    assert calls == ["问题一", "问题二"]
    assert [chunk.chunk_id for chunk in outcome.chunks] == ["a", "b", "c"]
    assert outcome.rounds == 1
    assert len(outcome.steps) == 2


def test_agentic_reflects_when_evidence_insufficient() -> None:
    llm = FakeLLM(
        [
            '[{"tool":"document","query":"第一轮问题"}]',
            '[{"tool":"document","query":"补充问题"}]',
        ]
    )
    results = {"第一轮问题": [], "补充问题": [_chunk("a")]}
    agent, calls = _build(llm, search_results=results)

    outcome = agent.run("复杂问题", [])

    assert calls == ["第一轮问题", "补充问题"]
    assert outcome.rounds == 2
    assert [chunk.chunk_id for chunk in outcome.chunks] == ["a"]


def test_agentic_stops_when_evidence_sufficient() -> None:
    llm = FakeLLM(['[{"tool":"document","query":"问题"}]'])
    results = {"问题": [_chunk("a"), _chunk("b")]}
    agent, calls = _build(llm, search_results=results)

    outcome = agent.run("问题", [])

    assert calls == ["问题"]
    assert outcome.rounds == 1
    assert llm.calls == 1  # 证据充足，不触发反思


def test_agentic_sql_tool_returns_terminal() -> None:
    terminal = StructuredTerminal(route="sql", answer="共 5 行", sql="SELECT count(*)")
    llm = FakeLLM(['[{"tool":"sql","query":"统计总数"}]'])
    agent, calls = _build(
        llm, structured_result=terminal, structured_available=True
    )

    outcome = agent.run("统计总数", [])

    assert calls == ["sql:统计总数"]
    assert outcome.terminal == terminal
    assert outcome.chunks == []


def test_agentic_filters_unavailable_tools_and_falls_back() -> None:
    llm = FakeLLM(['[{"tool":"sql","query":"统计总数"}]'])
    agent, calls = _build(llm, structured_available=False)

    outcome = agent.run("统计总数", [])

    # SQL 未启用 → 计划里唯一的步骤被过滤 → 回退 QueryRouter 的文档检索
    assert calls == ["统计总数"]
    assert outcome.steps[0].tool == "document"


def test_agentic_invalid_planner_output_falls_back_to_router() -> None:
    llm = FakeLLM(["我无法规划这个问题"])
    results = {"原始问题": [_chunk("a")]}
    agent, calls = _build(llm, search_results=results, config=AgenticConfig(
        enabled=True, max_rounds=1, min_evidence_chunks=1
    ))

    outcome = agent.run("原始问题", [])

    assert calls == ["原始问题"]
    assert outcome.steps[0].query == "原始问题"


def test_agentic_planner_disabled_uses_single_document_step() -> None:
    llm = FakeLLM([])
    results = {"问题": [_chunk("a")]}
    agent, calls = _build(
        llm,
        search_results=results,
        config=AgenticConfig(enabled=True, use_llm_planner=False, max_rounds=1),
    )

    outcome = agent.run("问题", [])

    assert calls == ["问题"]
    assert len(outcome.steps) == 1
    assert outcome.steps[0].tool == "document"
    assert llm.calls == 0


def test_agentic_without_llm_uses_single_fallback_step() -> None:
    agent, calls = _build(None, search_results={"问题": [_chunk("a")]})

    outcome = agent.run("问题", [])

    assert calls == ["问题"]
    assert outcome.rounds == 1
    assert [chunk.chunk_id for chunk in outcome.chunks] == ["a"]


class _FakeRegistry:
    """管线集成测试用注册表：固定返回同一个假 LLM，运行时对象为空。"""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def llm(self) -> LLMClient:
        return self._llm

    def runtime(self, kb: KBConfig) -> None:
        # query() 只是把 runtime 透传给 _retrieve；本测试已打桩 _retrieve
        return None


def test_pipeline_uses_agentic_loop_when_enabled() -> None:
    """kb.agentic.enabled + mode=auto：管线走多子问题检索，而不是单次检索。"""
    llm = FakeLLM(
        [
            '[{"tool":"document","query":"子问题一"},{"tool":"document","query":"子问题二"}]',
            "综合答案",
        ]
    )
    pipeline = RAGPipeline(settings=Settings(), registry=_FakeRegistry(llm))
    retrieved: list[str] = []

    def fake_retrieve(
        kb: KBConfig, runtime: object, question: str, **kwargs: object
    ) -> list[RetrievedChunk]:
        retrieved.append(question)
        return [_chunk(question)]

    pipeline._retrieve = fake_retrieve  # type: ignore[method-assign]
    kb = KBConfig(
        kb_id="kb-agentic-001",
        kb_code="agentic",
        name="Agentic 测试库",
        agentic=AgenticConfig(enabled=True, max_rounds=1, min_evidence_chunks=1),
    )

    result = pipeline.query(kb, "复合问题")

    assert retrieved == ["子问题一", "子问题二"]
    assert result.route == "document"
    assert result.model == "fake-agentic"
    assert [chunk.chunk_id for chunk in result.chunks] == ["子问题一", "子问题二"]


def test_agentic_config_validation() -> None:
    config = AgenticConfig()
    assert config.enabled is False
    assert config.max_rounds == 3

    with pytest.raises(ValueError):
        AgenticConfig(max_rounds=0)
    with pytest.raises(ValueError):
        AgenticConfig(min_evidence_chunks=30, max_evidence_chunks=20)
