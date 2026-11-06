"""v2.0.2 P1 能力测试：路由置信度、证据覆盖反思、执行预算、引用数字校验、
Chunk 向量复用与父块上下文预算。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.settings import Settings
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import AgenticConfig, RetrievalConfig
from app.generation.citation_validator import _facts_consistent, validate_citations
from app.generation.citations import Citation
from app.orchestration.agentic import AgenticRAG
from app.orchestration.pipeline import RAGPipeline
from app.providers.llm import ChatMessage, LLMClient, LLMResponse
from app.retrieval.hybrid import HybridRetriever, _truncate_text
from app.retrieval.query_router import QueryRouter

# ---------------------------------------------------------------- 假依赖


class _FakeLLM(LLMClient):
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)

    @property
    def model(self) -> str:
        return "fake"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
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


def _chunk(chunk_id: str, doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"证据 {chunk_id}",
        score=0.9,
        doc_id=doc_id,
        source_uri=f"s3://docs/{chunk_id}",
        title=f"文档 {doc_id}",
    )


def _agent(
    llm: LLMClient,
    search_results: dict[str, list[RetrievedChunk]],
    *,
    config: AgenticConfig | None = None,
) -> tuple[AgenticRAG, list[str]]:
    calls: list[str] = []

    def search(question: str) -> list[RetrievedChunk]:
        calls.append(question)
        return search_results.get(question, [])

    return (
        AgenticRAG(
            llm=llm,
            router=QueryRouter(),
            search=search,
            structured=lambda question: None,
            api=lambda question: None,
            structured_available=False,
            api_available=False,
            config=config
            or AgenticConfig(enabled=True, max_rounds=2, min_evidence_chunks=1),
        ),
        calls,
    )


# ---------------------------------------------------------------- 路由


def test_router_detects_document_context_and_lowers_confidence() -> None:
    decision = QueryRouter().route(
        "统计制度中规定的人员上限是多少", structured_available=True
    )

    assert decision.route == "document"
    assert decision.confidence < 0.5
    assert decision.intent == "policy_lookup"


def test_router_marks_multi_intent_and_weak_realtime() -> None:
    mixed = QueryRouter().route("统计当前库存情况", structured_available=True, api_available=True)
    assert mixed.route == "sql"
    assert mixed.confidence == 0.5
    assert mixed.intent == "mixed_analysis"

    weak = QueryRouter().route(
        "现在的年假政策如何", structured_available=False, api_available=True
    )
    assert weak.route == "document"
    assert weak.confidence < 0.5


# ---------------------------------------------------------------- Agentic


def test_agent_reflects_on_coverage_gap_even_when_chunk_count_is_enough() -> None:
    """两个子问题都命中同一份文档 = 覆盖不足，即使片段数量已达标也要补检索。"""
    llm = _FakeLLM(
        [
            '[{"tool":"document","query":"问题A"},{"tool":"document","query":"问题B"}]',
            '[{"tool":"document","query":"补充C"}]',
        ]
    )
    search_results = {
        "问题A": [_chunk("a", "doc-1")],
        "问题B": [_chunk("b", "doc-1")],
        "补充C": [_chunk("c", "doc-2")],
    }
    agent, calls = _agent(llm, search_results)

    outcome = agent.run("比较两份制度", [])

    assert calls == ["问题A", "问题B", "补充C"]
    assert outcome.rounds == 2
    assert len({chunk.doc_id for chunk in outcome.chunks}) == 2


def test_agent_stops_when_tool_call_budget_is_exhausted() -> None:
    llm = _FakeLLM(
        [
            (
                '[{"tool":"document","query":"A"},{"tool":"document","query":"B"},'
                '{"tool":"document","query":"C"}]'
            )
        ]
    )
    agent, calls = _agent(
        llm,
        {"A": [_chunk("a", "doc-1")]},
        config=AgenticConfig(
            enabled=True, max_rounds=3, min_evidence_chunks=1, max_tool_calls=1
        ),
    )

    outcome = agent.run("复杂问题", [])

    assert calls == ["A"]
    assert outcome.rounds == 1


# ---------------------------------------------------------------- 引用事实校验


def test_citation_numbers_must_exist_in_snippet() -> None:
    assert _facts_consistent("报销上限为 8000 元", "报销上限为 5000 元。") is False
    # 千分位/小数归一化后视为同一个数字
    assert _facts_consistent("金额 5,000", "金额 5000.0 元") is True
    # 片段没有数字时不做判定（摘要式引用不能被误伤）
    assert _facts_consistent("金额 8000", "相关制度见附件。") is True


def test_citation_with_conflicting_number_is_removed() -> None:
    citations = [
        Citation(
            index=1,
            chunk_id="c1",
            doc_id="d1",
            source_uri="s3://x",
            title="制度",
            snippet="报销上限为 5000 元。",
            score=0.9,
            metadata={},
        )
    ]

    result = validate_citations("报销上限为 8000 元 [1]。", citations)

    assert result.fact_conflicts == 1
    assert result.unverified_citations == 1
    assert result.cleaned_citations == []


# ---------------------------------------------------------------- 向量复用与父块预算


class _FakeVectorStore:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def get_vectors_by_ids(self, ids: Sequence[str]) -> dict[str, list[float]]:
        return {chunk_id: self._vectors[chunk_id] for chunk_id in ids if chunk_id in self._vectors}


class _FakeEmbedding:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


def _retriever(
    *,
    vectors: dict[str, list[float]] | None = None,
    parent_records: list[Any] | None = None,
    config: RetrievalConfig | None = None,
) -> tuple[HybridRetriever, _FakeEmbedding]:
    embedding = _FakeEmbedding()
    retriever = HybridRetriever(
        vector_store=_FakeVectorStore(vectors or {}),
        keyword_store=SimpleNamespace(
            get_by_ids=lambda ids: parent_records or []
        ),
        embedding=embedding,
        reranker=SimpleNamespace(rerank=lambda question, chunks, top_n=None, **kw: chunks),
        rewriter=SimpleNamespace(rewrite=lambda question: [question]),
        tenant_id="t",
        kb_id="kb",
        config=config or RetrievalConfig(),
    )
    return retriever, embedding


def test_mmr_reuses_stored_chunk_vectors_without_reembedding() -> None:
    chunks = [_chunk("c1", "d1"), _chunk("c2", "d1")]
    retriever, embedding = _retriever(vectors={"c1": [0.1], "c2": [0.2]})

    vectors = retriever._reuse_chunk_vectors(chunks)

    assert vectors == [[0.1], [0.2]]
    assert embedding.calls == []


def test_mmr_only_embeds_missing_chunk_vectors() -> None:
    chunks = [_chunk("c1", "d1"), _chunk("c2", "d1")]
    retriever, embedding = _retriever(vectors={"c1": [0.1]})

    vectors = retriever._reuse_chunk_vectors(chunks)

    assert vectors == [[0.1], [0.1, 0.2, 0.3]]
    assert embedding.calls == [["证据 c2"]]


def test_parent_content_respects_char_budget() -> None:
    long_parent = "第一句。" + "很长的正文内容。" * 20
    child = _chunk("c1", "d1")
    child.metadata["parent_chunk_id"] = "p1"
    retriever, _embedding = _retriever(
        parent_records=[SimpleNamespace(chunk_id="p1", content=long_parent)],
        config=RetrievalConfig(parent_max_chars=40, parent_context_max_chars=40),
    )

    expanded = retriever._expand_parents([child])

    parent_text = expanded[0].metadata["parent_content"]
    assert len(parent_text) <= 40
    assert parent_text.endswith("…")


def test_truncate_text_prefers_sentence_boundary() -> None:
    text = "第一句话。" + "第二句话。" + "第三句话。"
    assert _truncate_text(text, 10) == "第一句话。第二句话。…"
    # 句末太靠前（不足 1/3）时按硬上限截断，避免只留下一个很短的碎片
    assert _truncate_text("短。很长的内容继续继续", 6) == "短。很长的内…"


# ---------------------------------------------------------------- 统一证据


def test_pipeline_evidence_covers_document_sql_and_api() -> None:
    pipeline = RAGPipeline(settings=Settings(), registry=SimpleNamespace())

    document = pipeline._document_evidence([_chunk("c1", "doc-1")])[0]
    assert document.source_type == "document"
    assert document.source_id == "doc-1"
    assert document.locator == "c1"
    assert document.provenance["source_uri"] == "s3://docs/c1"

    database = pipeline._terminal_evidence("sql", "共 5 行", sql="SELECT count(*)")[0]
    assert database.source_type == "database"
    assert database.locator == "SELECT count(*)"
    assert database.provenance == {"sql": "SELECT count(*)"}

    api = pipeline._terminal_evidence("api", "当前库存 3")[0]
    assert api.source_type == "api"
    assert api.locator == "api"


# ------------------------------------------------------------------ v2.1 清单修复
def _citation(index: int) -> Citation:
    return Citation(
        index=index,
        chunk_id=f"c{index}",
        doc_id="d1",
        source_uri="s3://docs/d1",
        title="制度文档",
        snippet="员工允许报销",
        score=0.9,
        metadata={},
    )


def test_clean_answer_refs_removes_dropped_markers() -> None:
    """被剔除的引用编号必须从答案文本中同步移除（清单 P1-02）。"""
    from app.generation.citation_validator import clean_answer_refs

    answer = "报销限额 500 元 [1]。差旅标准见 [3]。编号 001、002 [1][2][4]"
    cleaned = clean_answer_refs(answer, [_citation(1), _citation(2)])
    assert "[3]" not in cleaned
    assert "[4]" not in cleaned
    assert "[1]" in cleaned
    assert "[2]" in cleaned


def test_clean_answer_refs_keeps_valid_markers() -> None:
    from app.generation.citation_validator import clean_answer_refs

    answer = "员工允许报销 [1]。"
    assert clean_answer_refs(answer, [_citation(1)]) == answer


def _sql_kb() -> Any:
    from app.domain.kb_config import KBConfig, StructuredConfig

    return KBConfig(
        kb_id="kb-sql-001",
        kb_code="sqlkb",
        name="SQL 测试库",
        structured=StructuredConfig(
            enabled=True,
            sql_url="sqlite:///./data/unused.db",
            allowed_tables=["expenses"],
        ),
    )


def test_query_sql_failure_returns_explicit_unavailable() -> None:
    """SQL 路由失败：明确降级告知，绝不静默改用历史文档作答（清单 P1-01）。"""
    from app.core.errors import AppError
    from app.generation.prompts import STRUCTURED_UNAVAILABLE_ANSWER

    pipeline = RAGPipeline(settings=Settings(), registry=SimpleNamespace())

    def _boom(kb: Any, question: str) -> Any:
        raise AppError("结构化查询超时")

    pipeline._structured_answer = _boom  # type: ignore[method-assign]
    result = pipeline.query(_sql_kb(), "统计各部门的报销总额")
    assert result.route == "sql"
    assert result.degraded is True
    assert "超时" in result.degrade_reason
    # 关键断言：答案不是文档检索的产物，而是明确的"数据不可用"文案
    assert result.answer == STRUCTURED_UNAVAILABLE_ANSWER
    assert result.citations == []


def test_query_stream_sql_failure_streams_unavailable_text() -> None:
    """流式路径同样明确降级：token 内容为不可用文案且带 degraded 标记。"""
    from app.core.errors import AppError
    from app.generation.prompts import STRUCTURED_UNAVAILABLE_ANSWER

    pipeline = RAGPipeline(settings=Settings(), registry=SimpleNamespace())

    def _boom(kb: Any, question: str) -> Any:
        raise AppError("结构化查询超时")

    pipeline._structured_answer = _boom  # type: ignore[method-assign]
    stream = pipeline.query_stream(_sql_kb(), "统计各部门的报销总额")
    assert stream.route == "sql"
    assert stream.degraded is True
    assert list(stream.tokens) == [STRUCTURED_UNAVAILABLE_ANSWER]


def test_query_api_failure_does_not_fall_back_to_documents() -> None:
    """实时 API 失败：不回退文档检索（"当前库存"答成旧说明书比失败更危险）。"""
    from app.core.errors import AppError
    from app.domain.kb_config import (
        ApiEndpointConfig,
        ApiRetrievalConfig,
        KBConfig,
    )
    from app.generation.prompts import STRUCTURED_UNAVAILABLE_ANSWER

    pipeline = RAGPipeline(settings=Settings(), registry=SimpleNamespace())

    def _boom(kb: Any, question: str) -> Any:
        raise AppError("实时接口超时")

    pipeline._api_answer = _boom  # type: ignore[method-assign]
    kb = KBConfig(
        kb_id="kb-api-001",
        kb_code="apikb",
        name="API 测试库",
        api=ApiRetrievalConfig(
            enabled=True,
            endpoints=[
                ApiEndpointConfig(
                    endpoint_id="stock",
                    url="http://inventory.example.com/api/stock",
                    description="库存查询接口",
                )
            ],
        ),
    )
    result = pipeline.query(kb, "当前库存是多少")
    assert result.route == "api"
    assert result.degraded is True
    assert result.answer == STRUCTURED_UNAVAILABLE_ANSWER


def test_api_retriever_enforces_byte_budget_while_streaming() -> None:
    """流式读取执行字节预算：分块传输且无 Content-Length 也无法超限缓冲（清单 P1-06）。"""
    import httpx as _httpx

    from app.core.errors import StructuredQueryError
    from app.domain.kb_config import ApiEndpointConfig, ApiRetrievalConfig
    from app.structured.api_retriever import ApiRetriever

    config = ApiRetrievalConfig(
        enabled=True,
        max_response_bytes=16,
        endpoints=[
            ApiEndpointConfig(
                endpoint_id="stock",
                url="http://inventory.example.com/api/stock",
                description="库存查询接口",
            )
        ],
    )

    def handler(request: _httpx.Request) -> _httpx.Response:
        # 64 字节分块响应，不声明 Content-Length：旧实现会先完整接收再检查
        return _httpx.Response(
            200, headers={"content-type": "application/json"}, content=b"x" * 64
        )

    retriever = ApiRetriever(
        config=config, client=_httpx.Client(transport=_httpx.MockTransport(handler))
    )
    with pytest.raises(StructuredQueryError, match="大小上限"):
        retriever.answer("当前库存")
