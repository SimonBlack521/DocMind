"""S3 检索/生成防护：上下文预算、提示词注入防护、低置信拒答。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from app.core.settings import Settings
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import KBConfig
from app.generation.chain import RAGChain
from app.generation.prompts import (
    LOW_CONFIDENCE_ANSWER,
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)
from app.orchestration.pipeline import RAGPipeline
from app.providers.llm import ChatMessage, LLMResponse


def _chunk(content: str, *, score: float = 0.9, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        score=score,
        content=content,
        source_uri="kb://t/a.txt",
        title="a",
    )


# ------------------------------------------------------------------ 上下文预算


def test_context_block_unlimited_by_default() -> None:
    chunks = [_chunk("甲" * 100, chunk_id="c1"), _chunk("乙" * 100, chunk_id="c2")]
    text = build_context_block(chunks)
    assert "[1]" in text and "[2]" in text
    assert "未纳入" not in text


def test_context_block_budget_omits_overflow() -> None:
    chunks = [
        _chunk("甲" * 40, chunk_id="c1"),
        _chunk("乙" * 40, chunk_id="c2"),
        _chunk("丙" * 40, chunk_id="c3"),
    ]
    unlimited = build_context_block(chunks)
    # 预算只够放下前两段
    budget = len("\n\n".join(unlimited.split("\n\n")[:2])) + 5
    text = build_context_block(chunks, max_chars=budget)
    assert "[1]" in text and "[2]" in text
    assert "[3]" not in text
    assert "另有 1 条相关资料因上下文长度限制未纳入" in text


def test_context_block_budget_truncates_first_when_oversized() -> None:
    chunks = [_chunk("甲" * 500, chunk_id="c1"), _chunk("乙" * 10, chunk_id="c2")]
    text = build_context_block(chunks, max_chars=100)
    # 首条即超预算：截断保留（资料区不为空），并标注其余被省略
    assert text.endswith("（另有 1 条相关资料因上下文长度限制未纳入）")
    assert "…" in text
    assert "乙乙" not in text


def test_user_prompt_passes_budget() -> None:
    chunks = [_chunk("甲" * 40, chunk_id="c1"), _chunk("乙" * 40, chunk_id="c2")]
    text = build_user_prompt("问题", chunks, max_chars=60)
    assert "【参考资料】" in text and "【用户问题】" in text
    assert "[2]" not in text  # 第二段被预算截掉


# ------------------------------------------------------------------ 注入防护


def test_system_prompt_has_injection_guard() -> None:
    assert "不可信" in SYSTEM_PROMPT
    assert "不得执行" in SYSTEM_PROMPT


# ------------------------------------------------------------------ RAGChain 预算接线


class _FakeLLM:
    model = "fake-llm"

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: object
    ) -> LLMResponse:
        self.messages = list(messages)
        return LLMResponse(text="依据资料的回答", model=self.model, usage={})

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: object) -> Iterator[str]:
        self.messages = list(messages)
        yield "依据资料的回答"


def test_chain_applies_context_budget() -> None:
    llm = _FakeLLM()
    chain = RAGChain(llm, max_context_chars=60)  # type: ignore[arg-type]
    chunks = [_chunk("甲" * 40, chunk_id="c1"), _chunk("乙" * 40, chunk_id="c2")]
    chain.answer("问题", chunks)
    user_message = llm.messages[-1].content
    assert "[2]" not in user_message
    assert "未纳入" in user_message


def test_chain_uses_compressed_context_but_original_citations() -> None:
    """上下文压缩只改变送模型的资料；引用 snippet 仍来自未压缩原文。"""
    from app.generation.compression import compress_chunks

    long_content = (
        "原始章节开头。"
        "员工需要在 30 天内提交报销申请。"
        "大段与问题无关的背景材料与流程沿革说明，仅用于内部存档。"
        "差旅住宿标准上限为 500 元。"
        "这些内容主要记录制度编写过程中的历史讨论与多轮意见征询材料。"
        "此外还附有若干内部存档说明，与当前问答没有直接关系。"
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="doc-1",
        score=0.9,
        content=long_content,
        source_uri="kb://t/a.txt",
        title="a",
        metadata={"parent_content": long_content},
    )
    compressed = compress_chunks([chunk], "住宿标准是多少", max_sentences=2, min_chars=0)

    llm = _FakeLLM()
    chain = RAGChain(llm)  # type: ignore[arg-type]
    result = chain.answer("住宿标准是多少", [chunk], context_chunks=compressed)

    user_message = llm.messages[-1].content
    assert "500 元" in user_message
    assert "背景材料" not in user_message
    # 引用按原始 chunk 生成：snippet 保留完整原文（含背景段），便于审计回查
    assert result.citations[0].snippet.startswith("原始章节开头")


# ------------------------------------------------------------------ 低置信拒答


class _FakeLLMRegistry:
    def __init__(self, llm: _FakeLLM) -> None:
        self._llm = llm

    def llm(self) -> _FakeLLM:
        return self._llm

    def runtime(self, kb: KBConfig) -> None:
        # query() 会取 runtime 传给 _retrieve；测试里 _retrieve 已被打桩，无需真实运行时
        return None


def _kb() -> KBConfig:
    return KBConfig(kb_id="kb-guard-001", kb_code="guard", name="防护测试库")


def test_low_confidence_refusal_skips_llm() -> None:
    """检索最高分低于阈值：直接返回拒答文案，不调用 LLM。"""
    llm = _FakeLLM()
    pipeline = RAGPipeline(
        settings=Settings(retrieval_min_score=0.5),
        registry=_FakeLLMRegistry(llm),  # type: ignore[arg-type]
    )
    low_chunks = [_chunk("弱证据", score=0.1)]
    pipeline._retrieve = lambda *a, **k: low_chunks  # type: ignore[method-assign]

    result = pipeline.query(_kb(), "一个匹配度很低的问题")
    assert result.answer == LOW_CONFIDENCE_ANSWER
    assert result.citations == []
    assert llm.messages == []  # LLM 未被调用
    assert result.chunks == low_chunks  # chunks 保留给审计


def test_high_confidence_calls_llm() -> None:
    """最高分达到阈值：正常调用 LLM。"""
    llm = _FakeLLM()
    pipeline = RAGPipeline(
        settings=Settings(retrieval_min_score=0.5),
        registry=_FakeLLMRegistry(llm),  # type: ignore[arg-type]
    )
    pipeline._retrieve = lambda *a, **k: [_chunk("强证据", score=0.9)]  # type: ignore[method-assign]

    result = pipeline.query(_kb(), "一个匹配度足够的问题")
    assert result.answer == "依据资料的回答"
    assert llm.messages != []


def test_min_score_disabled_by_default() -> None:
    """阈值 0（默认关闭）：即使分数很低也照常走 LLM。"""
    llm = _FakeLLM()
    pipeline = RAGPipeline(
        settings=Settings(),
        registry=_FakeLLMRegistry(llm),  # type: ignore[arg-type]
    )
    pipeline._retrieve = lambda *a, **k: [_chunk("弱证据", score=0.01)]  # type: ignore[method-assign]

    result = pipeline.query(_kb(), "问题")
    assert result.answer == "依据资料的回答"


# ------------------------------------------------------- 独立答案验证（v2.2）


class _ScriptedLLM(_FakeLLM):
    """按顺序返回预置文本，用于验证"失败 → 重生成 → 取更好结果"的链路。"""

    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self._texts = texts
        self.calls = 0

    def chat(
        self, messages: Sequence[ChatMessage], **kwargs: object
    ) -> LLMResponse:
        self.messages = list(messages)
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return LLMResponse(text=text, model=self.model, usage={})


def test_verify_marks_unsupported_claims_as_rejected() -> None:
    """回归（用户真实样本）：正文结论完全无法由证据支持时必须拒答。"""
    from app.generation.citation_validator import (
        ANSWER_REJECTED,
        verify_answer_support,
    )

    evidence = [_chunk("差旅报销需提前审批，住宿标准上限为 500 元。")]
    verification = verify_answer_support(
        "董事会决定在纽约设立新的研发中心，并追加十亿美元投资。",
        [],
        evidence,
    )
    assert verification.status == ANSWER_REJECTED
    assert verification.claims_total == 1
    assert verification.cited_claims == 0


def test_verify_accepts_summary_supported_by_full_evidence() -> None:
    """没有 [n] 但整句能在"实际送入模型的证据"里找到重叠，视为摘要式支持。"""
    from app.generation.citation_validator import ANSWER_VERIFIED, verify_answer_support

    evidence = [_chunk("差旅报销需提前审批，住宿标准上限为 500 元。")]
    verification = verify_answer_support("差旅报销需要提前审批。", [], evidence)
    assert verification.status == ANSWER_VERIFIED


def test_verify_flags_fact_conflict_and_low_citation_coverage() -> None:
    """回归：原文上限 5000、答案写 8000（数字冲突）必须标 unverified。"""
    from app.generation.citation_validator import (
        ANSWER_UNVERIFIED,
        verify_answer_support,
    )

    chunk = _chunk("报销上限为 5000 元。", chunk_id="c1")
    from app.generation.citations import build_citations

    citations = build_citations([chunk])
    verification = verify_answer_support(
        "报销上限为 8000 元，无需审批。",
        citations,
        [chunk],
        fact_conflicts=1,
    )
    assert verification.status == ANSWER_UNVERIFIED
    assert "fact_conflicts" in verification.reason


def test_chain_regenerates_once_and_prefers_cited_answer() -> None:
    """验证失败 → 带反馈重生成一次；引用完整的新答案优先返回。"""
    from app.generation.citation_validator import ANSWER_VERIFIED

    chunk = _chunk("差旅报销需提前审批，住宿标准上限为 500 元。", chunk_id="c1")
    llm = _ScriptedLLM(
        [
            # 第 1 版：结论无法被证据支持且没有引用
            "董事会决定在纽约设立新的研发中心。",
            # 第 2 版（收到校验反馈后）：逐条引用
            "差旅报销需提前审批[1]，住宿标准上限为 500 元[1]。",
        ]
    )
    chain = RAGChain(llm, regen_max=1)  # type: ignore[arg-type]
    result = chain.answer("报销标准？", [chunk])

    assert llm.calls == 2
    assert "校验反馈" in llm.messages[-1].content
    assert result.answer == "差旅报销需提前审批[1]，住宿标准上限为 500 元[1]。"
    assert result.verification is not None
    assert result.verification.status == ANSWER_VERIFIED


def test_chain_refuses_when_all_claims_unsupported() -> None:
    """重生成后仍全部无支持 → 返回保守拒答文案，不返回不可信断言。"""
    from app.generation.chain import UNSUPPORTED_ANSWER

    chunk = _chunk("差旅报销需提前审批，住宿标准上限为 500 元。", chunk_id="c1")
    llm = _ScriptedLLM(["董事会决定在纽约设立新的研发中心。"])
    chain = RAGChain(llm, regen_max=1)  # type: ignore[arg-type]
    result = chain.answer("报销标准？", [chunk])

    assert result.answer == UNSUPPORTED_ANSWER
    assert result.citations == []
    assert result.verification is not None
    assert result.verification.status == "rejected"


def test_chain_marks_unverified_without_regeneration() -> None:
    """regen_max=0：不重生成，但必须明确标注未验证（而不是静默通过）。"""
    from app.generation.citation_validator import ANSWER_UNVERIFIED

    chunk = _chunk("差旅报销需提前审批，住宿标准上限为 500 元。", chunk_id="c1")
    llm = _ScriptedLLM(["董事会决定在纽约设立新的研发中心。"])
    chain = RAGChain(llm, regen_max=0)  # type: ignore[arg-type]
    result = chain.answer("报销标准？", [chunk])

    assert llm.calls == 1
    assert result.verification is not None
    assert result.verification.status in {ANSWER_UNVERIFIED, "rejected"}
