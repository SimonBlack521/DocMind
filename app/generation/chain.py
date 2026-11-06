"""生成链：组装提示词 → 调用 LLM → 附带引用列表，支持一次性与流式两种产出。

提示词拼装（_messages）：
    system = SYSTEM_PROMPT（约束只依据参考资料作答、关键结论后标注 [n] 来源）
    user   = build_user_prompt（【参考资料】带编号片段 + 【用户问题】+ 作答指令）

硬约束：检索结果为空时不调用 LLM，直接返回固定文案（NO_CONTEXT_ANSWER），
避免模型在没有资料的情况下凭空编造。
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.metrics import metrics
from app.domain.dto import RetrievedChunk
from app.generation.citation_validator import (
    ANSWER_REJECTED,
    ANSWER_UNVERIFIED,
    ANSWER_VERIFIED,
    AnswerVerification,
    clean_answer_refs,
    validate_citations,
    verify_answer_support,
)
from app.generation.citations import Citation, build_citations
from app.generation.prompts import (
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_user_prompt,
    resolve_context_chunks,
)
from app.providers.llm import ChatMessage, LLMClient

logger = get_logger(__name__)

# 答案验证失败后的重生成指令（追加在用户消息末尾）：要求逐条标注来源、
# 资料未覆盖的内容明确说明，不得把"候选资料"当作"已支持的结论"。
_REGENERATE_INSTRUCTION = (
    "\n\n【校验反馈】上一版答案没有为关键结论标注可识别的引用，或与参考资料存在冲突。"
    "请严格依据上面的【参考资料】重新作答：每个关键结论后标注 [编号]；"
    "资料未覆盖的内容请直接写“资料中未提及”，不要推测或补充外部知识。"
)
# 全部结论都无法被证据支持时的保守文案（拒答而不是返回不可信断言）
UNSUPPORTED_ANSWER = "抱歉，本次检索到的资料不足以支持可靠回答，已停止作答。请补充资料或换个问法。"


@dataclass(slots=True)
class AnswerResult:
    """一次性回答结果：answer 为完整文本，usage 是 token 用量（调用时已计入指标）。"""

    answer: str
    citations: list[Citation]
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    # 独立答案验证状态（按实际送入模型的证据逐句校验）；None 表示未启用验证
    verification: AnswerVerification | None = None


@dataclass(slots=True)
class AnswerStream:
    """流式回答：citations 先行确定，tokens 是惰性迭代器（首次迭代才真正请求 LLM）。"""

    citations: list[Citation]
    model: str
    tokens: Iterator[str]


class RAGChain:
    """无状态，可复用；每次调用只依赖传入的检索片段。

    max_context_chars 是送 LLM 的参考资料总字符预算（0=不限），
    由全局配置注入，防止超长上下文撑爆 token、稀释关键证据。
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_context_chars: int = 0,
        verify_answers: bool = True,
        regen_max: int = 1,
        min_citation_coverage: float = 0.5,
    ) -> None:
        self._llm = llm
        self._max_context_chars = max_context_chars
        # 独立答案验证：引用列表非空不代表结论有证据支持，必须逐句校验
        self._verify_answers = verify_answers
        self._regen_max = max(0, regen_max)
        self._min_citation_coverage = min_citation_coverage

    def answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        context_chunks: Sequence[RetrievedChunk] | None = None,
    ) -> AnswerResult:
        """一次性生成：等待 LLM 完整输出后返回；无检索片段时短路返回固定文案。

        context_chunks 是"压缩后的提示词资料"（可选）：与 chunks 顺序、条数一一对应，
        只影响送入 LLM 的正文；引用列表仍按原始 chunks 生成，保证 snippet/定位信息
        不被压缩结果覆盖。无论成败都记录 llm_chat_duration_seconds（finally 保证）。

        citations 只覆盖"真正送入 LLM"的资料区前缀（resolve_context_chunks 按
        max_context_chars 预算解析）——被预算省略的片段不进 citations，
        防止引用列表里出现模型根本没见过的编号（清单第三优先级第 17 项）。

        答案验证（v2.2）：生成后按"实际送入模型的证据"逐句校验结论是否有支持；
        未通过且允许重生成时，带着校验反馈再生成一次，取更好的结果；仍失败时
        明确标注 unverified，全部结论无支持则返回保守拒答文案。
        """
        if not chunks:
            # 没有资料时不调用 LLM，避免模型凭空编造
            return AnswerResult(
                answer=NO_CONTEXT_ANSWER,
                citations=[],
                verification=AnswerVerification(
                    status=ANSWER_VERIFIED,
                    claims_total=0,
                    cited_claims=0,
                    coverage=1.0,
                    reason="no_context",
                ),
            )

        prompt_chunks = context_chunks if context_chunks is not None else chunks
        sent_count = len(
            resolve_context_chunks(prompt_chunks, max_chars=self._max_context_chars)
        )
        sent_chunks = list(prompt_chunks[:sent_count])
        citations = build_citations(chunks[:sent_count])
        result = self._generate_once(
            question, prompt_chunks, citations, sent_chunks, instruction=""
        )
        if (
            self._verify_answers
            and result.verification is not None
            and result.verification.status != ANSWER_VERIFIED
            and self._regen_max > 0
        ):
            # 带校验反馈重生成：只在验证失败时追加一次 LLM 调用，避免常态成本上升
            retry = self._generate_once(
                question,
                prompt_chunks,
                citations,
                sent_chunks,
                instruction=_REGENERATE_INSTRUCTION,
            )
            if self._verification_rank(retry.verification) > self._verification_rank(
                result.verification
            ):
                result = retry
            # 重生成结果也可能是 None（未开启验证分支），指标按"未验证"计
            retry_status = (
                result.verification.status
                if result.verification is not None
                else ANSWER_UNVERIFIED
            )
            metrics.inc("answer_regenerated_total", status=retry_status)
        if result.verification is not None and result.verification.status == ANSWER_REJECTED:
            # 全部关键结论都没有证据支持：拒答比返回不可信断言更安全
            logger.warning(
                "答案验证拒绝：结论全部无支持 examples=%s",
                result.verification.unsupported_examples,
            )
            return AnswerResult(
                answer=UNSUPPORTED_ANSWER,
                citations=[],
                model=result.model,
                usage=result.usage,
                verification=result.verification,
            )
        return result

    def _generate_once(
        self,
        question: str,
        prompt_chunks: Sequence[RetrievedChunk],
        citations: list[Citation],
        sent_chunks: list[RetrievedChunk],
        *,
        instruction: str,
    ) -> AnswerResult:
        """一次生成 + 引用校验 + 答案验证（重生成阶段复用同一套校验口径）。"""
        started = time.perf_counter()
        try:
            response = self._llm.chat(
                self._messages(
                    question, prompt_chunks, self._max_context_chars, instruction
                )
            )
        finally:
            metrics.observe(
                "llm_chat_duration_seconds", time.perf_counter() - started, model=self._llm.model
            )
        for kind, value in response.usage.items():
            metrics.inc("llm_tokens_total", value, kind=kind)

        answer_text = response.text.strip()
        # Citation Validator：过滤 LLM 幻觉引用（编号越界 / 引用位置不对应），
        # 并把答案文本中"已失效"的 [n] 同步清掉——引用列表与正文标注必须一致，
        # 否则前端会出现"有标注、无来源"的答案（v2.1 起答案文本一并修正）
        validation = validate_citations(answer_text, citations)
        if validation.out_of_range_refs or validation.unverified_citations:
            metrics.inc(
                "citation_issues_total",
                validation.out_of_range_refs + validation.unverified_citations,
            )
            logger.warning(
                "引用校验发现幻觉 answer_refs=%d out_of_range=%d unverified=%d",
                validation.total_answer_refs,
                validation.out_of_range_refs,
                validation.unverified_citations,
            )
        if validation.fact_conflicts:
            # L2 数字事实冲突（答案数字与引用片段不一致）：删除冲突引用与正文标注，
            # 冲突明细进指标与日志，供离线评测定位"数字幻觉"（清单第 17 项）
            metrics.inc("citation_fact_conflicts_total", validation.fact_conflicts)
            logger.warning(
                "数字事实校验冲突 count=%d conflicts=%s",
                validation.fact_conflicts,
                validation.conflict_details[:5],
            )
        # 全部引用被剔除时不拒答：正文仍在（已无任何 [n] 标注），避免丢掉可用摘要
        cleaned_answer = clean_answer_refs(answer_text, validation.cleaned_citations)
        verification: AnswerVerification | None = None
        if self._verify_answers:
            # 用"实际送入模型的完整证据"校验：不只看引用列表，还看正文结论
            verification = verify_answer_support(
                cleaned_answer,
                validation.cleaned_citations,
                sent_chunks,
                min_coverage=self._min_citation_coverage,
                fact_conflicts=validation.fact_conflicts,
            )
            metrics.inc("answer_verification_total", status=verification.status)
        return AnswerResult(
            answer=cleaned_answer,
            citations=validation.cleaned_citations,
            model=response.model,
            usage=dict(response.usage),
            verification=verification,
        )

    @staticmethod
    def _verification_rank(verification: AnswerVerification | None) -> int:
        """验证状态排序：verified > unverified > rejected；未启用验证视为最高。"""
        if verification is None:
            return 3
        if verification.status == ANSWER_VERIFIED:
            return 2
        if verification.status == ANSWER_UNVERIFIED:
            return 1
        return 0

    def stream_answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        context_chunks: Sequence[RetrievedChunk] | None = None,
    ) -> AnswerStream:
        """流式生成：引用列表先行确定（与提示词编号 [n] 对齐），tokens 惰性产出。

        LLM 请求推迟到 tokens 首次被迭代时发出，因此本方法本身几乎不耗时。
        """
        if not chunks:
            # 与一次性回答一致的硬约束：没有资料时不调用 LLM，直接流式产出固定文案
            return AnswerStream(citations=[], model="", tokens=iter([NO_CONTEXT_ANSWER]))

        prompt_chunks = context_chunks if context_chunks is not None else chunks
        # citations 只覆盖"真正送入 LLM"的资料区前缀（与一次性回答同一口径）
        sent_count = len(
            resolve_context_chunks(prompt_chunks, max_chars=self._max_context_chars)
        )
        return AnswerStream(
            citations=build_citations(chunks[:sent_count]),
            model=self._llm.model,
            tokens=self._timed_stream(
                self._llm.stream_chat(
                    self._messages(question, prompt_chunks, self._max_context_chars)
                ),
                self._llm.model,
            ),
        )

    @staticmethod
    def _timed_stream(tokens: Iterator[str], model: str) -> Iterator[str]:
        """包一层计时：从首次迭代（真正发起请求）到流结束记录 llm_stream_duration_seconds。"""
        started = time.perf_counter()
        try:
            yield from tokens
        finally:
            metrics.observe(
                "llm_stream_duration_seconds", time.perf_counter() - started, model=model
            )

    @staticmethod
    def _messages(
        question: str,
        chunks: Sequence[RetrievedChunk],
        max_context_chars: int = 0,
        instruction: str = "",
    ) -> list[ChatMessage]:
        """拼装对话消息：system 定规则，user 携带编号化资料与用户问题。

        instruction 非空时追加在用户消息末尾（用于验证失败后的带反馈重生成）。
        """
        prompt = build_user_prompt(question, chunks, max_chars=max_context_chars)
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"{prompt}{instruction}"),
        ]
