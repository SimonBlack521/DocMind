"""Agentic RAG（v2.0 完整版）：规划 → 执行工具 → 观察证据 → 反思补检索 → 交付答案。

与普通 RAG 的差异：
- 普通 RAG 只做一次"改写后的检索 + 生成"；
- Agentic RAG 先让 LLM 把问题拆成若干子问题并选择工具（文档检索 / 只读 SQL /
  实时 API），执行后检查证据是否充分，不充分则再规划补充查询，直到证据足够
  或用完轮次。

安全与边界：
- 工具集合受知识库配置约束：未启用 structured/api 时不向模型提供对应工具；
- SQL 仍走 SQLGuard 白名单校验，API 仍只访问已登记端点；
- 规划/反思失败或模型输出不合法时回退到 QueryRouter 的单步确定性路由，
  保证 Agentic 不可用时退化为普通 RAG，而不是报错；
- 证据合并按 chunk_id 去重并设总量上限，避免多轮检索把上下文撑爆。

流式问答：Agentic 阶段（规划 + 多轮检索）是同步阻塞 IO，在进入 token 生成前
完成；之后的生成仍走原有 RAGChain 流式通道，SSE 协议不变。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from app.core.errors import AppError
from app.core.logging import get_logger
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import AgenticConfig
from app.providers.llm import ChatMessage, LLMClient
from app.retrieval.query_router import QueryRouter

logger = get_logger(__name__)

AgentTool = Literal["document", "sql", "api"]

_TOOL_NAMES: dict[AgentTool, str] = {
    "document": "文档检索",
    "sql": "只读 SQL 统计",
    "api": "实时 API 查询",
}
_JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")
_PLAN_TEMPERATURE = 0.0
_PLAN_MAX_TOKENS = 512
_DOC_PREVIEW_CHARS = 240


@dataclass(frozen=True, slots=True)
class AgentStep:
    """一个可执行的工具调用：tool 决定走哪条链路，query 是该链路的问题。"""

    tool: AgentTool
    query: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class StructuredTerminal:
    """结构化工具（SQL/API）的终态结果：由编排层转成 QueryResult 返回。"""

    route: str
    answer: str
    sql: str | None = None


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """证据覆盖评估：不只看"检索到多少片段"，而是看"问题是否真的被覆盖"。

    chunk_count 不足以证明证据充分（10 个片段可能全来自同一份新制度）；
    这里按子问题覆盖、来源多样性、证据数量三个维度给出缺口，供反思阶段决策。
    """

    chunk_count: int
    source_count: int
    sub_question_coverage: float
    gaps: tuple[str, ...] = ()


@dataclass(slots=True)
class AgenticOutcome:
    """Agentic 执行结果：合并后的文档证据 + 可选的结构化终态 + 执行轨迹。"""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    terminal: StructuredTerminal | None = None
    steps: list[AgentStep] = field(default_factory=list)
    rounds: int = 0
    coverage: CoverageReport | None = None


SearchFn = Callable[[str], list[RetrievedChunk]]
ToolFn = Callable[[str], StructuredTerminal | None]


class AgenticPlanner:
    """LLM 规划器：产出初始子问题计划，以及证据不足时的补充检索计划。

    提示词要求严格 JSON 数组，解析失败就走 fallback；这里不追求复杂计划格式，
    因为工具只有三种、每轮步骤很少，过度结构化反而增加解析失败面。
    """

    def __init__(
        self,
        llm: LLMClient | None,
        router: QueryRouter,
        *,
        max_sub_queries: int,
        structured_available: bool,
        api_available: bool,
    ) -> None:
        self._llm = llm
        self._router = router
        self._max_sub_queries = max_sub_queries
        self._structured_available = structured_available
        self._api_available = api_available

    def plan(
        self, question: str, history: Sequence[tuple[str, str]]
    ) -> list[AgentStep]:
        """初始计划：LLM 拆解问题并选工具；失败回退确定性路由的单步计划。"""
        if self._llm is None:
            return [self.fallback_step(question)]
        history_text = _history_preview(history)
        system = (
            f"你是企业知识库的检索规划器。请把用户问题拆成 1~{self._max_sub_queries} 个可独立执行的"
            "子问题，并为每个子问题选择工具。\n"
            f"可选工具：{self._tools_description()}\n"
            "规则：\n"
            "- 需要精确统计/聚合时选 sql；需要实时状态（库存/订单进度等）时选 api；\n"
            "- 其余情况选 document（文档检索），这是默认工具；\n"
            "- 子问题要自包含、可独立检索，不要输出解释；\n"
            "- 只输出 JSON 数组，格式：[{\"tool\": \"document\", \"query\": \"子问题\"}]。"
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=f"{history_text}当前问题：{question}"),
        ]
        try:
            response = self._llm.chat(
                messages, temperature=_PLAN_TEMPERATURE, max_tokens=_PLAN_MAX_TOKENS
            )
            steps = self._parse_steps(response.text)
        except (AppError, ValueError) as exc:
            logger.warning("Agentic 规划失败，回退单步检索：%s", exc)
            steps = []
        if not steps:
            steps = [self.fallback_step(question)]
        return steps[: self._max_sub_queries]

    def reflect(
        self,
        question: str,
        chunk_count: int,
        previews: Sequence[str],
        round_no: int,
        coverage: CoverageReport | None = None,
    ) -> list[AgentStep]:
        """反思：按证据覆盖缺口给出补充查询；证据已足够由调用方先行判断。"""
        if self._llm is None:
            return []
        evidence = "\n".join(previews) if previews else "（无检索结果）"
        gaps = "、".join(coverage.gaps) if coverage and coverage.gaps else "无"
        coverage_text = (
            f"子问题覆盖率：{coverage.sub_question_coverage:.0%}；"
            f"不同来源数：{coverage.source_count}；"
            f"覆盖缺口：{gaps}"
            if coverage
            else "未评估"
        )
        system = (
            "你是证据覆盖检查器。不要只看证据数量，要判断：\n"
            "1) 每个子问题是否都有证据；2) 是否覆盖了必要的全部文档/版本；\n"
            "3) 数字、条件、例外是否齐全；4) 不同来源是否存在冲突。\n"
            f"如果存在证据缺口，给出 1~{self._max_sub_queries} 个补充检索子问题；"
            "如果已经足够，输出空数组 []。\n"
            f"可选工具：{self._tools_description()}\n"
            "只输出 JSON 数组：[{\"tool\": \"document\", \"query\": \"补充子问题\"}]。"
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(
                role="user",
                content=(
                    f"第 {round_no} 轮\n用户问题：{question}\n"
                    f"已检索片段数：{chunk_count}\n{coverage_text}\n证据摘要：\n{evidence}"
                ),
            ),
        ]
        try:
            response = self._llm.chat(
                messages, temperature=_PLAN_TEMPERATURE, max_tokens=_PLAN_MAX_TOKENS
            )
            return self._parse_steps(response.text)[: self._max_sub_queries]
        except (AppError, ValueError) as exc:
            logger.warning("Agentic 反思失败，结束补充检索：%s", exc)
            return []

    def _parse_steps(self, text: str) -> list[AgentStep]:
        """从模型输出中提取合法步骤；工具必须已启用，文档类子问题必须有内容。"""
        match = _JSON_ARRAY_PATTERN.search(text)
        if match is None:
            raise ValueError("模型输出中没有 JSON 数组")
        payload = json.loads(match.group(0))
        if not isinstance(payload, list):
            raise ValueError("规划结果不是数组")
        steps: list[AgentStep] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "document")).strip().lower()
            query = str(item.get("query", "")).strip()
            if tool not in _TOOL_NAMES or not query:
                continue
            if tool == "sql" and not self._structured_available:
                continue
            if tool == "api" and not self._api_available:
                continue
            steps.append(AgentStep(tool=cast(AgentTool, tool), query=query))
        return steps

    def fallback_step(self, question: str) -> AgentStep:
        """确定性兜底：使用与普通 RAG 相同的 QueryRouter 决策。"""
        decision = self._router.route(
            question,
            structured_available=self._structured_available,
            api_available=self._api_available,
        )
        tool: AgentTool = cast(AgentTool, decision.route)
        if tool not in _TOOL_NAMES:
            tool = "document"
        return AgentStep(tool=tool, query=question, reason=decision.reason)

    def _tools_description(self) -> str:
        tools: list[AgentTool] = ["document"]
        if self._structured_available:
            tools.append("sql")
        if self._api_available:
            tools.append("api")
        return "、".join(f"{name}（{_TOOL_NAMES[name]}）" for name in tools)


class AgenticRAG:
    """执行器：按计划调用工具、合并证据、按反思结果补检索，直到证据足够或轮次用完。"""

    def __init__(
        self,
        *,
        llm: LLMClient | None,
        router: QueryRouter,
        search: SearchFn,
        structured: ToolFn,
        api: ToolFn,
        config: AgenticConfig,
        structured_available: bool,
        api_available: bool,
    ) -> None:
        self._config = config
        self._search = search
        self._structured = structured
        self._api = api
        self._planner = AgenticPlanner(
            llm,
            router,
            max_sub_queries=config.max_sub_queries,
            structured_available=structured_available,
            api_available=api_available,
        )

    def run(
        self, question: str, history: Sequence[tuple[str, str]]
    ) -> AgenticOutcome:
        """完整 Agentic 循环：初始计划 → 执行 → 证据检查 → 反思补检索。"""
        outcome = AgenticOutcome()
        if self._config.use_llm_planner:
            steps = self._planner.plan(question, history)
        else:
            steps = [self._planner.fallback_step(question)]

        budget = _ExecutionBudget(
            max_tool_calls=self._config.max_tool_calls,
            deadline_ms=self._config.deadline_ms,
        )
        terminal, chunks, per_step = self._execute(steps, budget)
        outcome.steps.extend(steps)
        outcome.chunks = chunks
        outcome.rounds = 1
        outcome.coverage = _assess_coverage(steps, per_step, chunks)
        if terminal is not None:
            outcome.terminal = terminal
            return outcome

        while (
            outcome.rounds < self._config.max_rounds
            and budget.allowed()
            and _needs_more_evidence(outcome.coverage, self._config.min_evidence_chunks)
        ):
            next_round = outcome.rounds + 1
            followups = self._planner.reflect(
                question,
                len(outcome.chunks),
                _previews(outcome.chunks),
                next_round,
                outcome.coverage,
            )
            if not followups:
                break
            outcome.steps.extend(followups)
            terminal, new_chunks, per_step = self._execute(followups, budget)
            outcome.rounds = next_round
            if terminal is not None:
                outcome.terminal = terminal
                return outcome
            outcome.chunks = _merge_chunks(
                outcome.chunks, new_chunks, limit=self._config.max_evidence_chunks
            )
            outcome.coverage = _assess_coverage(followups, per_step, outcome.chunks)

        outcome.chunks = outcome.chunks[: self._config.max_evidence_chunks]
        logger.info(
            "Agentic 完成：rounds=%s chunks=%s steps=%s",
            outcome.rounds,
            len(outcome.chunks),
            [step.tool for step in outcome.steps],
        )
        return outcome

    def _execute(
        self, steps: Sequence[AgentStep], budget: _ExecutionBudget
    ) -> tuple[StructuredTerminal | None, list[RetrievedChunk], list[list[RetrievedChunk]]]:
        """执行一批步骤；遇到 SQL/API 结果即作为终态返回（不再继续检索）。

        每次工具调用前检查预算（工具调用数 + 截止时间），预算耗尽立即停止执行，
        剩余步骤不再发起任何检索/查询，避免复杂问题把成本拖到不可控。

        多个文档子问题串行执行：每次检索都走缓存与同一组 ACL 参数，
        串行实现简单且不会对下游存储造成瞬时并发压力。
        """
        chunks: list[RetrievedChunk] = []
        per_step: list[list[RetrievedChunk]] = []
        for step in steps:
            if not budget.allowed():
                logger.info("Agentic 预算耗尽，停止后续工具调用")
                break
            budget.spend()
            if step.tool == "sql":
                terminal = self._structured(step.query)
                if terminal is not None:
                    return terminal, chunks, per_step
                continue
            if step.tool == "api":
                terminal = self._api(step.query)
                if terminal is not None:
                    return terminal, chunks, per_step
                continue
            found = self._search(step.query)
            per_step.append(found)
            chunks.extend(found)
        return (
            None,
            _merge_chunks([], chunks, limit=self._config.max_evidence_chunks),
            per_step,
        )


@dataclass(slots=True)
class _ExecutionBudget:
    """单次 Agentic 请求的执行预算：工具调用次数 + 墙钟截止时间。"""

    max_tool_calls: int
    deadline_ms: int
    started: float = field(default_factory=time.monotonic)
    used: int = 0

    def allowed(self) -> bool:
        if self.used >= self.max_tool_calls:
            return False
        elapsed_ms = (time.monotonic() - self.started) * 1000
        return elapsed_ms <= self.deadline_ms

    def spend(self) -> None:
        self.used += 1


def _assess_coverage(
    steps: Sequence[AgentStep],
    per_step: Sequence[list[RetrievedChunk]],
    chunks: Sequence[RetrievedChunk],
) -> CoverageReport:
    """评估证据覆盖：子问题是否都有命中、来源是否多样、证据量是否达标。

    per_step 与 steps 中"文档类步骤"一一对应（SQL/API 终态不进入这里）。
    """
    document_count = sum(1 for step in steps if step.tool == "document")
    covered_count = sum(1 for result in per_step if result)
    gaps: list[str] = []
    if document_count and covered_count == 0:
        gaps.append("没有任何文档证据")
    elif covered_count < document_count:
        gaps.append(f"{document_count - covered_count} 个子问题缺少证据")
    source_count = len({chunk.doc_id for chunk in chunks})
    if document_count >= 2 and source_count < 2:
        gaps.append("证据来源单一")
    coverage = covered_count / document_count if document_count else 0.0
    return CoverageReport(
        chunk_count=len(chunks),
        source_count=source_count,
        sub_question_coverage=coverage,
        gaps=tuple(gaps),
    )


def _needs_more_evidence(coverage: CoverageReport | None, min_chunks: int) -> bool:
    """是否需要补检索：数量不足或覆盖存在缺口（覆盖缺口优先）。"""
    if coverage is None:
        return True
    if coverage.chunk_count < min_chunks:
        return True
    return bool(coverage.gaps)


def _merge_chunks(
    existing: list[RetrievedChunk], incoming: list[RetrievedChunk], *, limit: int
) -> list[RetrievedChunk]:
    """按 chunk_id 去重合并，保持首次出现顺序（子问题优先级即出现顺序）。"""
    merged = list(existing)
    seen = {chunk.chunk_id for chunk in merged}
    for chunk in incoming:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged[:limit]


def _previews(chunks: Sequence[RetrievedChunk]) -> list[str]:
    """证据摘要：标题 + 内容前 N 字符，供反思提示词判断覆盖度。"""
    previews: list[str] = []
    for chunk in chunks[:8]:
        title = chunk.title or chunk.doc_id
        content = chunk.content[:_DOC_PREVIEW_CHARS].replace("\n", " ")
        previews.append(f"- [{title}] {content}")
    return previews


def _history_preview(history: Sequence[tuple[str, str]]) -> str:
    """最近两轮对话压缩成一行，帮助规划器理解追问指代。"""
    if not history:
        return ""
    recent = history[-4:]
    text = "；".join(f"{role}: {content[:120]}" for role, content in recent)
    return f"对话上下文：{text}\n"
