"""查询分类：按问题类型（FAQ / 多跳 / 流程 / 对比）选择检索策略。

分类结果决定：
- 是否启用 Query Decomposition（多跳/流程类问题拆成子问题分别召回再合并）；
- RRF 融合权重（对比类问题给向量路更高权重，因为需要跨段落语义匹配）；
- 精排候选池上限（多跳问题需要更大候选池）。

实现策略：
- 规则优先（轻量、快速、确定性强）：覆盖大部分 FAQ / 简单问题；
- LLM 兜底（复杂问题用 LLM 判断类型）：失败直接回退"general"，不阻塞主链路。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.core.errors import AppError
from app.core.logging import get_logger
from app.providers.llm import ChatMessage, LLMClient

logger = get_logger(__name__)

QueryType = Literal["faq", "multi_hop", "procedural", "comparison", "general"]


@dataclass(frozen=True, slots=True)
class QueryClassification:
    query_type: QueryType
    confidence: float  # 0~1；规则命中时接近 1.0，LLM 判断后给 0.5~0.9


# 规则特征：每种问题类型的语言标记（中文/英文混合）
_MULTI_HOP_MARKERS = (
    "为什么", "原因", "哪些", "条件", "例外", "分别", "同时", "以及",
    "how", "why", "what factors",
)
_PROCEDURAL_MARKERS = (
    "步骤", "流程", "怎么做", "如何申请", "申请流程", "操作指南",
    "step", "procedure", "process", "how to",
)
_COMPARISON_MARKERS = (
    "对比", "比较", "区别", "差异", "不同", "新旧", "哪个好", "vs", "versus",
)

# 分句数：多个问号/分号通常意味着多个子问题
def _clause_count(text: str) -> int:
    return sum(text.count(mark) for mark in ("？", "?", "；", ";", "、")) + 1


def rule_based_classify(query: str) -> QueryClassification:
    """规则分类：优先用于 FAQ / 简单问题（响应快、确定性强）。"""
    text = query.strip().lower()
    length = len(query.strip())
    clauses = _clause_count(query)

    # 先看标记，后看长度/分句数（标记优先级高于启发式）
    if any(marker in text for marker in _COMPARISON_MARKERS):
        return QueryClassification("comparison", confidence=0.9)
    if clauses >= 3 or any(marker in text for marker in _MULTI_HOP_MARKERS):
        return QueryClassification("multi_hop", confidence=0.85)
    if any(marker in text for marker in _PROCEDURAL_MARKERS):
        return QueryClassification("procedural", confidence=0.9)

    # 长度短 + 分句少 = 大概率 FAQ
    if length <= 20 and clauses <= 2:
        return QueryClassification("faq", confidence=0.8)

    return QueryClassification("general", confidence=0.3)


_CLASSIFY_SYSTEM = (
    "你是企业知识库检索助手。请把用户问题归类为以下类型之一，只输出一个标签：\n"
    "- faq: 简单事实查询，单条证据即可回答（如'年假多少天'）；\n"
    "- multi_hop: 需要多个证据组合才能回答（如'报销需要哪些材料以及流程是什么'）；\n"
    "- procedural: 操作步骤 / 流程类（如'如何申请报销'）；\n"
    "- comparison: 对比 / 差异类（如'新旧制度的区别'）；\n"
    "- general: 其他不确定的通用问题。\n"
    "只输出标签，不要解释、不要换行。"
)

_DECOMPOSE_SYSTEM = (
    "你是企业知识库检索助手。用户问题需要拆解为独立的检索子问题，"
    "让每个子问题可以单独在知识库中找到证据。要求：\n"
    "1. 每个子问题完整、自包含，包含原问题的关键实体和限定条件；\n"
    "2. 只输出 2-3 个子问题，每行一条，不要编号或前缀；\n"
    "3. 不要回答问题，只生成检索用的子问题。"
)

# 去掉 LLM 偶尔带出的 "1." "-" "*" 等列表前缀
_DECOMPOSE_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.、)）])\s*")
_MAX_SUBQUERY_LEN = 150
_MAX_SUBQUERIES = 3


class QueryClassifier:
    """查询分类器：规则优先、LLM 兜底。

    外部注入 LLMClient 才会启用 LLM 兜底；测试/降级场景传 None 即只用规则。
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        rule_confidence_threshold: float = 0.6,
    ) -> None:
        self._llm = llm
        self._threshold = rule_confidence_threshold

    def classify(self, query: str) -> QueryClassification:
        rule_result = rule_based_classify(query)
        if rule_result.confidence >= self._threshold:
            return rule_result
        # 规则不确定时才走 LLM，失败回退 rule_result
        if self._llm is None:
            return rule_result
        try:
            response = self._llm.chat(
                [
                    ChatMessage(role="system", content=_CLASSIFY_SYSTEM),
                    ChatMessage(role="user", content=query),
                ],
                temperature=0.0,
                max_tokens=20,
            )
        except AppError as exc:
            logger.warning("LLM 分类失败，回退规则结果：%s", exc.message)
            return rule_result
        label = response.text.strip().lower()
        if label in ("faq", "multi_hop", "procedural", "comparison", "general"):
            return QueryClassification(label, confidence=0.7)
        logger.warning("LLM 分类输出非预期标签 %s，回退规则", label)
        return rule_result

    def decompose(
        self, query: str, *, classification: QueryClassification | None = None
    ) -> list[str]:
        """多跳/流程/对比类问题拆子问题（v1.3-3）。

        返回子问题列表（不含原问题），供调用方并入 Multi-Query variants；
        非多意图问题直接返回空列表，LLM 失败也返回空（不阻塞主链路）。
        对比类（"比较新版和旧版"）同样需要多组子查询并行召回，
        避免单边证据遗漏（清单第三优先级第 14 项）。
        classification 传入可复用调用方已有的分类结果，避免重复调用 LLM。
        """
        base = query.strip()
        if not base or self._llm is None:
            return []
        cls = classification if classification is not None else self.classify(base)
        if cls.query_type not in ("multi_hop", "procedural", "comparison"):
            return []
        try:
            response = self._llm.chat(
                [
                    ChatMessage(role="system", content=_DECOMPOSE_SYSTEM),
                    ChatMessage(role="user", content=base),
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except AppError as exc:
            logger.warning("Query Decomposition LLM 失败，跳过：%s", exc.message)
            return []
        subqueries: list[str] = []
        for line in response.text.splitlines():
            cleaned = _DECOMPOSE_LIST_PREFIX.sub("", line).strip()
            if not cleaned or len(cleaned) > _MAX_SUBQUERY_LEN:
                continue
            if cleaned == base or cleaned in subqueries:
                continue
            subqueries.append(cleaned)
            if len(subqueries) >= _MAX_SUBQUERIES:
                break
        if subqueries:
            logger.info(
                "Query Decomposition 产出 %d 个子问题 type=%s",
                len(subqueries),
                cls.query_type,
            )
        return subqueries
