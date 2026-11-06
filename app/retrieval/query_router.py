"""查询路由（v2.0）：判断一个问题该走文档检索还是结构化 SQL 查询。

为什么需要路由：
    文档检索擅长"制度/流程/条款"等非结构化知识；"本季度各部门报销总额是多少"
    这类问题需要精确统计值，从片段里猜数字必然不可靠。路由把两类问题分流到
    各自最合适的检索器，避免用一种能力硬解所有问题。

设计取舍：
    - 确定性规则优先（无额外 LLM 延迟、可预测、可单测）；
    - 只在知识库显式启用结构化查询（structured.enabled）时才允许路由到 SQL；
    - 调用方可用 mode 强制指定路由（document / structured），便于调试与评测；
    - 规则拿不准时一律回退文档检索——文档检索的失败代价远小于误跑 SQL。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.core.errors import StructuredQueryError

QueryRoute = Literal["document", "sql", "api"]
RequestedMode = Literal["auto", "document", "structured"]

# 统计/聚合语义的强特征词：命中即认为"要精确数字"，而不是从文档里读描述。
# 刻意不收录"多少/数量"这类弱特征——"年假多少天"是典型文档问题，误路由代价高。
_STRUCTURED_MARKERS = (
    "统计",
    "总数",
    "总量",
    "合计",
    "汇总",
    "求和",
    "加总",
    "平均值",
    "均值",
    "排名",
    "排行",
    "占比",
    "百分比",
    "同比",
    "环比",
    "趋势",
    "分布",
    "去重",
    "count(",
    "group by",
    "order by",
    "how many",
    "how much",
    "average",
    "total ",
    " sum(",
    " avg(",
    "distinct",
)
# 用户直接把 SQL 贴进来时（如调试/评测），也走结构化链路（仍需过 SQLGuard）
# 实时状态类问题的强特征词：这些问题的答案会随时间变化，必须现查接口而不是读文档
_REALTIME_MARKERS = (
    "实时",
    "当前状态",
    "当前库存",
    "当前余额",
    "最新数据",
    "此刻",
    "现在",
    "库存",
    "工单状态",
    "订单状态",
    "live",
    "real-time",
    "realtime",
    "current status",
    "status of",
)
_SQL_START_PATTERN = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# 文档语境词：出现时说明用户问的是"制度/条款怎么规定"，即使命中"总额"也不该跑 SQL
_DOCUMENT_CONTEXT_MARKERS = (
    "制度",
    "规定",
    "政策",
    "条款",
    "流程",
    "标准",
    "办法",
    "规范",
    "手册",
    "文档",
)
# 弱实时词：单独出现不足以证明需要现查接口（"现在年假多少天"仍是文档问题）
_WEAK_REALTIME_MARKERS = frozenset({"现在", "此刻", "当前", "最新", "实时"})


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """路由结果：route 决定链路，confidence/intent/reason 供日志与降级判断。

    confidence 语义：
    - >=0.8：强特征，可直接执行；
    - 0.5~0.8：可执行但存在歧义，调用方应保留降级/复核空间；
    - <0.5：模糊问题，默认回退文档检索（不直接跑 SQL/API）。
    """

    route: QueryRoute
    reason: str
    confidence: float = 1.0
    intent: str = "document_lookup"


class QueryRouter:
    """无状态的规则路由器；输入问题与"结构化能力是否可用"，输出目标链路。"""

    def route(
        self,
        question: str,
        *,
        structured_available: bool,
        api_available: bool = False,
        requested: RequestedMode = "auto",
    ) -> RouteDecision:
        """显式 mode 优先；auto 模式下按规则判定，拿不准回退文档检索。

        优先级：用户显式指定 > SQL 统计特征 > 实时 API 特征 > 文档检索。
        SQL 优先于 API 的原因：统计问题需要聚合历史数据，API 通常只返回当前快照。
        """
        if requested == "structured":
            if not structured_available:
                raise StructuredQueryError("当前知识库未启用结构化查询（structured.enabled=false）")
            return RouteDecision("sql", "explicit_structured", confidence=1.0, intent="explicit")
        if requested == "document":
            return RouteDecision(
                "document", "explicit_document", confidence=1.0, intent="explicit"
            )

        text = question.strip().lower()
        if _SQL_START_PATTERN.match(text):
            if not structured_available:
                return RouteDecision("document", "sql_unavailable", confidence=1.0)
            return RouteDecision(
                "sql", "sql_statement", confidence=1.0, intent="sql_statement"
            )

        structured_hits = (
            [marker for marker in _STRUCTURED_MARKERS if marker in text]
            if structured_available
            else []
        )
        realtime_hits = (
            [marker for marker in _REALTIME_MARKERS if marker in text]
            if api_available
            else []
        )
        strong_realtime = [
            marker for marker in realtime_hits if marker not in _WEAK_REALTIME_MARKERS
        ]
        document_context = any(
            marker in text for marker in _DOCUMENT_CONTEXT_MARKERS
        )

        if structured_hits and realtime_hits:
            # 多意图（如"统计当前库存"）：统计口径需要历史数据，优先 SQL 但降低置信度
            return RouteDecision(
                "sql",
                f"multi_intent:{structured_hits[0]}+{realtime_hits[0]}",
                confidence=0.5,
                intent="mixed_analysis",
            )
        if structured_hits:
            if document_context and len(structured_hits) == 1:
                # "制度中规定的总额上限是多少"：命中统计词但同时是制度语境 → 文档检索
                return RouteDecision(
                    "document",
                    f"ambiguous:{structured_hits[0]}+document_context",
                    confidence=0.35,
                    intent="policy_lookup",
                )
            confidence = 0.85 if len(structured_hits) >= 2 else 0.6
            return RouteDecision(
                "sql",
                f"marker:{structured_hits[0]}",
                confidence=confidence,
                intent="aggregate_query",
            )
        if strong_realtime:
            return RouteDecision(
                "api",
                f"marker:{strong_realtime[0]}",
                confidence=0.8,
                intent="realtime_lookup",
            )
        if realtime_hits:
            # 只有"现在/当前"这类弱词：不足以证明问题依赖实时状态
            return RouteDecision(
                "document",
                f"weak_realtime:{realtime_hits[0]}",
                confidence=0.4,
            )
        if not structured_available and not api_available:
            return RouteDecision("document", "structured_disabled", confidence=0.7)
        return RouteDecision("document", "no_structured_marker", confidence=0.7)
