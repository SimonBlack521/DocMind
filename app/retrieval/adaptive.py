"""自适应召回：按问题复杂度调整 Top-K，并在分数断层处提前停止。

为什么需要动态 K（对应 9.3）：
- K 写死有两个方向的错误：太小会把证据漏在召回阶段（Rerank 救不回来），
  太大则把大量噪声送进精排，推理量和延迟一起上升，还会挤占 LLM 上下文预算；
- FAQ 类简单问题通常少量候选就足够；多跳/对比/流程类问题证据分散，
  需要更大候选池才能保证 Recall 底线；
- 分数断层（前几名很高、之后断崖下跌）说明后续候选基本是噪声，
  此时继续硬捞满 K 只会增加下游成本，属于"召回阶段早停"（不是精排早停）。

本模块只做纯计算，不访问存储与模型，便于单元测试与离线评测复用。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

# 多跳/复合问题的典型语言标记：出现越多，越需要放大候选池
_MULTI_HOP_MARKERS = (
    "并且",
    "以及",
    "同时",
    "分别",
    "各自",
    "对比",
    "比较",
    "区别",
    "步骤",
    "流程",
    "原因",
    "为什么",
    "如何",
    "怎样",
    "哪些",
    "条件",
    "例外",
)

# 复杂度阈值：低于 SIMPLE 取小 K，高于 COMPLEX 取大 K，中间保持基准 K
_SIMPLE_BELOW = 0.35
_COMPLEX_ABOVE = 0.65


class _Scored(Protocol):
    """只需要 score 字段：向量命中 / 关键词命中 / 统一后的 chunk 都适用。"""

    score: float


_ScoredT = TypeVar("_ScoredT", bound=_Scored)


def estimate_query_complexity(question: str) -> float:
    """把问题文本折算成 0~1 的复杂度分。

    只使用可解释的轻量特征（长度 + 多跳标记 + 分句数量），不引入分类模型：
    1. 长度：40 字符视为"足够复杂"，更长不再继续加分；
    2. 多跳标记：每个 +0.15，最多 +0.45；
    3. 分句数量：多个问号/分号通常意味着多个子问题。
    """
    text = question.strip()
    if not text:
        return 0.0
    length_part = min(len(text) / 40.0, 1.0)
    marker_part = min(
        sum(1 for marker in _MULTI_HOP_MARKERS if marker in text) * 0.15, 0.45
    )
    clause_part = min((text.count("？") + text.count("?") + text.count("；")) * 0.1, 0.2)
    return min(1.0, 0.2 + 0.45 * length_part + marker_part + clause_part)


def resolve_dynamic_top_k(
    question: str,
    *,
    base_k: int,
    min_k: int = 0,
    max_k: int = 0,
) -> int:
    """按问题复杂度在 [min_k, max_k] 内调整召回 K；未配置上下限时保持 base_k。

    - 简单问题：`base_k / 2`，但不低于 min_k；
    - 复杂问题：`base_k × 2`，但不超过 max_k；
    - 中间地带：保持 base_k，避免频繁抖动。
    """
    if base_k <= 0:
        return base_k
    if min_k <= 0 and max_k <= 0:
        return base_k

    low = min_k if min_k > 0 else base_k
    high = max_k if max_k > 0 else base_k
    if high < low:
        high = low

    complexity = estimate_query_complexity(question)
    if complexity < _SIMPLE_BELOW:
        target = max(low, base_k // 2)
    elif complexity > _COMPLEX_ABOVE:
        target = min(high, max(base_k, base_k * 2))
    else:
        target = base_k
    return max(1, min(max(target, low), high))


# 查询类型 → 召回候选池缩放（清单第三优先级第 14 项）：
# 事实查询答案集中，小候选池即可；多跳/对比/流程类需要多组证据并行召回，放大候选池
_TYPE_TOP_K_SCALE: dict[str, float] = {
    "faq": 0.5,
    "comparison": 2.0,
    "multi_hop": 2.0,
    "procedural": 1.5,
}


def adjust_top_k_by_type(
    top_k: int,
    *,
    query_type: str,
    base_k: int,
    min_k: int,
    max_k: int,
) -> int:
    """按查询分类结果修正召回 Top-K（数据驱动的规则层）。

    只对已知类型（faq/multi_hop/procedural/comparison）生效，general 原样返回；
    结果夹在 [min_k, max_k] 内（未配置时用 base_k 推导的默认区间）。
    """
    scale = _TYPE_TOP_K_SCALE.get(query_type)
    if scale is None:
        return top_k
    low = min_k if min_k > 0 else max(1, base_k // 2)
    high = max_k if max_k > 0 else max(base_k, base_k * 2)
    if high < low:
        high = low
    return max(low, min(high, round(top_k * scale)))


def truncate_at_score_cliff(
    hits: Sequence[_ScoredT],
    *,
    min_keep: int = 1,
    ratio: float = 0.0,
) -> list[_ScoredT]:
    """在单路召回结果的最大分数断层处截断（ratio<=0 或候选太少时原样返回）。

    判定方式：相邻两名分数的相对跌幅 `|s_i - s_{i+1}| / |s_i|`。
    跌幅 >= ratio 视为断层，直接丢掉断层之后的候选；至少保留 min_keep 条，
    保证精排仍有足够候选可用。只处理单路内部（同一打分尺度），
    不跨路比较——BM25 与余弦分数的量纲不可比。
    """
    items = list(hits)
    if ratio <= 0 or len(items) <= min_keep:
        return items

    cut_at = len(items)
    # 从 min_keep 之后开始找：前 min_keep 条无论如何都要保留
    for index in range(min_keep - 1, len(items) - 1):
        current = float(items[index].score)
        nxt = float(items[index + 1].score)
        drop = abs(current - nxt)
        relative = drop / max(abs(current), 1e-9)
        if relative >= ratio:
            cut_at = index + 1
            break
    return items[:cut_at]
