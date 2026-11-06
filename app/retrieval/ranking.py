"""检索结果治理加权：把来源权威度与时序新鲜度折算成排序增益。

为什么放在 RRF 之后、精排之前：
- RRF 只按"两路排名"融合，并不感知文档的权威级别与采集时间；
- 精排模型只看"问题-片段语义相关性"，同样不感知治理信息；
- 在融合结果上再乘一个治理系数，能在候选筛选阶段把同一内容的新版本文档、
  高权威来源排到前面；精排开启时仍由精排做最终裁决（治理加权只影响送到精排的候选集）。

加权公式（乘法而非加法，保证多因子可叠加且单调）：
    weight = (1 + authority_weight × level/100) × (1 + recency_weight × decay)
    decay  = 0.5 ^ (age_days / half_life_days)

- 权威度：metadata["authority_level"]（0-100，缺省按 0 处理）；
- 新鲜度：优先 metadata["collected_at"]，其次 updated_at / created_at；
  缺失时间戳时 decay=0（不额外加分，也不倒扣）；
- 两个权重都为 0 时完全不做排序变更，便于按知识库关闭该能力。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from app.domain.dto import RetrievedChunk


def _parse_timestamp(value: Any) -> datetime | None:
    """把 ISO 字符串 / datetime 解析成带时区的 datetime；无法解析按"无时间"处理。"""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _authority_boost(metadata: dict[str, Any], authority_weight: float) -> float:
    raw = metadata.get("authority_level", 0)
    try:
        level = float(raw)
    except (TypeError, ValueError):
        level = 0.0
    # 钳制到 0-100：脏数据不能产生超过设定倍率的增益
    level = min(max(level, 0.0), 100.0)
    return 1.0 + authority_weight * (level / 100.0)


def _recency_boost(
    metadata: dict[str, Any], recency_weight: float, half_life_days: float, now: datetime
) -> float:
    timestamp = _parse_timestamp(
        metadata.get("collected_at") or metadata.get("updated_at") or metadata.get("created_at")
    )
    if timestamp is None or half_life_days <= 0:
        return 1.0
    age_days = max((now - timestamp).total_seconds() / 86400.0, 0.0)
    decay = 0.5 ** (age_days / half_life_days)
    return 1.0 + recency_weight * decay


def governance_weight(
    metadata: dict[str, Any],
    *,
    authority_weight: float,
    recency_weight: float,
    half_life_days: float,
    now: datetime | None = None,
) -> float:
    """单条片段的治理加权系数；metadata 中缺少治理字段时恒为 1。"""
    if authority_weight <= 0 and recency_weight <= 0:
        return 1.0
    current = now or datetime.now(UTC)
    return _authority_boost(metadata, max(authority_weight, 0.0)) * _recency_boost(
        metadata, max(recency_weight, 0.0), half_life_days, current
    )


def apply_governance_weighting(
    chunks: Sequence[RetrievedChunk],
    *,
    authority_weight: float,
    recency_weight: float,
    half_life_days: float,
    now: datetime | None = None,
) -> list[RetrievedChunk]:
    """治理加权重排：final_score = rrf_score × weight。

    乘法而非加法，多因子可叠加、单调；排序键使用 final_score（相关性加权后），
    避免出现"高权威但低相关性"的 chunk 把真正相关但低权威的候选完全挤掉。
    两个治理权重都为 0 时原样返回（便于按知识库关闭该能力）。

    写回 metadata["governance_weight"] 和 metadata["final_score"]，便于评测集
    与线上排查解释排序原因；chunk.score 保持为"原始融合分"，语义清晰。
    """
    if not chunks:
        return []
    if authority_weight <= 0 and recency_weight <= 0:
        return list(chunks)

    current = now or datetime.now(UTC)
    ranked: list[tuple[float, int, RetrievedChunk]] = []
    for position, chunk in enumerate(chunks):
        weight = governance_weight(
            chunk.metadata,
            authority_weight=authority_weight,
            recency_weight=recency_weight,
            half_life_days=half_life_days,
            now=current,
        )
        final_score = float(chunk.score) * weight
        chunk.metadata["governance_weight"] = round(weight, 6)
        chunk.metadata["final_score"] = round(final_score, 6)
        # 主键 final_score（相关性×治理），次键 -position（稳定：高 final_score 同分时
        # 保留原融合顺序）；绝不让治理系数单独决定排名——相关性是第一道门槛
        ranked.append((final_score, -position, chunk))

    ranked.sort(reverse=True)
    return [chunk for _, _, chunk in ranked]
