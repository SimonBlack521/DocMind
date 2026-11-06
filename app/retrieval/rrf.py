"""RRF（Reciprocal Rank Fusion）融合：多路检索结果按 chunk_id 去重后加权求和。

公式：score(d) = Σ weight_i / (k + rank_i + 1)

数学含义：
- rank 从 0 开始，第 1 名贡献 weight/(k+1)，第 2 名贡献 weight/(k+2)……
  名次越靠前贡献越大；倒数函数让相邻名次的差距逐渐收窄（衰减平滑），
  不会出现“第 1 名碾压后面所有”的极端分布；
- k 是平滑因子（默认 60，源自 RRF 原始论文的经验值）：k 越大，头部与尾部的
  单路贡献差距越小，越不容易被某一路的高排名“一票定音”；
- 不同检索器的原始分数尺度不可比（BM25 分数没有上界，向量相似度在 [-1, 1]，
  直接相加没有意义），RRF 只使用“名次”这一天然可比较的信息；
- weight_i 是每一路的权重：向量路权重大表示更信任语义召回，BM25 权重大表示
  更信任精确词匹配；同一 chunk 被多路召回时分数累加——这是混合检索的核心收益，
  两路都认为相关的文档会稳定排在最前。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from app.core.errors import ConfigurationError
from app.domain.dto import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    channels: Sequence[Sequence[RetrievedChunk]],
    *,
    weights: Sequence[float] | None = None,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """融合多路检索结果，返回按融合分降序的候选集。

    - 同一个 chunk_id 在多路出现时只保留一份，分数累加（RRF 的核心收益）；
    - 同一路里重复出现只按第一次（最高名次）计分，避免某一路反复加分；
    - 排序在分数相同时按 chunk_id 兜底，保证结果稳定可复现。
    """
    # k 是平滑因子，k <= 0 会让头部优势无限尖锐，失去融合意义，属于配置错误
    if k <= 0:
        raise ConfigurationError("RRF 的 k 必须大于 0")

    if weights is None:
        # 未显式给权重时各路等权（1.0）
        normalized_weights = [1.0] * len(channels)
    else:
        # 权重数量必须与检索路数一一对应，错位会静默放大/压缩某一路的贡献
        if len(weights) != len(channels):
            raise ConfigurationError(
                "RRF 权重数量与检索路数不一致",
                details={"channels": len(channels), "weights": len(weights)},
            )
        normalized_weights = [float(weight) for weight in weights]
        # 负权重会让“被召回”变成减分项，违背融合的语义，直接拒绝
        if any(weight < 0 for weight in normalized_weights):
            raise ConfigurationError("RRF 权重不能为负数")

    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for channel_index, hits in enumerate(channels):
        weight = normalized_weights[channel_index]
        if weight == 0:
            # 权重为 0 的路整体跳过：配置层面关闭某一路，等价于不参与融合
            continue
        seen: set[str] = set()
        for rank, chunk in enumerate(hits):
            if chunk.chunk_id in seen:
                # 同一路内重复出现只按第一次（最高名次）计分，防止单路反复加分
                continue
            seen.add(chunk.chunk_id)
            # 核心公式：累加该路贡献 weight / (k + rank + 1)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank + 1)
            # merged 只保留首次出现的 chunk 对象（承载 content/metadata），分数统一写回
            merged.setdefault(chunk.chunk_id, chunk)

    fused = [replace(chunk, score=scores[chunk_id]) for chunk_id, chunk in merged.items()]
    # 按融合分降序；分数相同按 chunk_id 兜底，保证结果稳定可复现
    fused.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
    return fused
