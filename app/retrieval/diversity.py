"""候选去重与 MMR 多样性选择（对应 9.5）。

去重口径：
- 同一 chunk_id 重复（多路召回 / 父块回取）只保留首次出现；
- 同一文档 + 同一原文区间（char_start/char_end）视为同一证据，只保留排名更靠前的；
- 正文内容指纹相同（不同文档抄同一段文字）也去重，避免上下文重复。

MMR（Maximal Marginal Relevance）：
    score = λ × Sim(candidate, query) − (1−λ) × max Sim(candidate, selected)
- λ=1 时退化为纯相关性排序（关闭多样性）；
- λ 越小越强调"与已选片段不同"，避免 Top-N 全来自同一段/同一文档。

多样性相似度有两档：
1. 向量余弦距离（推荐，v1.3-4）：调用方传入 chunk_vectors 时启用，语义级去重，
   能把"同事实不同表述"的候选判为相似；
2. 词面 Jaccard（兜底）：无向量时回退，轻量、确定性强，在 Top-N ≤ 50 上开销可忽略。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from app.core.text import token_set as _token_set
from app.domain.dto import RetrievedChunk


def _normalized_content(text: str) -> str:
    """压缩空白并小写，避免仅空白/大小写差异造成的重复。"""
    return " ".join(text.lower().split())


def _content_fingerprint(text: str) -> str:
    return hashlib.sha256(_normalized_content(text).encode("utf-8")).hexdigest()


def dedupe_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """按 chunk_id → 原文区间 → 内容指纹三级去重，保留排名更靠前的候选。"""
    seen_ids: set[str] = set()
    seen_spans: set[tuple[str, int, int]] = set()
    seen_fingerprints: set[str] = set()
    kept: list[RetrievedChunk] = []

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            continue
        seen_ids.add(chunk.chunk_id)

        start = chunk.metadata.get("char_start")
        end = chunk.metadata.get("char_end")
        if isinstance(start, int) and isinstance(end, int):
            span = (chunk.doc_id, start, end)
            if span in seen_spans:
                continue
            seen_spans.add(span)

        fingerprint = _content_fingerprint(chunk.content)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        kept.append(chunk)
    return kept


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


# 结构性重复惩罚（清单第三优先级第 13 项）：同一文档 / 同一章节 / 相邻 chunk 的
# 内容天然高度重叠，词面或语义相似度之外再叠加来源多样性惩罚，
# 让 Top-N 尽量覆盖不同文档与章节，而不是同一段落的连续切片
_STRUCT_SAME_DOC = 0.3
_STRUCT_SAME_SECTION = 0.2
_STRUCT_ADJACENT = 0.1


def _structural_penalty(candidate: RetrievedChunk, chosen: RetrievedChunk) -> float:
    """候选与已选片段的结构性重复惩罚（0 表示无结构重叠，上限 1.0）。"""
    if candidate.doc_id != chosen.doc_id:
        return 0.0
    penalty = _STRUCT_SAME_DOC
    cand_section = str(candidate.metadata.get("section_path") or "")
    chosen_section = str(chosen.metadata.get("section_path") or "")
    if cand_section and cand_section == chosen_section:
        penalty += _STRUCT_SAME_SECTION
    cand_index = candidate.metadata.get("chunk_index")
    chosen_index = chosen.metadata.get("chunk_index")
    if (
        isinstance(cand_index, int)
        and isinstance(chosen_index, int)
        and abs(cand_index - chosen_index) == 1
    ):
        penalty += _STRUCT_ADJACENT
    return min(1.0, penalty)


def _normalize(vec: list[float]) -> list[float] | None:
    """L2 归一化向量；零向量返回 None（避免除零）。"""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return None
    return [v / norm for v in vec]


def _cosine(left: list[float], right: list[float]) -> float:
    """两个已归一化向量的余弦相似度（点积）。"""
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalized_vectors(vectors: Sequence[list[float]]) -> list[list[float]]:
    """批量归一化，零向量跳过（保持 None 占位，调用方需过滤）。"""
    out: list[list[float]] = []
    for vec in vectors:
        normed = _normalize(vec)
        if normed is not None:
            out.append(normed)
    return out


def _normalized_relevance(chunks: Sequence[RetrievedChunk]) -> list[float]:
    """把各候选分数压到 0~1：精排分 / RRF 分都可能是不同量纲，先相对归一。"""
    scores = [float(chunk.score) for chunk in chunks]
    low = min(scores)
    high = max(scores)
    if high <= low:
        return [0.5] * len(chunks)
    span = high - low
    return [(score - low) / span for score in scores]


def mmr_select(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    top_n: int,
    lambda_: float = 1.0,
    chunk_vectors: Sequence[list[float]] | None = None,
    query_vector: list[float] | None = None,
) -> list[RetrievedChunk]:
    """在已排序候选上做 MMR 选择，返回 top_n 条（λ>=1 时直接截断）。

    v1.3-4 向量升级：
    - 传入 chunk_vectors 时，多样性惩罚用余弦距离（语义级）；
    - 同时传入 query_vector 时，用 query-chunk 余弦补充相关性判断（替代词面覆盖）；
    - 不传任何向量时回退到词面 Jaccard，保持旧行为。
    这是"排序后出口"的轻量多样性约束，不替代精排的相关性判断。
    """
    items = list(chunks)
    if top_n <= 0 or not items:
        return []
    if lambda_ >= 1.0:
        return items[:top_n]

    relevance = _normalized_relevance(items)
    token_sets = [_token_set(chunk.content) for chunk in items]
    query_tokens = _token_set(question)

    # 准备向量（可选）
    use_vectors = False
    normed_chunks: list[list[float]] = []
    normed_query: list[float] | None = None
    if chunk_vectors is not None and len(chunk_vectors) == len(items):
        normed_chunks = _normalized_vectors(chunk_vectors)
        if len(normed_chunks) == len(items):  # 没有零向量才能启用
            use_vectors = True
            if query_vector is not None:
                normed_query = _normalize(query_vector)

    # relevance 补充：query 词面覆盖（始终算）+ 可选 query-chunk 余弦
    if use_vectors and normed_query is not None:
        # 向量版：0.5 × 模型相关性 + 0.3 × query 余弦 + 0.2 × 词面覆盖
        base_relevance = [
            0.5 * rel
            + 0.3 * _cosine(normed_query, vec)
            + 0.2 * _jaccard(query_tokens, tokens)
            for rel, vec, tokens in zip(relevance, normed_chunks, token_sets, strict=True)
        ]
    else:
        # 旧版：0.7 × 模型相关性 + 0.3 × 词面覆盖
        base_relevance = [
            0.7 * rel + 0.3 * _jaccard(query_tokens, tokens)
            for rel, tokens in zip(relevance, token_sets, strict=True)
        ]

    selected: list[int] = []
    remaining = list(range(len(items)))
    diversity_penalty = [0.0] * len(items)

    while remaining and len(selected) < top_n:
        best_index = remaining[0]
        best_mmr = float("-inf")
        for index in remaining:
            score = lambda_ * base_relevance[index] - (1.0 - lambda_) * diversity_penalty[index]
            if score > best_mmr:
                best_mmr = score
                best_index = index
        selected.append(best_index)
        remaining.remove(best_index)
        # 已选片段的相似度提高其余候选的"重复惩罚"（取与已选集合的最大相似度）；
        # 叠加结构性惩罚：同文档/同章节/相邻块的重复不依赖词面也能识别
        best_chunk = items[best_index]
        if use_vectors:
            best_vec = normed_chunks[best_index]
            for index in remaining:
                similarity = _cosine(normed_chunks[index], best_vec)
                similarity = min(
                    1.0, similarity + _structural_penalty(items[index], best_chunk)
                )
                if similarity > diversity_penalty[index]:
                    diversity_penalty[index] = similarity
        else:
            best_tokens = token_sets[best_index]
            for index in remaining:
                similarity = _jaccard(token_sets[index], best_tokens)
                similarity = min(
                    1.0, similarity + _structural_penalty(items[index], best_chunk)
                )
                if similarity > diversity_penalty[index]:
                    diversity_penalty[index] = similarity

    return [items[index] for index in selected]
