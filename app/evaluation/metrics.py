"""离线评测指标：检索质量、证据完整性、冗余与成本、拒答质量。

这些函数只做纯计算、不依赖数据库/模型，便于在 pytest 与脚本中复用；
评测脚本负责取数（检索结果）与写报告，指标口径集中在这里，避免两份实现漂移。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from app.domain.dto import RetrievedChunk


def recall_at_k(
    ranked_relevant: Sequence[bool],
    k: int,
    *,
    golden_total: int | None = None,
    relevance_ids: Sequence[str] | None = None,
) -> float:
    """Recall@K：前 K 个结果覆盖了多少比例的相关项。

    分母必须来自 golden 标注全集（golden_total），而非检索列表里的相关项计数。
    历史调用只传 ranked_relevant 时兼容：退化为用检索列表里的 True 数——但这在
    评测集有"检索永远漏召回"的场景下会虚高。清单 E01 明确要求分母来自全集。

    relevance_ids 给出每个排名项的标注标识（如文档 title/doc_id）时必须去重：
    一篇文档会被切成多个 chunk，同一标注项在前 K 里出现多次只能算命中一次。
    真实评测中同一文档 9 个 chunk 进前 5 时，chunk 级计数会得到 recall@5=1.145>1，
    这不是合法指标值。
    """
    total = golden_total if golden_total is not None else sum(
        1 for flag in ranked_relevant if flag
    )
    if total <= 0:
        return 0.0

    if relevance_ids is not None:
        hit_ids: set[str] = set()
        for flag, rel_id in zip(ranked_relevant[:k], relevance_ids[:k], strict=False):
            if flag:
                hit_ids.add(rel_id)
        return min(len(hit_ids) / total, 1.0)

    return sum(1 for flag in ranked_relevant[:k] if flag) / total


def reciprocal_rank(ranked_relevant: Sequence[bool]) -> float:
    """MRR：第一个相关结果排名的倒数（无相关结果时为 0）。"""
    for rank, flag in enumerate(ranked_relevant, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_relevant: Sequence[bool],
    k: int,
    *,
    golden_total: int | None = None,
    graded_relevance: Sequence[float] | None = None,
    golden_grades: Sequence[float] | None = None,
    relevance_ids: Sequence[str] | None = None,
) -> float:
    """nDCG@K：DCG / IDCG，衡量相关结果是否排在更靠前的位置。

    IDCG 必须基于 golden 全集的理想排序：
    - golden_grades（golden 全集的等级相关性，如 [3, 2, 1]）：IDCG 直接用它降序取前 k，
      是等级相关性下唯一不依赖召回结果的严格口径（清单项29 的相关性等级标注）；
    - 二元相关性 + golden_total：理想排序 = [1.0] * golden_total + [0.0] * 剩余
      （与检索是否返回了这些项无关），否则当检索漏召回严重时 nDCG 会虚高；
    - graded_relevance（检索列表逐项等级）+ golden_total：理想排序用检索列表等级降序
      （妥协口径：漏召回时 IDCG 偏小、nDCG 偏大，仅在无 golden_grades 时使用）；
    - 都没传时兼容旧行为：IDCG 基于检索列表本身（评测口径不严格，仅做相对对比）。
    """
    if graded_relevance is not None:
        gains = list(graded_relevance[:k])
    else:
        gains = [1.0 if flag else 0.0 for flag in ranked_relevant[:k]]

    if relevance_ids is not None:
        # 同一标注项的多个 chunk 只给首次出现计收益，否则 DCG 会超过 IDCG
        seen: set[str] = set()
        deduped: list[float] = []
        for gain, rel_id in zip(gains, relevance_ids[:k], strict=False):
            if gain > 0 and rel_id in seen:
                deduped.append(0.0)
            else:
                deduped.append(gain)
                if gain > 0:
                    seen.add(rel_id)
        gains = deduped

    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))

    if golden_grades is not None:
        # golden 全集等级降序取前 k：理想排序与召回结果完全无关，口径最严
        ideal_all = sorted((float(g) for g in golden_grades), reverse=True)[:k]
        idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_all, start=1))
    elif graded_relevance is not None and golden_total is not None:
        # 等级相关性 + 全集总数：理想排序 = golden 全集按等级降序取前 k
        all_golden = sorted(graded_relevance, reverse=True)[:k]
        idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(all_golden, start=1))
    elif golden_total is not None:
        # 二元 + 全集总数：理想排序 = golden_total 个 1.0 接剩余 0.0
        ideal_all = [1.0] * min(golden_total, k) + [0.0] * max(0, k - golden_total)
        idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_all, start=1))
    else:
        # 仅用检索列表本身（兼容旧接口，评测口径不严格）
        ideal_all = sorted(gains, reverse=True)
        idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_all, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def _covered_by(chunks: Sequence[RetrievedChunk], evidence: str) -> tuple[bool, bool]:
    """返回（是否完整覆盖, 是否发生边界截断）。

    完整覆盖：某个片段正文包含整段证据；
    边界截断：没有完整覆盖，但证据的首尾各有一半出现在片段里——说明切块把证据切断了。
    """
    probe_head = evidence[: max(10, len(evidence) // 3)]
    probe_tail = evidence[-max(10, len(evidence) // 3):]
    truncated = False
    for chunk in chunks:
        if evidence in chunk.content:
            return True, False
        if probe_head in chunk.content or probe_tail in chunk.content:
            truncated = True
    return False, truncated


def evidence_metrics(
    chunks: Sequence[RetrievedChunk], expected_evidence: Sequence[str]
) -> dict[str, float]:
    """证据四项：边界截断率、证据完整率、相邻块依赖率、引用定位成功率。

    - 边界截断率：证据被切块切断的占比（越高说明 chunk 边界设计不合理）；
    - 证据完整率：整段证据落在同一个片段内的占比；
    - 相邻块依赖率：返回片段中"与同文档另一片段相邻/重叠"的比例
      （过高说明单块不足以表达完整语义，需要父子切片或更大 overlap）；
    - 引用定位成功率：带 section_path 或 char_start 的片段比例
      （决定答案引用能否回指到原文小节）。
    """
    if not expected_evidence:
        return {
            "evidence_completeness": 0.0,
            "boundary_truncation_rate": 0.0,
            "adjacent_dependency_rate": _adjacent_dependency_rate(chunks),
            "citation_locatable_rate": _citation_locatable_rate(chunks),
        }

    complete = 0
    truncated = 0
    for evidence in expected_evidence:
        is_complete, is_truncated = _covered_by(chunks, evidence)
        complete += int(is_complete)
        truncated += int(is_truncated and not is_complete)

    total = len(expected_evidence)
    return {
        "evidence_completeness": complete / total,
        "boundary_truncation_rate": truncated / total,
        "adjacent_dependency_rate": _adjacent_dependency_rate(chunks),
        "citation_locatable_rate": _citation_locatable_rate(chunks),
    }


def _adjacent_dependency_rate(chunks: Sequence[RetrievedChunk]) -> float:
    """相邻块依赖率：按 doc_id 分组，统计与另一块区间相邻/重叠的片段占比。"""
    if not chunks:
        return 0.0
    ranges: dict[str, list[tuple[int, int]]] = {}
    for chunk in chunks:
        start = chunk.metadata.get("char_start")
        end = chunk.metadata.get("char_end")
        if isinstance(start, int) and isinstance(end, int):
            ranges.setdefault(chunk.doc_id, []).append((start, end))

    dependent = 0
    for spans in ranges.values():
        spans.sort()
        adjacent: set[int] = set()
        for index in range(len(spans) - 1):
            current_end = spans[index][1]
            # 下一块起点 ≤ 当前块终点：两块的证据边界相接或重叠，单独一块读不通
            if spans[index + 1][0] <= current_end:
                adjacent.add(index)
                adjacent.add(index + 1)
        dependent += len(adjacent)
    return dependent / len(chunks)


def _citation_locatable_rate(chunks: Sequence[RetrievedChunk]) -> float:
    """引用定位成功率：片段是否带结构路径或原文偏移，可用于回指。"""
    if not chunks:
        return 0.0
    locatable = sum(
        1
        for chunk in chunks
        if chunk.metadata.get("section_path") or chunk.metadata.get("char_start") is not None
    )
    return locatable / len(chunks)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    """轻量分词：英文/数字按词、中文按单字，用于冗余度估算（不引入分词依赖）。"""
    return _TOKEN_RE.findall(text.lower())


def redundancy_metrics(chunks: Sequence[RetrievedChunk]) -> dict[str, float]:
    """冗余三项：重复 Chunk 比例、同文档候选占比、重复 Token 比例。"""
    if not chunks:
        return {
            "duplicate_chunk_ratio": 0.0,
            "same_doc_candidate_ratio": 0.0,
            "duplicate_token_ratio": 0.0,
        }

    fingerprints = {str(chunk.metadata.get("content_hash") or chunk.content) for chunk in chunks}
    doc_counts: dict[str, int] = {}
    for chunk in chunks:
        doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1

    tokens = [token for chunk in chunks for token in _tokenize(chunk.content)]
    return {
        "duplicate_chunk_ratio": 1.0 - len(fingerprints) / len(chunks),
        "same_doc_candidate_ratio": max(doc_counts.values()) / len(chunks),
        "duplicate_token_ratio": 1.0 - len(set(tokens)) / len(tokens) if tokens else 0.0,
    }


def estimate_tokens(text: str) -> int:
    """Token 估算：中文约 1.5 字符/token，英文按 4 字符/token 混算，取保守值。"""
    return math.ceil(len(text) / 1.5) if text else 0


def p95(values: Sequence[float]) -> float:
    """P95：不足 20 个样本时直接取最大值，避免小样本分位数失真。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) < 20:
        return ordered[-1]
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def cost_metrics(
    chunks: Sequence[RetrievedChunk],
    *,
    vector_dim: int,
    latencies_ms: Sequence[float] = (),
) -> dict[str, float]:
    """成本五项：返回 Chunk 数、向量存储估算、上下文 Token、检索 P95（毫秒）。

    vector_storage_bytes 按 float32 × 维度估算（单条向量本身的字节数），
    不含主键/元数据开销；作为跨配置对比的相对指标足够。
    """
    return {
        "chunk_count": float(len(chunks)),
        "vector_storage_bytes_per_chunk": float(vector_dim * 4),
        "context_token_estimate": float(
            sum(estimate_tokens(str(c.metadata.get("parent_content") or c.content)) for c in chunks)
        ),
        "retrieval_p95_ms": p95(latencies_ms),
    }


def refusal_quality(*, predicted_refusal: bool, expected_no_answer: bool) -> str:
    """拒答质量四分类：
    correct_answer（该答且答）、correct_refusal（该拒且拒）、
    false_refusal（有答案却拒答）、hallucination（无答案却硬答）。
    """
    if expected_no_answer:
        return "correct_refusal" if predicted_refusal else "hallucination"
    return "false_refusal" if predicted_refusal else "correct_answer"


def summarize_refusal(labels: Sequence[str]) -> dict[str, int]:
    """汇总拒答四分类计数，便于脚本输出与阈值告警。"""
    summary = {"correct_answer": 0, "correct_refusal": 0, "false_refusal": 0, "hallucination": 0}
    for label in labels:
        summary[label] = summary.get(label, 0) + 1
    return summary
