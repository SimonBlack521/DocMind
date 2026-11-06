"""图增强重排（Graph-enhanced Reranking，默认关闭）。

定位说明：本模块不是完整 GraphRAG（没有实体/关系抽取、持久化知识图谱与图检索
通道），而是在本次候选集上做"共享词元图 + 中心度"的轻量重排增强，因此统一
使用 Graph-enhanced Reranking 这个名称，避免技术定位被误解。

做什么：
    在候选片段之间建"共享词元"图：两个片段共享足够多词元就连一条边。
    综合"与问题的相关性排名"和"在候选图中的中心度排名"，用 RRF 式倒数排名
    融合重新排序。与问题高度相关、且被其他候选共同提及的片段更可能承载
    该主题的完整上下文，从而排到前面。

不做什么（与完整 GraphRAG 的差异，也是命名收敛的原因）：
    - 不做离线实体抽取与全局图谱构建，只在本次候选集上构图；
    - 不更新 chunk.score，避免打乱"重排分数阈值"的语义，只调整顺序；
    - 不做新增候选召回，因此没有额外存储/检索开销。

适用场景：multi-hop 问题里，答案分散在多个片段、且这些片段共享实体词时，
图中心度能把"相互印证"的片段提到前面。默认关闭，按知识库配置启用。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from app.domain.dto import RetrievedChunk

# 中文高频虚词/疑问词不参与构图，否则几乎所有片段都会连在一起
_STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "及",
    "对",
    "为",
    "有",
    "这",
    "那",
    "什么",
    "怎么",
    "如何",
    "哪些",
    "请问",
    "一下",
    "我们",
    "你们",
    "可以",
    "是否",
    "以及",
    "进行",
    "关于",
    "一个",
    "这个",
    "那个",
}
_MAX_TERMS = 60


def graph_enhanced_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    k: int = 60,
    boost: float = 0.5,
) -> list[RetrievedChunk]:
    """按"相关性 + 图中心度"重排候选；返回新列表，不修改 chunk 对象。"""
    if len(chunks) <= 1:
        return list(chunks)

    query_terms = _terms(query)
    chunk_terms = [_terms(chunk.content) for chunk in chunks]
    document_frequency = Counter(term for terms in chunk_terms for term in terms)
    total = len(chunks)

    # 1) 相关性：问题词元的 IDF 加权命中（区分度高的词命中更有价值）
    relevance = [
        sum(
            math.log((total + 1) / (document_frequency[term] + 1)) + 1.0
            for term in query_terms & terms
        )
        for terms in chunk_terms
    ]
    # 2) 图中心度：与多少个其他片段共享 >=2 个词元
    centrality = [0] * total
    for left in range(total):
        for right in range(left + 1, total):
            if len(chunk_terms[left] & chunk_terms[right]) >= 2:
                centrality[left] += 1
                centrality[right] += 1

    relevance_rank = _rank(relevance)
    centrality_rank = _rank(centrality)
    scores = [
        1.0 / (k + relevance_rank[index]) + boost / (k + centrality_rank[index])
        for index in range(total)
    ]
    # 稳定排序：分数相同保持原始顺序（重排前的相关性顺序）
    order = sorted(range(total), key=lambda index: scores[index], reverse=True)
    return [chunks[index] for index in order]


def _rank(values: Sequence[float]) -> list[int]:
    """把数值降序转成 1-based 排名（同分按原顺序，保持确定性）。"""
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _terms(text: str) -> set[str]:
    """轻量词元：英文/数字按词切分，中文按连续字串 + 2 字片段，过滤停用词。"""
    terms: set[str] = set()
    for word in _split_words(text.lower()):
        if word in _STOPWORDS:
            continue
        if _is_cjk(word):
            if len(word) >= 2:
                terms.add(word)
            terms.update(word[index : index + 2] for index in range(len(word) - 1))
        elif len(word) >= 2:
            terms.add(word)
        if len(terms) >= _MAX_TERMS:
            break
    return terms


def _split_words(text: str) -> list[str]:
    """连续字母数字或连续汉字视为一个词；其余字符作为分隔符。"""
    words: list[str] = []
    buffer = ""
    for char in text:
        if char.isalnum():
            buffer += char
        elif buffer:
            words.append(buffer)
            buffer = ""
    if buffer:
        words.append(buffer)
    return words


def _is_cjk(word: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in word)
