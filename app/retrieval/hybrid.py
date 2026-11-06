"""混合检索：向量召回 + BM25 召回并行执行，按 RRF 融合，可选精排。

隔离策略（三层兜底）：
- 每个知识库独立 Milvus collection（rag_{kb_code}）与独立 ES 索引；
- 向量检索再叠加 tenant_id + kb_id 过滤表达式，ES 检索再叠加 kb_id 过滤；
- 用户组受限时（allowed_groups 非 None）两路召回都叠加文档级 ACL 过滤（检索时过滤，
  不是召回后过滤），公开文档靠哨兵 "*" 命中。
"""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from app.core.errors import AppError, ModelUnavailableError
from app.core.logging import get_logger
from app.core.metrics import metrics
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import RetrievalConfig
from app.providers.embedding import EmbeddingProvider
from app.providers.reranker import Reranker
from app.retrieval.adaptive import (
    adjust_top_k_by_type,
    estimate_query_complexity,
    resolve_dynamic_top_k,
    truncate_at_score_cliff,
)
from app.retrieval.diversity import dedupe_chunks, mmr_select
from app.retrieval.embedding_cache import QueryEmbeddingCache
from app.retrieval.graph import graph_enhanced_rerank
from app.retrieval.query_classifier import QueryClassification, QueryClassifier
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.ranking import apply_governance_weighting
from app.retrieval.rrf import reciprocal_rank_fusion
from app.storage.keyword_base import KeywordStore
from app.storage.vector_base import VectorStore, scope_expression

# 时效过滤窗口（秒）：给时间比较一点抖动余量，避免同一秒内的时钟偏移把合法文档踢掉
_TEMPORAL_TOLERANCE_SECONDS = 5

logger = get_logger(__name__)


_SENTENCE_END = ("。", "！", "？", "；", ".", "!", "?", ";", "\n")


def _truncate_text(text: str, limit: int) -> str:
    """按字符预算截断父块内容：优先在句末截断，避免把一句话拦腰截断。

    limit<=0 表示不限制；截断后追加省略号，提示 LLM 上下文被压缩过。
    """
    if limit <= 0 or len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(mark) for mark in _SENTENCE_END)
    if cut >= limit // 3:
        window = window[: cut + 1]
    return f"{window.rstrip()}…"


def _chunk_version(chunk: RetrievedChunk) -> int:
    """chunk metadata 里的文档版本；历史数据缺该字段按 0 处理。"""
    try:
        return int(chunk.metadata.get("version") or 0)
    except (TypeError, ValueError):
        return 0


# ISO 格式容错：从 metadata 拿出来的 valid_from/valid_to 可能是字符串


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_temporally_active(chunk: RetrievedChunk, *, now: datetime | None = None) -> bool:
    """时间有效性过滤：valid_from <= now <= valid_to（None 表示无约束）。"""
    now = now or datetime.now(UTC)
    tolerance = _TEMPORAL_TOLERANCE_SECONDS
    valid_from = _parse_datetime(chunk.metadata.get("valid_from"))
    valid_to = _parse_datetime(chunk.metadata.get("valid_to"))
    if valid_from is not None and (now - valid_from).total_seconds() < -tolerance:
        return False
    return not (
        valid_to is not None
        and (now - valid_to).total_seconds() > tolerance
    )


def _dedupe_to_latest_version(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """同一 doc_id 只保留最高版本的 chunk（保序）。

    增量重建采用"先写新版、后清旧版"，窗口期新旧版本在索引里并存；
    这里保证检索永远只读最新版本。缺 version 的历史数据按 0 处理：
    同一文档全是旧数据时原样保留，新旧并存时旧数据被过滤。
    """
    latest: dict[str, int] = {}
    for chunk in chunks:
        version = _chunk_version(chunk)
        if version > latest.get(chunk.doc_id, -1):
            latest[chunk.doc_id] = version
    return [chunk for chunk in chunks if _chunk_version(chunk) >= latest[chunk.doc_id]]


class HybridRetriever:
    """一个实例绑定一个知识库的检索配置与存储连接。"""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embedding: EmbeddingProvider,
        reranker: Reranker,
        rewriter: QueryRewriter,
        tenant_id: str,
        kb_id: str,
        config: RetrievalConfig,
        authority_weight: float = 0.2,
        recency_weight: float = 0.1,
        recency_half_life_days: float = 180.0,
        embedding_cache: QueryEmbeddingCache | None = None,
        embedding_version: str = "",
        rerank_max_candidates: int = 0,
        rerank_skip_confidence: float = 0.0,
        executor: ThreadPoolExecutor | None = None,
        classifier: QueryClassifier | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._keyword_store = keyword_store
        self._embedding = embedding
        self._reranker = reranker
        self._rewriter = rewriter
        self._config = config
        self._tenant_id = tenant_id
        self._kb_id = kb_id
        self._authority_weight = authority_weight
        self._recency_weight = recency_weight
        self._recency_half_life_days = recency_half_life_days
        self._embedding_cache = embedding_cache
        self._embedding_version = embedding_version
        self._rerank_max_candidates = rerank_max_candidates
        self._rerank_skip_confidence = rerank_skip_confidence
        self._classifier = classifier
        # 共享线程池（v1.4-5）：外部传入则复用，否则自建 max_workers=8
        # 所有 variants 的并行、vector/keyword 双路并行都走这一个池，
        # 避免每个请求单独建池带来的线程爆炸与资源争抢
        self._executor = executor or ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="rag-retrieve"
        )

    @property
    def embedding_dim(self) -> int:
        """向量维度：评测脚本用它估算单条 chunk 的向量存储开销。"""
        return self._embedding.dim

    def _recall_one_variant(
        self, variant: str, vector: list[float], top_k: int, scope: str,
        allowed_groups: frozenset[str] | None, excluded_doc_ids: Collection[str] | None,
        min_keep: int,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[AppError], float]:
        """单个查询变体的双路召回：返回向量路与 BM25 路的原始结果（不做变体内融合）。

        v1.5 起融合改为单层 Multi-Channel RRF（清单第三优先级第 12 项）：
        每个变体的每一路都作为独立 channel 进统一 RRF，保留各变体各路的
        独立排名信息，避免"先合并去重再融合"丢失变体间的排名差异。
        向量预先批量算好（v1.4-4 Embedding 批处理），这里直接用避免多次小请求。
        """
        keyword_hits, k_errs = self._keyword_channel(
            [variant], top_k, allowed_groups, excluded_doc_ids, min_keep
        )
        errors = list(k_errs)
        # 用缓存好的向量直接搜，不走 embed_query（embed_documents 批量更快）
        raw_vector_hits = self._vector_store.search(vector, top_k=top_k, expr=scope)
        vector_hits = [
            RetrievedChunk(
                chunk_id=raw.chunk_id,
                content=raw.content,
                score=raw.score,
                doc_id=raw.doc_id,
                source_uri=raw.source_uri,
                title=str(raw.metadata.get("title", "")),
                metadata=dict(raw.metadata),
            )
            for raw in raw_vector_hits
        ]
        # 单路分数断层早停（同一路内分数尺度可比）
        vector_hits = truncate_at_score_cliff(
            vector_hits, min_keep=min_keep, ratio=self._config.score_cliff_ratio
        )
        top_vector_score = max((float(h.score) for h in vector_hits), default=0.0)
        return vector_hits, keyword_hits, errors, top_vector_score

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        use_rewrite: bool | None = None,
        use_rerank: bool | None = None,
        rerank_top_n: int | None = None,
        allowed_groups: frozenset[str] | None = None,
        excluded_doc_ids: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        """混合检索主流程：查询改写（可选）→ 多路变体并行召回 → 双层 RRF 融合 → 精排（可选）。

        Multi-Query RRF（v1.3-1）：
        - 每个查询变体（原问题 + 改写变种）独立跑完向量 + BM25 两路，各自做本地 RRF；
        - 跨变体再做一次 RRF 融合，原问题变体权重 2x（避免改写变种淹没用户原始意图）；
        - variants 并行处理，所有并行都走共享有界线程池，避免线程爆炸。

        兜底顺序（任何一环降级都不影响整体出结果）：
        1. 查询改写失败 → 回退原查询（见 QueryRewriter）；
        2. 单变体单路失败 → 记日志跳过，同变体另一路照常出结果；
        3. 全部变体全部挂且零候选 → 抛出第一个错误（API 层映射成 502/503）；
        4. 精排服务不可用 → 回退 RRF 融合顺序。
        """
        query = question.strip()
        if not query:
            return []

        # 生效参数解析
        classification: QueryClassification | None = None
        if self._classifier is not None:
            classification = self._classifier.classify(query)
        if top_k is None:
            top_k = (
                resolve_dynamic_top_k(
                    query,
                    base_k=self._config.top_k,
                    min_k=self._config.top_k_min,
                    max_k=self._config.top_k_max,
                )
                if self._config.adaptive_top_k
                else self._config.top_k
            )
            # 查询类型修正（清单第三优先级第 14 项）：事实查询收紧候选池、
            # 多跳/对比/流程类放大；仅在自适应开启时应用，夹在 min/max 内
            if self._config.adaptive_top_k and classification is not None:
                top_k = adjust_top_k_by_type(
                    top_k,
                    query_type=classification.query_type,
                    base_k=self._config.top_k,
                    min_k=self._config.top_k_min,
                    max_k=self._config.top_k_max,
                )
        use_rewrite = self._config.use_query_rewrite if use_rewrite is None else use_rewrite
        use_rerank = self._config.use_rerank if use_rerank is None else use_rerank
        final_n = rerank_top_n or self._config.rerank_top_n
        min_keep = max(1, min(top_k, final_n))

        # 改写产出多条查询变体（第一条永远是原问题）
        variants = self._rewriter.rewrite(query) if use_rewrite else [query]

        # v1.3-3 Query Decomposition：多跳/流程/对比问题拆子问题，追加进 variants
        # 失败直接跳过（空列表），不影响 Multi-Channel RRF 主链路；
        # classification 复用上面已算好的分类结果，避免重复调用 LLM
        if self._classifier is not None:
            for sub in self._classifier.decompose(query, classification=classification):
                if sub not in variants:
                    variants.append(sub)
        scope = scope_expression(
            self._tenant_id, self._kb_id, allowed_groups, excluded_doc_ids
        )

        # v1.4-4 Embedding 批处理：所有 variants 一次性 embed_documents，
        # 比在线程池里分别 embed_query（多次小请求）快得多——GPU/HTTP 都受益
        cached: dict[str, list[float]] = {}
        uncached = [v for v in variants if v not in cached]
        if uncached:
            # 先查进程内 Query Embedding 缓存，再批量算剩余的
            if self._embedding_cache is not None and self._embedding_cache.enabled:
                for v in uncached:
                    hit = self._embedding_cache.get(
                        tenant_id=self._tenant_id,
                        kb_id=self._kb_id,
                        model=self._embedding.model,
                        embedding_version=self._embedding_version,
                        text=v,
                    )
                    if hit is not None:
                        cached[v] = hit
                uncached = [v for v in uncached if v not in cached]
            if uncached:
                vectors = self._embedding.embed_documents(uncached)
                for variant, vector in zip(uncached, vectors, strict=True):
                    cached[variant] = vector
                # 写回缓存
                if self._embedding_cache is not None and self._embedding_cache.enabled:
                    for variant, vector in zip(uncached, vectors, strict=True):
                        self._embedding_cache.set(
                            tenant_id=self._tenant_id,
                            kb_id=self._kb_id,
                            model=self._embedding.model,
                            embedding_version=self._embedding_version,
                            text=variant,
                            vector=vector,
                        )

        # Multi-Channel RRF（v1.5，清单第 12 项）：每个变体的向量/BM25 两路
        # 作为独立 channel 进统一 RRF，原问题变体（index 0）权重 2x，
        # 保留各变体各路的独立排名信息，防止改写变体淹没用户原始意图
        futures = [
            self._executor.submit(
                self._recall_one_variant,
                variant, cached[variant], top_k, scope, allowed_groups, excluded_doc_ids, min_keep,
            )
            for variant in variants
        ]
        per_variant = [future.result() for future in futures]
        all_variant_errors: list[AppError] = []
        for _, _, errors, _ in per_variant:
            all_variant_errors.extend(errors)

        channels: list[list[RetrievedChunk]] = []
        channel_weights: list[float] = []
        for index, (vector_hits, keyword_hits, _, _) in enumerate(per_variant):
            variant_weight = 2.0 if index == 0 else 1.0
            if vector_hits:
                channels.append(vector_hits)
                channel_weights.append(variant_weight * self._config.vector_weight)
            if keyword_hits:
                channels.append(keyword_hits)
                channel_weights.append(variant_weight * self._config.bm25_weight)
        # 原问题变体的向量首名相似度——用于"是否值得精排"的置信度判断
        original_vector_top = per_variant[0][3] if per_variant else 0.0
        if not channels and all_variant_errors:
            raise all_variant_errors[0]
        if not channels:
            return []

        fused = reciprocal_rank_fusion(
            channels, weights=channel_weights, k=self._config.rrf_k
        )
        metrics.observe(
            "multi_query_variants_count",
            float(len(variants)),
            kb=self._kb_id,
        )

        # 增量重建窗口期同一文档的新旧版本 chunk 会并存：按 doc 分组只保留最高版本
        fused = _dedupe_to_latest_version(fused)
        fused = dedupe_chunks(fused)
        # 时间有效性过滤：valid_from/valid_to 过期的文档不应参与回答
        before = len(fused)
        fused = [chunk for chunk in fused if _is_temporally_active(chunk)]
        if len(fused) < before:
            metrics.inc("temporal_filtered_total", kb=self._kb_id)
            logger.info(
                "时效过滤剔除 %d 条过期/未生效候选 kb=%s", before - len(fused), self._kb_id
            )
        # 治理加权：按来源权威度与采集时间重排
        fused = apply_governance_weighting(
            fused,
            authority_weight=self._authority_weight,
            recency_weight=self._recency_weight,
            half_life_days=self._recency_half_life_days,
        )
        # 候选数取 top_k 与精排输出数的较大者；同时受全局候选上限约束
        pool_size = max(top_k, final_n)
        if self._rerank_max_candidates > 0:
            pool_size = min(pool_size, max(final_n, self._rerank_max_candidates))
        candidates = fused[:pool_size]

        query_vec = cached.get(query)
        if use_rerank and candidates and not self._skip_rerank(
            query, original_vector_top, fused
        ):
            started = time.perf_counter()
            try:
                reranked = self._reranker.rerank(query, candidates, top_n=final_n)
                return self._expand_parents(
                    self._select_final(query, reranked, final_n, query_vector=query_vec)
                )
            except (ModelUnavailableError, ValueError, RuntimeError) as exc:
                # 精排是可选增强：服务不可用、返回结构异常、本地推理 OOM/超时
                # 都降级为 RRF 排序，不能因为精排故障让整条问答请求失败
                message = getattr(exc, "message", str(exc))
                logger.warning("精排不可用，回退 RRF 排序：%s", message)
            finally:
                metrics.observe(
                    "rerank_duration_seconds", time.perf_counter() - started, kb=self._kb_id
                )

        return self._expand_parents(
            self._select_final(query, fused, final_n, query_vector=query_vec)
        )

    def _select_final(
        self,
        question: str,
        ranked: Sequence[RetrievedChunk],
        final_n: int,
        *,
        query_vector: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """出口选择：按配置做 MMR 多样性（λ=1 时退化为直接截断）。

        v1.3-4 向量升级：
        调用方可传 query_vector 以启用余弦相似度的多样性惩罚（语义级去重）；
        不传则保持词面 Jaccard 兜底。候选 chunk 的向量优先从向量库按主键回取
        （入库时已算过，直接复用），只有索引里缺失的才重新 Embedding。
        """
        if self._config.mmr_lambda < 1.0 and len(ranked) > 1:
            # 先复用已入库向量，避免对候选片段重复调用 Embedding
            chunk_vectors = self._reuse_chunk_vectors(ranked)
            if chunk_vectors is None:
                # 向量不可用时由 mmr_select 回退到词面 Jaccard，不影响出结果
                logger.info("MMR 候选向量不可用，回退词面多样性 kb=%s", self._kb_id)
            selected = mmr_select(
                question,
                ranked,
                top_n=final_n,
                lambda_=self._config.mmr_lambda,
                chunk_vectors=chunk_vectors,
                query_vector=query_vector,
            )
            return self._graph_rerank(question, selected)
        return self._graph_rerank(question, list(ranked[:final_n]))

    def _reuse_chunk_vectors(
        self, chunks: Sequence[RetrievedChunk]
    ) -> list[list[float]] | None:
        """按 chunk_id 从向量库回取入库向量；缺失部分才重新 Embedding。

        返回与 chunks 顺序一一对应的向量列表；拿不到完整向量时返回 None，
        让 MMR 走词面 Jaccard 兜底（多样性是增强，不应让问答失败）。
        """
        ids = [chunk.chunk_id for chunk in chunks]
        try:
            stored = self._vector_store.get_vectors_by_ids(ids)
        except AppError as exc:
            logger.warning("MMR 回取入库向量失败：%s", exc.message)
            stored = {}
        missing = [chunk for chunk in chunks if chunk.chunk_id not in stored]
        computed: dict[str, list[float]] = {}
        if missing:
            try:
                vectors = self._embedding.embed_documents(
                    [chunk.content for chunk in missing]
                )
            except (AppError, ValueError) as exc:
                logger.warning(
                    "MMR 补齐缺失向量失败：%s", getattr(exc, "message", str(exc))
                )
                return None
            computed = {
                chunk.chunk_id: vector
                for chunk, vector in zip(missing, vectors, strict=True)
            }
        vectors_by_id = {**stored, **computed}
        ordered = [vectors_by_id.get(chunk.chunk_id) for chunk in chunks]
        if any(vector is None for vector in ordered):
            return None
        return [vector for vector in ordered if vector is not None]

    def _graph_rerank(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """可选图增强重排（Graph-enhanced Reranking）：默认关闭，不改变 chunk.score。"""
        if not self._config.graph_rerank_enabled or len(chunks) <= 1:
            return chunks
        try:
            return graph_enhanced_rerank(
                question, chunks, boost=self._config.graph_rerank_boost
            )
        except ValueError as exc:
            logger.warning("图增强重排失败，保持原顺序：%s", exc)
            return chunks

    def _skip_rerank(
        self,
        question: str,
        original_vector_top: float,
        fused: Sequence[RetrievedChunk],
    ) -> bool:
        """按需精排：简单问题且向量首名高置信时跳过。

        置信度判断用原始向量首名相似度（余弦，量纲正确），
        断层判断用融合后的 top 候选分布（看是否有明显名次断层值得精排重排）。
        """
        if self._rerank_skip_confidence <= 0 or not fused:
            return False
        if estimate_query_complexity(question) >= 0.35:
            return False
        if original_vector_top < self._rerank_skip_confidence:
            return False
        # 融合后前 3 名的 RRF 分数是否有断层：如果有，说明排序可能还能改善
        flat_heights = [float(c.score) for c in list(fused)[:3]]
        if len(flat_heights) >= 2:
            top = max(flat_heights)
            cliff = max(flat_heights[i] - flat_heights[i + 1] for i in range(len(flat_heights) - 1))
            if top > 0 and cliff > 0.1 * top:
                return False
        metrics.inc("rerank_skipped_total", kb=self._kb_id)
        logger.info(
            "高置信简单问题跳过精排 kb=%s vector_top=%.4f",
            self._kb_id,
            original_vector_top,
        )
        return True

    def _expand_parents(self, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """父子切片回取：命中的是子块，送生成的是它所属小节的整节内容。

        实现要点：
        - 只发一次批量回取（ES mget），父块正文不会随每个子块重复存储；
        - 同一父块被多个子块命中时只保留排名更靠前的子块，避免上下文重复；
        - 回取失败只降级为"用子块内容"并记日志，不影响本次问答出结果。
        """
        parent_ids = [
            str(chunk.metadata["parent_chunk_id"])
            for chunk in chunks
            if chunk.metadata.get("parent_chunk_id")
        ]
        if not parent_ids:
            return list(chunks)

        try:
            parents = self._keyword_store.get_by_ids(parent_ids)
        except AppError as exc:
            logger.warning("父块回取失败，降级使用子块内容：%s", exc.message)
            return list(chunks)

        parent_contents = {hit.chunk_id: hit.content for hit in parents}
        expanded: list[RetrievedChunk] = []
        seen_parents: set[str] = set()
        used_chars = 0
        for chunk in chunks:
            parent_id = str(chunk.metadata.get("parent_chunk_id") or "")
            if parent_id:
                if parent_id in seen_parents:
                    # 同一小节的多个子块命中：保留排名更靠前的一个，避免重复上下文
                    continue
                seen_parents.add(parent_id)
                parent_content = parent_contents.get(parent_id)
                if parent_content:
                    # 单个父块先按上限截断（句末优先），再受全局上下文预算约束；
                    # 超出总预算的后续父块直接退回子块内容，不把整节都塞进提示词
                    truncated = _truncate_text(
                        parent_content, self._config.parent_max_chars
                    )
                    if (
                        self._config.parent_context_max_chars <= 0
                        or used_chars + len(truncated)
                        <= self._config.parent_context_max_chars
                    ):
                        chunk.metadata["parent_content"] = truncated
                        used_chars += len(truncated)
            expanded.append(chunk)
        return expanded

    def _vector_channel(
        self, variants: Sequence[str], top_k: int, scope: str, min_keep: int
    ) -> tuple[list[RetrievedChunk], list[AppError]]:
        """向量召回：每个查询变体单独向量化并检索；单个变体失败只记录，不影响其他变体。

        scope 表达式把 tenant/kb/ACL 过滤下推到 Milvus 内部执行（检索时过滤），
        保证 top_k 名额只被有权访问的文档占用，而不是召回后再筛掉越权文档。
        每个变体返回后先做分数断层早停，再进入融合。
        """
        hits: list[RetrievedChunk] = []
        errors: list[AppError] = []
        for variant in variants:
            try:
                vector = self._embed_query(variant)
                batch = [
                    RetrievedChunk(
                        chunk_id=raw.chunk_id,
                        content=raw.content,
                        score=raw.score,
                        doc_id=raw.doc_id,
                        source_uri=raw.source_uri,
                        title=str(raw.metadata.get("title", "")),
                        metadata=dict(raw.metadata),
                    )
                    for raw in self._vector_store.search(vector, top_k=top_k, expr=scope)
                ]
                hits.extend(
                    truncate_at_score_cliff(
                        batch, min_keep=min_keep, ratio=self._config.score_cliff_ratio
                    )
                )
            except AppError as exc:
                logger.warning("向量检索失败，跳过该查询：%s", exc.message)
                errors.append(exc)
        return _dedupe_by_chunk(hits), errors

    def _embed_query(self, text: str) -> list[float]:
        """带 Query Embedding 缓存地向量化：同问题 TTL 内不重复调用模型。"""
        cache = self._embedding_cache
        if cache is None or not cache.enabled:
            return self._embedding.embed_query(text)
        hit = cache.get(
            tenant_id=self._tenant_id,
            kb_id=self._kb_id,
            model=self._embedding.model,
            embedding_version=self._embedding_version,
            text=text,
        )
        if hit is not None:
            metrics.inc("query_embedding_cache_hits_total", kb=self._kb_id)
            return hit
        vector = self._embedding.embed_query(text)
        cache.set(
            tenant_id=self._tenant_id,
            kb_id=self._kb_id,
            model=self._embedding.model,
            embedding_version=self._embedding_version,
            text=text,
            vector=vector,
        )
        return vector

    def _keyword_channel(
        self,
        variants: Sequence[str],
        top_k: int,
        allowed_groups: frozenset[str] | None,
        excluded_doc_ids: Collection[str] | None = None,
        min_keep: int = 1,
    ) -> tuple[list[RetrievedChunk], list[AppError]]:
        """BM25 召回：查询变体原样进 ES（全文检索不需要向量化）。

        与向量路一致，kb_id / ACL 过滤在 ES 查询里执行（检索时过滤）；
        单个变体失败只记录，不影响其他变体；每个变体结果先做分数断层早停。
        """
        hits: list[RetrievedChunk] = []
        errors: list[AppError] = []
        for variant in variants:
            try:
                batch = [
                    RetrievedChunk(
                        chunk_id=raw.chunk_id,
                        content=raw.content,
                        score=raw.score,
                        doc_id=raw.doc_id,
                        source_uri=raw.source_uri,
                        title=str(raw.metadata.get("title", "")),
                        metadata=dict(raw.metadata),
                    )
                    for raw in self._keyword_store.search(
                        variant,
                        top_k=top_k,
                        kb_id=self._kb_id,
                        allowed_groups=allowed_groups,
                        excluded_doc_ids=excluded_doc_ids,
                    )
                ]
                hits.extend(
                    truncate_at_score_cliff(
                        batch, min_keep=min_keep, ratio=self._config.score_cliff_ratio
                    )
                )
            except AppError as exc:
                logger.warning("BM25 检索失败，跳过该查询：%s", exc.message)
                errors.append(exc)
        return _dedupe_by_chunk(hits), errors


def _dedupe_by_chunk(hits: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """同一路内按 chunk_id 去重，保留首次出现（即排名更靠前）的那条。"""
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in hits:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    return unique
