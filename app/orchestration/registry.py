"""按知识库装配运行时依赖，并缓存较重的连接对象。

- Embedding 可按知识库覆盖模型/维度，因此每个知识库一个实例；
- Milvus collection / ES 索引按知识库隔离，每个知识库一套连接；
- Reranker 与 LLM 使用全局配置，整个进程共享一个实例；
- 所有对象都是懒加载：构建时不建立网络连接，首次调用才真正连 Milvus/ES。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.core.settings import Settings
from app.domain.kb_config import KBConfig
from app.providers.embedding import EmbeddingProvider, build_embedding_provider
from app.providers.llm import LLMClient, build_llm
from app.providers.reranker import Reranker, build_reranker
from app.storage.es_store import ElasticsearchKeywordStore
from app.storage.keyword_base import KeywordStore
from app.storage.milvus_store import MilvusVectorStore
from app.storage.vector_base import VectorStore


@dataclass(slots=True)
class KBRuntime:
    """一个知识库的存储与向量化依赖。"""

    config: KBConfig
    embedding: EmbeddingProvider
    vector_store: VectorStore
    keyword_store: KeywordStore


class RuntimeRegistry:
    """运行时注册表：按 kb_id 缓存 KBRuntime，LLM / Reranker 进程内共享单例。

    线程安全：所有懒加载都走同一把锁；runtime() 热路径用双检锁避免每次加锁。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._runtimes: dict[str, KBRuntime] = {}
        self._lock = threading.Lock()
        self._llm: LLMClient | None = None
        self._reranker: Reranker | None = None

    def runtime(self, kb: KBConfig) -> KBRuntime:
        """按知识库取运行时；未命中时加锁构建（双检锁，避免并发下重复建连）。"""
        cached = self._runtimes.get(kb.kb_id)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._runtimes.get(kb.kb_id)
            if cached is not None:
                return cached
            runtime = self._build_runtime(kb)
            self._runtimes[kb.kb_id] = runtime
            return runtime

    def evict(self, kb_id: str) -> None:
        """配置变更 / 删除知识库后丢弃对应运行时，下次访问按新配置重建。"""
        with self._lock:
            runtime = self._runtimes.pop(kb_id, None)
        if runtime is not None:
            runtime.embedding.close()
            runtime.vector_store.close()
            runtime.keyword_store.close()

    def llm(self) -> LLMClient:
        """进程级共享的 LLM 客户端：首次调用才按全局配置构建（懒加载）。"""
        with self._lock:
            if self._llm is None:
                self._llm = build_llm(self._settings)
            return self._llm

    def reranker(self) -> Reranker:
        """进程级共享的精排客户端：首次调用才按全局配置构建（懒加载）。"""
        with self._lock:
            if self._reranker is None:
                self._reranker = build_reranker(self._settings)
            return self._reranker

    def _build_runtime(self, kb: KBConfig) -> KBRuntime:
        """真正装配一套知识库运行时：Embedding 可被知识库覆盖，存储按知识库隔离。"""
        settings = self._settings
        return KBRuntime(
            config=kb,
            embedding=build_embedding_provider(
                settings, kb_model=kb.embedding.model, kb_dim=kb.embedding.dim
            ),
            vector_store=MilvusVectorStore(
                uri=settings.milvus_uri,
                collection=kb.milvus_collection,
                token=settings.milvus_token,
                db_name=settings.milvus_db_name,
                index_type=settings.milvus_index_type,
                hnsw_m=settings.milvus_hnsw_m,
                hnsw_ef_construction=settings.milvus_hnsw_ef_construction,
                hnsw_ef_search=settings.milvus_hnsw_ef_search,
                auto_reconcile_index=settings.milvus_auto_reconcile_index,
            ),
            keyword_store=ElasticsearchKeywordStore(
                url=settings.es_url,
                index=kb.es_index,
                alias=kb.es_alias,
                username=settings.es_username,
                password=settings.es_password,
                analyzer=settings.es_analyzer,
                number_of_shards=settings.es_number_of_shards,
                number_of_replicas=settings.es_number_of_replicas,
            ),
        )

    def close(self) -> None:
        """进程退出时统一释放：先在锁内摘出全部引用，再在锁外逐个关闭连接。"""
        with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
            llm, reranker = self._llm, self._reranker
            self._llm = None
            self._reranker = None

        for runtime in runtimes:
            runtime.embedding.close()
            runtime.vector_store.close()
            runtime.keyword_store.close()
        if llm is not None:
            llm.close()
        if reranker is not None:
            reranker.close()
