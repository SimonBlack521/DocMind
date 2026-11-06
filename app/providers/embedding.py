"""Embedding 抽象：远程优先，失败时降级到本地 GPU/CPU。

三种后端的取舍：
- RemoteEmbeddingProvider：OpenAI 兼容 /embeddings 接口（BGE、Jina、火山、通义
  等均可代理）。吞吐高、不占本机显存，但依赖网络；
- LocalEmbeddingProvider：本地 SentenceTransformer，懒加载权重，按
  cuda -> cpu 顺序选设备。零网络依赖，代价是首次调用慢、吃内存/显存；
- FallbackEmbeddingProvider：把 remote 与 local 串成降级链，remote 抛
  ModelUnavailableError 时自动落到 local，保证入库/检索不被单点故障卡死。

降级硬约束：同模型、同维度。维度不一致直接抛 ModelDimensionMismatchError，
避免不同模型产生的向量混进同一个 collection（那会让相似度计算彻底失真）。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

from app.core.errors import ModelDimensionMismatchError, ModelUnavailableError
from app.core.logging import get_logger
from app.core.settings import Settings

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    def __init__(self, *, model: str, dim: int) -> None:
        self.model = model
        self.dim = dim

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化，顺序与输入严格一致。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化。"""

    def healthcheck(self) -> bool:
        try:
            self.embed_query("ping")
        except ModelUnavailableError:
            return False
        return True

    def close(self) -> None:
        """默认无资源需要释放，子类可覆盖。"""

    def _ensure_dim(self, vector: Sequence[float], *, chunk_id: str) -> list[float]:
        values = [float(item) for item in vector]
        if len(values) != self.dim:
            raise ModelDimensionMismatchError(
                f"Embedding 维度不一致：期望 {self.dim}，实际 {len(values)}",
                details={"model": self.model, "chunk_id": chunk_id},
            )
        return values


class RemoteEmbeddingProvider(EmbeddingProvider):
    """OpenAI 兼容 /embeddings 接口（BGE、Jina、火山、通义等均可代理）。"""

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        api_base: str,
        api_key: str = "",
        timeout_s: float = 30.0,
        batch_size: int = 32,
        max_concurrency: int = 0,
    ) -> None:
        super().__init__(model=model, dim=dim)
        if not api_base:
            raise ModelUnavailableError("EMBEDDING_API_BASE 未配置，无法使用远程 Embedding")
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._batch_size = max(1, batch_size)
        self._client = httpx.Client(timeout=timeout_s)
        # 进程内并发闸门（清单第 24 项）：限制同时打到模型网关的在途请求数，
        # 0 表示不限制；多实例部署时总并发 = 该值 × 实例数
        self._semaphore = (
            threading.BoundedSemaphore(max_concurrency) if max_concurrency > 0 else None
        )

    def _endpoint(self) -> str:
        if self._api_base.endswith("/embeddings"):
            return self._api_base
        return f"{self._api_base}/embeddings"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(self._request(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _request(self, texts: list[str]) -> list[list[float]]:
        if self._semaphore is not None:
            with self._semaphore:
                return self._do_request(texts)
        return self._do_request(texts)

    def _do_request(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = self._client.post(
                self._endpoint(), json={"model": self.model, "input": texts}, headers=headers
            )
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                f"远程 Embedding 请求失败：{self._endpoint()}", details={"error": str(exc)}
            ) from exc

        if response.status_code >= 400:
            raise ModelUnavailableError(
                f"远程 Embedding 返回错误：HTTP {response.status_code}",
                details={"body": response.text[:500]},
            )

        payload: Any = response.json()
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise ModelUnavailableError("远程 Embedding 返回结构非法")

        ordered: dict[int, list[float]] = {}
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise ModelUnavailableError("远程 Embedding 返回条目非法")
            index = int(item.get("index", position))
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise ModelUnavailableError("远程 Embedding 缺少 embedding 字段")
            ordered[index] = self._ensure_dim(vector, chunk_id=f"batch:{index}")

        missing = [i for i in range(len(texts)) if i not in ordered]
        if missing:
            raise ModelUnavailableError(
                "远程 Embedding 返回条目不完整", details={"missing": missing}
            )
        return [ordered[i] for i in range(len(texts))]

    def close(self) -> None:
        self._client.close()


class LocalEmbeddingProvider(EmbeddingProvider):
    """本地 SentenceTransformer，模型懒加载，按 cuda -> cpu 顺序选设备。"""

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        device_order: Sequence[str] = ("cuda", "cpu"),
        batch_size: int = 32,
    ) -> None:
        super().__init__(model=model, dim=dim)
        self._device_order = list(device_order)
        self._batch_size = max(1, batch_size)
        self._encoder: Any = None
        self._device: str | None = None

    def _load(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelUnavailableError(
                "本地 Embedding 依赖未安装：pip install -e '.[local-models]'"
            ) from exc

        device = "cpu"
        for candidate in self._device_order:
            if candidate == "cpu" or (candidate == "cuda" and torch.cuda.is_available()):
                device = candidate
                break

        try:
            encoder = SentenceTransformer(self.model, device=device)
        except Exception as exc:
            raise ModelUnavailableError(
                f"本地 Embedding 模型加载失败：{self.model}", details={"error": str(exc)}
            ) from exc

        actual_dim = encoder.get_sentence_embedding_dimension()
        if actual_dim is not None and int(actual_dim) != self.dim:
            raise ModelDimensionMismatchError(
                f"本地模型维度 {actual_dim} 与知识库配置维度 {self.dim} 不一致",
                details={"model": self.model},
            )

        self._encoder = encoder
        self._device = device
        logger.info("本地 Embedding 已加载 model=%s device=%s", self.model, device)
        return encoder

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        encoder = self._load()
        raw = encoder.encode(
            list(texts),
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        vectors = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return [self._ensure_dim(vector, chunk_id=f"batch:{i}") for i, vector in enumerate(vectors)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FallbackEmbeddingProvider(EmbeddingProvider):
    """按顺序尝试多个后端，只有同模型同维度的后端才允许挂进降级链。"""

    def __init__(self, providers: Sequence[EmbeddingProvider]) -> None:
        if not providers:
            raise ModelUnavailableError("Embedding 降级链为空")
        primary = providers[0]
        super().__init__(model=primary.model, dim=primary.dim)
        for provider in providers:
            if provider.dim != self.dim or provider.model != self.model:
                raise ModelDimensionMismatchError(
                    "Embedding 降级链中的后端必须同模型同维度",
                    details={
                        "expected": f"{self.model}/{self.dim}",
                        "actual": f"{provider.model}/{provider.dim}",
                    },
                )
        self._providers = list(providers)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                return provider.embed_documents(texts)
            except ModelUnavailableError as exc:
                logger.warning("Embedding 后端不可用，尝试降级：%s", exc.message)
                last_error = exc
        raise ModelUnavailableError("所有 Embedding 后端均不可用") from last_error

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def close(self) -> None:
        for provider in self._providers:
            provider.close()


def build_embedding_provider(
    settings: Settings, *, kb_model: str | None = None, kb_dim: int | None = None
) -> EmbeddingProvider:
    """按配置组装：remote -> 本地 cuda -> 本地 cpu。"""
    model = kb_model or settings.embedding_model
    dim = kb_dim or settings.embedding_dim
    device_order = tuple(
        device for device in settings.model_device_order if device in {"cuda", "cpu"}
    ) or ("cpu",)

    local = LocalEmbeddingProvider(
        model=model, dim=dim, device_order=device_order, batch_size=settings.embedding_batch_size
    )

    if settings.embedding_backend == "local" or not settings.embedding_api_base:
        return local

    remote = RemoteEmbeddingProvider(
        model=model,
        dim=dim,
        api_base=settings.embedding_api_base,
        api_key=settings.embedding_api_key,
        timeout_s=settings.embedding_timeout_s,
        batch_size=settings.embedding_batch_size,
        max_concurrency=settings.embedding_max_concurrency,
    )
    return FallbackEmbeddingProvider([remote, local])
