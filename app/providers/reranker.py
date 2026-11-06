"""Reranker 抽象：远程 Jina/Cohere 风格接口、本地 cross-encoder，或 Ollama 轻量 LLM 排序。

四种后端的取舍（由 settings.rerank_backend 选择）：
- none   -> NoopReranker：不重排，直接按检索原分数截断 top_n。零成本，适合
  对延迟敏感或尚未采购 rerank 模型的场景；
- remote -> RemoteReranker：Jina/Cohere 兼容 /rerank 接口，质量最好但要付费
  且数据出内网；
- local  -> LocalReranker：本地 cross-encoder（BCE / bge-reranker），逐对
  (query, chunk) 打分，质量接近远程，代价是要装 torch/transformers 且占显存；
- ollama -> OllamaReranker：复用本地 Ollama 上的轻量 LLM 做纯排序，不装额外
  模型，质量介于 noop 与 cross-encoder 之间，是资源受限环境的折中。
"""

from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import httpx

from app.core.errors import ModelUnavailableError
from app.core.logging import get_logger
from app.core.settings import Settings
from app.domain.dto import RetrievedChunk

logger = get_logger(__name__)


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        """对候选集精排，返回得分最高的 top_n 条。"""

    def close(self) -> None:
        """默认无资源需要释放，子类可覆盖。"""


class NoopReranker(Reranker):
    """关闭重排时直接截断，保持原检索分数。"""

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        return list(chunks)[:top_n]


class RemoteReranker(Reranker):
    """兼容 Jina / Cohere 的 rerank 接口：{"results": [{"index", "relevance_score"}]}。"""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 30.0,
        max_concurrency: int = 0,
    ) -> None:
        if not api_base:
            raise ModelUnavailableError("RERANK_API_BASE 未配置，无法使用远程 Reranker")
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = httpx.Client(timeout=timeout_s)
        # 进程内并发闸门（清单第 24 项）：限制同时打到 rerank 网关的在途请求数
        self._semaphore = (
            threading.BoundedSemaphore(max_concurrency) if max_concurrency > 0 else None
        )

    def _endpoint(self) -> str:
        if self._api_base.endswith("/rerank"):
            return self._api_base
        return f"{self._api_base}/rerank"

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "query": query,
            "documents": [chunk.content for chunk in chunks],
            "top_n": min(top_n, len(chunks)),
        }
        try:
            if self._semaphore is not None:
                with self._semaphore:
                    response = self._client.post(self._endpoint(), json=payload, headers=headers)
            else:
                response = self._client.post(self._endpoint(), json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                "远程 Reranker 请求失败", details={"error": str(exc)}
            ) from exc

        if response.status_code >= 400:
            raise ModelUnavailableError(
                f"远程 Reranker 返回错误：HTTP {response.status_code}",
                details={"body": response.text[:500]},
            )

        body: Any = response.json()
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise ModelUnavailableError("远程 Reranker 返回结构非法")

        scored: list[RetrievedChunk] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", -1))
            if not 0 <= index < len(chunks):
                continue
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            scored.append(replace(chunks[index], score=score))

        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_n]

    def close(self) -> None:
        self._client.close()


class LocalReranker(Reranker):
    """本地 cross-encoder（BCE / bge-reranker），懒加载。"""

    def __init__(
        self,
        *,
        model: str,
        device_order: Sequence[str] = ("cuda", "cpu"),
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        # 构造时就做轻量 import 检查（不加载模型权重），避免首次 rerank 才炸
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
        except ImportError as exc:
            raise ModelUnavailableError(
                "本地 Reranker 依赖未安装：pip install -e '.[local-models]'"
            ) from exc

        self._model_name = model
        self._device_order = list(device_order)
        self._max_length = max_length
        self._batch_size = max(1, batch_size)
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._tokenizer, self._model, self._device

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelUnavailableError(
                "本地 Reranker 依赖未安装：pip install -e '.[local-models]'"
            ) from exc

        device = torch.device("cpu")
        for candidate in self._device_order:
            if candidate == "cpu" or (candidate == "cuda" and torch.cuda.is_available()):
                device = torch.device(candidate)
                break

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        except Exception as exc:
            raise ModelUnavailableError(
                f"本地 Reranker 模型加载失败：{self._model_name}", details={"error": str(exc)}
            ) from exc

        model.eval()
        model.to(device)
        self._tokenizer, self._model, self._device = tokenizer, model, device
        logger.info("本地 Reranker 已加载 model=%s device=%s", self._model_name, device)
        return tokenizer, model, device

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        import torch

        tokenizer, model, device = self._load()
        # 输入长度分桶：按正文长度排序后再切 batch，让同一 batch 内文本长度接近，
        # padding 显著减少（短样本不会被长样本补齐），GPU 算力利用率更高。
        order = sorted(range(len(chunks)), key=lambda index: len(chunks[index].content))
        scored_by_index: list[RetrievedChunk | None] = [None] * len(chunks)

        for start in range(0, len(order), self._batch_size):
            batch_indices = order[start : start + self._batch_size]
            pairs = [[query, chunks[index].content] for index in batch_indices]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            scores = logits.view(-1).float().cpu().tolist()
            for offset, score in enumerate(scores):
                index = batch_indices[offset]
                scored_by_index[index] = replace(chunks[index], score=float(score))

        scored = [chunk for chunk in scored_by_index if chunk is not None]
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_n]


_OLLAMA_RERANK_SYSTEM = """你是严格的检索片段排序器。给定【问题】与编号的候选片段，
按与问题的相关性从高到低排序后，只输出一个 JSON 数组，不要任何解释。
数组元素是候选片段的 1 开始的编号，且每个编号只能出现一次。
格式示例：[3, 1, 5, 2, 4]"""


class OllamaReranker(Reranker):
    """用本地 Ollama 上的轻量 LLM（如 qwen2.5:3b）做纯排序型重排。

    不走 cross-encoder 的二分类打分，而是让 LLM 直接对候选片段排序并输出编号数组。
    好处是不需要额外装 rerank 模型或 torch/transformers，复用已有 Ollama 实例即可；
    质量上弱于专用 cross-encoder，但在 20 条以内的候选集上常能比纯向量 BM25 融合更稳。

    失败回退设计：rerank 是"锦上添花"环节，任何一步出错（请求异常、HTTP 错误、
    返回结构非法、JSON 数组解析失败）都不上抛异常，而是记录 warning 并返回
    原检索顺序截断 top_n——宁可重排失效，也不让问答主链路失败。
    编号输出还会做去重与范围校验（1..len(chunks)），模型幻觉出的非法编号直接丢弃；
    没被排到的片段按原顺序补到末尾，保证返回数量足够 top_n。
    """

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        timeout_s: float = 30.0,
        max_chunk_chars: int = 400,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._max_chunk_chars = max_chunk_chars
        self._client = httpx.Client(timeout=timeout_s)

    def _endpoint(self) -> str:
        return f"{self._api_base}/api/chat"

    def _parse_ranking(self, text: str, total: int) -> list[int] | None:
        match = re.search(r"\[[\d\s,]+\]", text)
        if not match:
            return None
        try:
            raw = json.loads(match.group(0))
        except ValueError:
            return None
        if not isinstance(raw, list):
            return None
        ranks: list[int] = []
        seen: set[int] = set()
        for item in raw:
            if isinstance(item, str) and item.isdigit():
                idx = int(item)
            elif isinstance(item, int):
                idx = item
            else:
                continue
            if 1 <= idx <= total and idx not in seen:
                ranks.append(idx - 1)
                seen.add(idx)
        return ranks or None

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        lines = [
            f"{idx}. {chunk.content[: self._max_chunk_chars].strip().replace(chr(10), ' ')}"
            for idx, chunk in enumerate(chunks, 1)
        ]
        user_prompt = (
            f"【问题】\n{query}\n\n【候选片段】\n" + "\n".join(lines) + "\n\n"
            "只输出一个 JSON 数组，不要解释。"
        )

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _OLLAMA_RERANK_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        }
        try:
            response = self._client.post(self._endpoint(), json=payload)
        except Exception as exc:
            logger.warning("Ollama Reranker 请求失败，回退原顺序：%s", exc)
            return list(chunks)[:top_n]

        if response.status_code >= 400:
            logger.warning(
                "Ollama Reranker HTTP %s，回退原顺序", response.status_code
            )
            return list(chunks)[:top_n]

        try:
            body = response.json()
            text = body["message"]["content"]
        except (KeyError, ValueError, TypeError):
            logger.warning("Ollama Reranker 返回结构非法，回退原顺序")
            return list(chunks)[:top_n]

        ranks = self._parse_ranking(text, len(chunks))
        if ranks is None:
            logger.warning(
                "Ollama Reranker 无法解析输出（model=%s），回退原顺序。原始输出：%s",
                self._model,
                text[:200],
            )
            return list(chunks)[:top_n]

        # 名次越高得分越高；给个 0.9 / 0.8 递减，避免全挤在 0 附近
        scored: list[RetrievedChunk] = []
        used: set[int] = set()
        for pos, idx in enumerate(ranks):
            if idx in used:
                continue
            used.add(idx)
            chunk = chunks[idx]
            score = 1.0 - (pos + 1) / (len(ranks) + 1)
            scored.append(replace(chunk, score=score))
        # 未被 LLM 排到的 chunk 按原顺序补到末尾，避免返回不足 top_n
        for idx, chunk in enumerate(chunks):
            if idx not in used:
                scored.append(chunk)
        return scored[:top_n]

    def close(self) -> None:
        self._client.close()


def build_reranker(settings: Settings) -> Reranker:
    """按 settings.rerank_backend 组装实现；未配置远程地址时自动落到 local 兜底。"""
    if settings.rerank_backend == "none":
        return NoopReranker()

    device_order = tuple(
        device for device in settings.model_device_order if device in {"cuda", "cpu"}
    ) or ("cpu",)

    if settings.rerank_backend == "ollama":
        return OllamaReranker(
            api_base=settings.rerank_ollama_base,
            model=settings.rerank_model,
            timeout_s=settings.rerank_timeout_s,
        )

    if settings.rerank_backend == "local" or not settings.rerank_api_base:
        return LocalReranker(
            model=settings.rerank_model,
            device_order=device_order,
            batch_size=settings.rerank_batch_size,
        )

    return RemoteReranker(
        api_base=settings.rerank_api_base,
        model=settings.rerank_model,
        api_key=settings.rerank_api_key,
        timeout_s=settings.rerank_timeout_s,
        max_concurrency=settings.rerank_max_concurrency,
    )
