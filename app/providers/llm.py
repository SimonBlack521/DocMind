"""LLM 抽象：OpenAI 兼容远程接口或本地 Ollama，支持一次性与流式两种调用。

两种后端的取舍：
- RemoteLLM：任何 OpenAI 兼容的 /chat/completions 服务（官方 OpenAI、vLLM、
  各类云厂商网关均可）。走 SSE 流式协议（data: {...} 行，data: [DONE] 结束）。
  优点是质量高、免本地算力；代价是依赖网络与 API Key，数据要出内网。
- OllamaLLM：本地 Ollama 实例（/api/chat），协议是 NDJSON（每行一个 JSON，
  done=true 结束），参数名也不同（max_tokens 映射为 options.num_predict）。
  优点是数据不出本机、零密钥成本；质量取决于本地模型规模。

上层（orchestration）只依赖 LLMClient 接口，由 build_llm 按 settings.llm_backend
选择实现，切换后端不改任何业务代码。
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from app.core.errors import ConfigurationError, ModelUnavailableError
from app.core.settings import Settings

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(ABC):
    @property
    @abstractmethod
    def model(self) -> str:
        """当前配置的模型名。"""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """一次性返回完整回答。"""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式返回回答增量文本。

        返回的是惰性迭代器：首次迭代才真正发起请求；连接/响应错误在迭代时
        抛 ModelUnavailableError，调用方需在消费循环里捕获。
        """

    def close(self) -> None:
        """默认无资源需要释放，子类可覆盖。"""


class RemoteLLM(LLMClient):
    """OpenAI 兼容远程后端：自带 /chat/completions 补全、Bearer 鉴权与 SSE 解析。"""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.1,
        timeout_s: float = 120.0,
        max_concurrency: int = 0,
    ) -> None:
        if not api_base:
            raise ConfigurationError("LLM_API_BASE 未配置，无法使用远程 LLM")
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = httpx.Client(timeout=timeout_s)
        # 进程内并发闸门（清单第 24 项）：流式与非流式共用额度，
        # 流式请求持有信号量直到响应流结束（在途请求才占 LLM 服务容量）
        self._semaphore = (
            threading.BoundedSemaphore(max_concurrency) if max_concurrency > 0 else None
        )

    @property
    def model(self) -> str:
        return self._model

    def _endpoint(self) -> str:
        if self._api_base.endswith("/chat/completions"):
            return self._api_base
        return f"{self._api_base}/chat/completions"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            if self._semaphore is not None:
                with self._semaphore:
                    response = self._client.post(self._endpoint(), json=payload, headers=headers)
            else:
                response = self._client.post(self._endpoint(), json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ModelUnavailableError("远程 LLM 请求失败", details={"error": str(exc)}) from exc

        if response.status_code >= 400:
            raise ModelUnavailableError(
                f"远程 LLM 返回错误：HTTP {response.status_code}",
                details={"body": response.text[:500]},
            )

        body: Any = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ModelUnavailableError("远程 LLM 返回结构非法")

        text = str(choices[0].get("message", {}).get("content", ""))
        usage_raw = body.get("usage", {}) if isinstance(body, dict) else {}
        usage = {key: int(value) for key, value in usage_raw.items() if isinstance(value, int)}
        return LLMResponse(text=text, model=self._model, usage=usage)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            # 生成器惰性执行：首次迭代才 acquire，消费结束（或调用方 close）才释放，
            # 保证流式请求在整个响应期间只占用一个并发额度
            if self._semaphore is not None:
                self._semaphore.acquire()
            try:
                with self._client.stream(
                    "POST", self._endpoint(), json=payload, headers=headers
                ) as response:
                    if response.status_code >= 400:
                        raise ModelUnavailableError(
                            f"远程 LLM 返回错误：HTTP {response.status_code}",
                            details={
                                "body": response.read().decode("utf-8", "replace")[:500]
                            },
                        )
                    # OpenAI 兼容的 SSE：每行 data: {...}，以 data: [DONE] 结束
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            return
                        chunk: Any = json.loads(data)
                        choices = chunk.get("choices") if isinstance(chunk, dict) else None
                        if not isinstance(choices, list) or not choices:
                            continue
                        delta = choices[0].get("delta")
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(content, str) and content:
                            yield content
            finally:
                if self._semaphore is not None:
                    self._semaphore.release()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                "远程 LLM 流式请求失败", details={"error": str(exc)}
            ) from exc

    def close(self) -> None:
        self._client.close()


class OllamaLLM(LLMClient):
    """本地 Ollama 后端：/api/chat + NDJSON 流式，无需 API Key，适合内网/开发环境。"""

    def __init__(
        self,
        *,
        api_base: str = "http://127.0.0.1:11434",
        model: str,
        temperature: float = 0.1,
        timeout_s: float = 120.0,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._client = httpx.Client(timeout=timeout_s)

    @property
    def model(self) -> str:
        return self._model

    def _endpoint(self) -> str:
        return f"{self._api_base}/api/chat"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        options: dict[str, Any] = {
            "temperature": self._temperature if temperature is None else temperature
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        try:
            response = self._client.post(self._endpoint(), json=payload)
        except httpx.HTTPError as exc:
            raise ModelUnavailableError("Ollama 请求失败", details={"error": str(exc)}) from exc

        if response.status_code >= 400:
            raise ModelUnavailableError(
                f"Ollama 返回错误：HTTP {response.status_code}",
                details={"body": response.text[:500]},
            )

        body: Any = response.json()
        message = body.get("message") if isinstance(body, dict) else None
        if not isinstance(message, dict):
            raise ModelUnavailableError("Ollama 返回结构非法")

        usage = {
            key: int(body[key])
            for key in ("prompt_eval_count", "eval_count")
            if isinstance(body, dict) and isinstance(body.get(key), int)
        }
        return LLMResponse(text=str(message.get("content", "")), model=self._model, usage=usage)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        options: dict[str, Any] = {
            "temperature": self._temperature if temperature is None else temperature
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": options,
        }
        try:
            with self._client.stream("POST", self._endpoint(), json=payload) as response:
                if response.status_code >= 400:
                    raise ModelUnavailableError(
                        f"Ollama 返回错误：HTTP {response.status_code}",
                        details={"body": response.read().decode("utf-8", "replace")[:500]},
                    )
                # Ollama 流式是 NDJSON：每行一个 JSON，done=true 表示结束
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk: Any = json.loads(line)
                    if not isinstance(chunk, dict):
                        continue
                    message = chunk.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content, str) and content:
                        yield content
                    if chunk.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                "Ollama 流式请求失败", details={"error": str(exc)}
            ) from exc

    def close(self) -> None:
        self._client.close()


def build_llm(settings: Settings) -> LLMClient:
    """按 settings.llm_backend 组装实现："ollama" 走本地，其余一律走 OpenAI 兼容远程。"""
    if settings.llm_backend == "ollama":
        return OllamaLLM(
            api_base=settings.llm_api_base or "http://127.0.0.1:11434",
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_s=settings.llm_timeout_s,
        )

    return RemoteLLM(
        api_base=settings.llm_api_base,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_timeout_s,
        max_concurrency=settings.llm_max_concurrency,
    )
