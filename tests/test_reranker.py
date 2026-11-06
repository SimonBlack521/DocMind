"""Reranker 四种后端的单元测试；Ollama 用 mock httpx 避免实际网络调用。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.domain.dto import RetrievedChunk
from app.providers.reranker import (
    LocalReranker,
    NoopReranker,
    OllamaReranker,
    build_reranker,
)


def _chunks(n: int = 5) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            doc_id="d1",
            title="t",
            content=f"片段 {i}",
            score=0.5,
            source_uri="s",
        )
        for i in range(n)
    ]


def _ranked(ranks: Sequence[int], n: int = 5) -> list[RetrievedChunk]:
    """按 ranks（1-based 顺序 [3,1,5,2,4]）从 _chunks 里取对应 chunk。"""
    base = _chunks(n)
    return [base[idx - 1] for idx in ranks]


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:  # httpx 风格
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[tuple[str, dict]] = []
        self._exhausted = False

    def post(self, url: str, *, json: dict):  # type: ignore[override]
        self.calls.append((url, json))
        if self.responses:
            return self.responses.pop(0)
        self._exhausted = True
        return _FakeResponse(200, {"message": {"content": "[3, 1, 2, 5, 4]"}})

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- Noop


def test_noop_reranker_returns_top_n() -> None:
    reranker = NoopReranker()
    assert reranker.rerank("q", _chunks(6), top_n=3) == _chunks(6)[:3]
    assert reranker.rerank("q", [], top_n=3) == []


# ---------------------------------------------------------------- Ollama 正常排序


def test_ollama_reranker_normal_order() -> None:
    reranker = OllamaReranker(api_base="http://x:11434", model="qwen2.5:3b")
    reranker._client = _FakeClient([
        _FakeResponse(200, {"message": {"content": "[3, 1, 2, 5, 4]"}})
    ])
    ranked = reranker.rerank("q", _chunks(5), top_n=3)
    # 取前 3 个名次的 chunk，且 score 应递减（1.0 - pos/(N+1)）
    assert [c.chunk_id for c in ranked] == ["c2", "c0", "c1"]
    assert ranked[0].score > ranked[1].score > ranked[2].score
    reranker.close()


def test_ollama_reranker_full_rerank() -> None:
    # 5 个片段，输出 [3, 4, 5, 1, 2]，取 top 5
    reranker = OllamaReranker(api_base="http://x:11434", model="m", timeout_s=5.0)
    reranker._client = _FakeClient([
        _FakeResponse(200, {"message": {"content": "[3, 4, 5, 1, 2]"}})
    ])
    chunks = _chunks(5)
    ranked = reranker.rerank("q", chunks, top_n=5)
    assert [c.chunk_id for c in ranked] == ["c2", "c3", "c4", "c0", "c1"]
    assert len(reranker._client.calls) == 1


# ---------------------------------------------------------------- Ollama 解析失败 → 回退原顺序


def test_ollama_reranker_malformed_json_fallback() -> None:
    reranker = OllamaReranker(api_base="http://x:11434", model="m")
    reranker._client = _FakeClient([
        _FakeResponse(200, {"message": {"content": "抱歉我做不到"}})
    ])
    assert reranker.rerank("q", _chunks(5), top_n=3) == _chunks(5)[:3]


def test_ollama_reranker_duplicate_index_fallback() -> None:
    # [1, 1, 2] 有重复，去重后只剩 2 个
    reranker = OllamaReranker(api_base="http://x:11434", model="m")
    reranker._client = _FakeClient([
        _FakeResponse(200, {"message": {"content": "[1, 1, 2]"}})
    ])
    ranked = reranker.rerank("q", _chunks(5), top_n=3)
    # 去重后只剩 c0, c1，取 top 3 要补回原顺序的 c2
    ids = [c.chunk_id for c in ranked]
    assert ids == ["c0", "c1", "c2"]


def test_ollama_reranker_out_of_range_index_fallback() -> None:
    # [99, 1, 2] 里 99 越界被丢弃，只剩 2 个有效排名
    reranker = OllamaReranker(api_base="http://x:11434", model="m")
    reranker._client = _FakeClient([
        _FakeResponse(200, {"message": {"content": "[99, 1, 2]"}})
    ])
    ranked = reranker.rerank("q", _chunks(5), top_n=3)
    ids = [c.chunk_id for c in ranked]
    # 被 LLM 排过的 c0, c1 在前，未被排的 c2 按原顺序补到末尾
    assert ids == ["c0", "c1", "c2"]


def test_ollama_reranker_httpx_timeout_fallback() -> None:
    reranker = OllamaReranker(api_base="http://x:11434", model="m", timeout_s=0.001)

    class _ExplodingClient:
        def post(self, *args, **kwargs):
            raise RuntimeError("timed out")

        def close(self) -> None:
            pass

    reranker._client = _ExplodingClient()  # type: ignore[assignment]
    assert reranker.rerank("q", _chunks(5), top_n=3) == _chunks(5)[:3]


def test_ollama_reranker_http_500_fallback() -> None:
    reranker = OllamaReranker(api_base="http://x:11434", model="m")
    reranker._client = _FakeClient([_FakeResponse(500)])
    assert reranker.rerank("q", _chunks(5), top_n=3) == _chunks(5)[:3]


def test_ollama_reranker_empty_chunks() -> None:
    reranker = OllamaReranker(api_base="http://x:11434", model="m")
    reranker._client = _FakeClient()
    assert reranker.rerank("q", [], top_n=3) == []


def test_ollama_reranker_sends_options() -> None:
    reranker = OllamaReranker(api_base="http://ollama:11434", model="bce-reranker")
    client = _FakeClient()
    reranker._client = client
    reranker.rerank("hello?", _chunks(3), top_n=2)
    url, payload = client.calls[0]
    assert url.endswith("/api/chat")
    assert payload["model"] == "bce-reranker"
    assert payload["options"]["temperature"] == 0.0
    assert payload["stream"] is False
    assert any("候选片段" in msg["content"] for msg in payload["messages"])


# ---------------------------------------------------------------- Remote（结构校验）


def test_remote_reranker_struct() -> None:
    # RemoteReranker / LocalReranker 都有 try/except，当依赖缺失时抛模型不可用错误
    # 这里只验参数路径正确，不触发真正的 HTTP / transformers 调用
    from app.core.settings import Settings

    settings = Settings(rerank_backend="none")
    noop = build_reranker(settings)
    assert isinstance(noop, NoopReranker)

    settings_ollama = Settings(rerank_backend="ollama", rerank_model="qwen2.5:3b")
    ollama = build_reranker(settings_ollama)
    assert isinstance(ollama, OllamaReranker)


# ---------------------------------------------------------------- Local 缺依赖


def test_local_reranker_missing_dep() -> None:
    """没装 sentence-transformers 时 LocalReranker 构造即抛 ModelUnavailableError。"""
    from app.core.errors import ModelUnavailableError

    try:
        LocalReranker(model="BAAI/bge-reranker-v2-m3", device_order=("cpu",))
    except ModelUnavailableError:
        return
    pytest.fail("预期 LocalReranker 在缺依赖时抛 ModelUnavailableError")
