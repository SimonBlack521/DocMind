"""流式问答测试：RAGChain 流式约束 + SSE 端点契约（不访问外部服务）。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from fastapi.testclient import TestClient

from app.core.errors import ModelUnavailableError
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import KBConfig
from app.generation.chain import RAGChain
from app.generation.citations import build_citations
from app.generation.prompts import NO_CONTEXT_ANSWER
from app.main import create_app
from app.orchestration.pipeline import QueryStream
from app.providers.llm import ChatMessage, LLMClient, LLMResponse


class FakeLLM(LLMClient):
    """记录调用次数的假 LLM：stream_chat 逐段产出预置 token。"""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens or []
        self.chat_calls = 0
        self.stream_calls = 0

    @property
    def model(self) -> str:
        return "fake-llm"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.chat_calls += 1
        return LLMResponse(text="".join(self._tokens), model=self.model)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self.stream_calls += 1
        return iter(self._tokens)


def _chunk(chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content="差旅报销需提前审批",
        score=0.9,
        doc_id="d1",
        source_uri="kb://finance/policy.txt",
        title="报销制度",
    )


def test_stream_answer_without_chunks_never_calls_llm() -> None:
    """无检索资料是硬约束：流式路径同样不调用 LLM，直接产出固定文案。"""
    llm = FakeLLM(tokens=["不应出现"])
    stream = RAGChain(llm).stream_answer("报销流程？", [])

    assert list(stream.tokens) == [NO_CONTEXT_ANSWER]
    assert stream.citations == []
    assert stream.model == ""
    assert llm.chat_calls == 0
    assert llm.stream_calls == 0


def test_stream_answer_with_chunks_streams_tokens_and_citations() -> None:
    llm = FakeLLM(tokens=["需要", "提前", "审批"])
    stream = RAGChain(llm).stream_answer("报销流程？", [_chunk()])

    assert list(stream.tokens) == ["需要", "提前", "审批"]
    assert stream.model == "fake-llm"
    assert [c.index for c in stream.citations] == [1]
    assert stream.citations[0].chunk_id == "c1"
    assert llm.stream_calls == 1


class FakePipeline:
    """模拟管线：query_stream 直接返回预置流式结果，不触达检索与模型。"""

    def __init__(self, stream: QueryStream) -> None:
        self._stream = stream

    def query_stream(
        self,
        kb: KBConfig,
        question: str,
        *,
        top_k: int | None = None,
        use_rewrite: bool | None = None,
        use_rerank: bool | None = None,
        allowed_groups: frozenset[str] | None = None,
        excluded_doc_ids: frozenset[str] | None = None,
        history: list[tuple[str, str]] | None = None,
        mode: str = "auto",
    ) -> QueryStream:
        return self._stream


class FakeRepo:
    """捕获审计写入的假仓库；superseded 返回空集合（本测试不覆盖该过滤）。

    会话方法只做最小内存实现：本测试关注 SSE 事件契约，多轮行为由
    tests/test_conversation.py 覆盖。
    """

    def __init__(self) -> None:
        self.audits: list[dict[str, Any]] = []
        self.messages: dict[str, list[tuple[str, str]]] = {}

    async def write_audit(self, **kwargs: Any) -> None:
        self.audits.append(kwargs)

    async def list_superseded_doc_ids(self, *, kb_id: str, tenant_id: str) -> frozenset[str]:
        return frozenset()

    async def list_hidden_doc_ids(self, *, kb_id: str, tenant_id: str) -> frozenset[str]:
        # v2.1 检索不可见集合（已被替代 + DELETING）：测试桩返回空集
        return frozenset()

    async def create_conversation(
        self,
        conversation_id: str,
        *,
        tenant_id: str,
        kb_id: str,
        user_id: str,
        title: str | None = None,
    ) -> None:
        self.messages[conversation_id] = []

    async def get_conversation(self, conversation_id: str) -> None:
        return None

    async def list_conversation_messages(
        self, conversation_id: str, *, limit: int = 20
    ) -> list[Any]:
        return []

    async def append_conversation_messages(
        self, conversation_id: str, messages: list[Any]
    ) -> None:
        self.messages.setdefault(conversation_id, []).extend(messages)


def _parse_sse(body: str) -> list[tuple[str, Any]]:
    """把 SSE 报文解析成 (event, data) 序列，便于断言事件次序。"""
    events: list[tuple[str, Any]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        event = next(
            line.removeprefix("event: ") for line in lines if line.startswith("event: ")
        )
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((event, json.loads(data)))
    return events


def test_chat_stream_sse_contract() -> None:
    app = create_app()
    with TestClient(app) as client:
        chunk = _chunk()
        app.state.pipeline = FakePipeline(
            QueryStream(
                citations=build_citations([chunk]),
                model="fake-llm",
                tokens=iter(["需要", "提前审批"]),
                chunks=[chunk],
            )
        )
        repo = FakeRepo()
        app.state.db_repo = repo

        response = client.post(
            "/api/v1/kbs/finance/chat:stream", json={"question": "报销流程？"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    # v2.1（清单 P1-03）：token 推完后增加最终引用校验事件，done 前多一个 validation
    # v2.2（答案验证/SSE 一致性）：validation 前增加 answer 事件，用于把前端已渲染的
    # 原始 token 替换为"清理无效引用后的最终正文"（与会话记录、非流式响应一致）
    assert [event for event, _ in events] == [
        "citations",
        "token",
        "token",
        "answer",
        "validation",
        "done",
    ]
    assert events[0][1][0]["chunk_id"] == "c1"
    assert "".join(data["content"] for _, data in events[1:3]) == "需要提前审批"
    # 最终正文以 answer 事件为准（replace=True）
    assert events[3][1]["content"] == "需要提前审批"
    assert events[3][1]["replace"] is True
    validation = events[4][1]
    assert isinstance(validation["citations"], list)
    assert validation["total_answer_refs"] >= 0
    # v2.2 独立答案验证状态随 validation 一起下发
    assert validation["verification"]["status"] in {
        "verified",
        "unverified",
        "rejected",
    }
    done = events[5][1]
    assert done["model"] == "fake-llm"
    assert done["retrieved_count"] == 1
    # 多轮升级：单轮调用也会分配会话 id，随 done 事件返回
    assert done["conversation_id"]

    # 问答完成后写入审计，且标记为流式
    assert len(repo.audits) == 1
    audit = repo.audits[0]
    assert audit["action"] == "chat.query"
    assert audit["detail"]["stream"] is True
    assert audit["detail"]["retrieved_count"] == 1


def test_chat_stream_replaces_invalid_refs_in_answer_and_history() -> None:
    """回归（用户复现）：无效 [n] 不能只从引用列表删掉，正文与会话记录也必须一致。

    场景：只有 1 条引用，模型输出 [9]；旧实现只改 citations，界面与历史仍保留
    "[9]"。修复后 answer 事件给出清理文本，落库也用同一份。
    """
    app = create_app()
    with TestClient(app) as client:
        chunk = _chunk()
        app.state.pipeline = FakePipeline(
            QueryStream(
                citations=build_citations([chunk]),
                model="fake-llm",
                tokens=iter(["需要提前审批", "[9]"]),
                chunks=[chunk],
            )
        )
        repo = FakeRepo()
        app.state.db_repo = repo

        response = client.post(
            "/api/v1/kbs/finance/chat:stream", json={"question": "报销流程？"}
        )

    events = _parse_sse(response.text)
    answer_event = next(data for event, data in events if event == "answer")
    assert answer_event["content"] == "需要提前审批"
    validation = next(data for event, data in events if event == "validation")
    assert validation["out_of_range_refs"] == 1
    # 有效引用 [1] 保留，越界的 [9] 已被清理
    assert [item["index"] for item in validation["citations"]] == [1]

    conversation_id = next(
        data["conversation_id"] for event, data in events if event == "done"
    )
    stored = repo.messages[conversation_id]
    assert stored[-1].content == "需要提前审批"
    assert "[9]" not in stored[-1].content


def test_chat_stream_llm_failure_emits_error_event() -> None:
    """LLM 流式中途失败：已产出的 token 保留，以 error 事件收尾，不写审计。"""

    def failing_tokens() -> Iterator[str]:
        yield "部分回答"
        raise ModelUnavailableError("模型不可用")

    app = create_app()
    with TestClient(app) as client:
        app.state.pipeline = FakePipeline(
            QueryStream(citations=[], model="fake-llm", tokens=failing_tokens(), chunks=[])
        )
        repo = FakeRepo()
        app.state.db_repo = repo

        response = client.post(
            "/api/v1/kbs/finance/chat:stream", json={"question": "任意问题"}
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event for event, _ in events] == ["citations", "token", "error"]
    assert events[-1][1]["code"] == "model_unavailable"
    assert repo.audits == []
