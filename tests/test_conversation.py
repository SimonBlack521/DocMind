"""多轮对话测试（v2.0）：追问改写器 + 会话 API 契约（不访问外部服务）。

- 改写器是纯逻辑：用假 LLM 断言"只在追问特征命中时才调用、失败回退原问题"；
- 会话 API 走 FastAPI TestClient + 假仓库/假管线，验证归属校验（403/404）
  与"历史注入检索、问答落库"两条关键链路。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from fastapi.testclient import TestClient

from app.auth.base import AuthProvider, CurrentUser
from app.core.errors import ModelUnavailableError
from app.db.repo import ConversationInfo, ConversationMessage
from app.main import create_app
from app.orchestration.pipeline import QueryResult, QueryStream
from app.providers.llm import ChatMessage, LLMClient, LLMResponse
from app.retrieval.conversation import ConversationQueryRewriter

_TENANT = "default"
_KB_ID = "kb-finance-001"


# ------------------------------------------------------------------ 追问改写器
class FakeLLM(LLMClient):
    """记录 chat 调用次数并返回预置改写结果的假 LLM。"""

    def __init__(self, text: str = "", *, fail: bool = False) -> None:
        self._text = text
        self._fail = fail
        self.chat_calls = 0

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
        if self._fail:
            raise ModelUnavailableError("模型不可用")
        return LLMResponse(text=self._text, model=self.model)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        return iter([self._text])


def test_rewriter_returns_original_without_history() -> None:
    """无历史 / 无 LLM 时零调用、原样返回。"""
    llm = FakeLLM("年假有多少天")
    rewriter = ConversationQueryRewriter(llm)

    assert rewriter.rewrite("它的有效期呢？", None) == "它的有效期呢？"
    assert rewriter.rewrite("它的有效期呢？", []) == "它的有效期呢？"
    assert ConversationQueryRewriter(None).rewrite("它的有效期呢？", [("user", "年假")]) == (
        "它的有效期呢？"
    )
    assert llm.chat_calls == 0


def test_rewriter_skips_self_contained_question() -> None:
    """长问题或没有指代特征的问句视为自包含，不调 LLM。"""
    llm = FakeLLM("改写结果")
    rewriter = ConversationQueryRewriter(llm)
    history = [("user", "差旅报销流程是什么"), ("assistant", "需要提前审批")]

    assert rewriter.rewrite("差旅报销流程是什么？", history) == "差旅报销流程是什么？"
    # 超长问题即使含"该"也按自包含处理
    long_question = "请详细说明差旅报销制度中关于该差旅费用审批的完整流程和所需材料"
    assert rewriter.rewrite(long_question, history) == long_question
    assert llm.chat_calls == 0


def test_rewriter_rewrites_followup_with_history() -> None:
    llm = FakeLLM("差旅报销需要提前几天审批？")
    rewriter = ConversationQueryRewriter(llm)
    history = [("user", "差旅报销流程是什么"), ("assistant", "需要提前审批")]

    rewritten = rewriter.rewrite("它要提前几天？", history)

    assert rewritten == "差旅报销需要提前几天审批？"
    assert llm.chat_calls == 1


def test_rewriter_falls_back_on_llm_failure() -> None:
    """LLM 异常不阻塞主链路，回退原问题。"""
    llm = FakeLLM(fail=True)
    rewriter = ConversationQueryRewriter(llm)

    assert rewriter.rewrite("还要多久？", [("user", "审批流程")]) == "还要多久？"


def test_rewriter_falls_back_on_empty_or_too_long_result() -> None:
    history = [("user", "报销流程")]

    assert ConversationQueryRewriter(FakeLLM("")).rewrite("那个呢？", history) == "那个呢？"
    long_text = "改" * 201
    assert ConversationQueryRewriter(FakeLLM(long_text)).rewrite("那个呢？", history) == (
        "那个呢？"
    )


# ------------------------------------------------------------------ 假仓库 / 假管线
class FakeRepo:
    """内存版元数据仓库：只实现会话与审计相关方法。"""

    def __init__(self) -> None:
        self.conversations: dict[str, ConversationInfo] = {}
        self.messages: dict[str, list[ConversationMessage]] = {}
        self.audits: list[dict[str, Any]] = []

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
        self.conversations[conversation_id] = ConversationInfo(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user_id,
            title=title,
        )
        self.messages[conversation_id] = []

    async def get_conversation(self, conversation_id: str) -> ConversationInfo | None:
        return self.conversations.get(conversation_id)

    async def append_conversation_messages(
        self, conversation_id: str, messages: list[ConversationMessage]
    ) -> None:
        self.messages.setdefault(conversation_id, []).extend(messages)

    async def list_conversation_messages(
        self, conversation_id: str, *, limit: int = 20
    ) -> list[ConversationMessage]:
        return list(self.messages.get(conversation_id, []))[-limit:]

    async def write_audit(self, **kwargs: Any) -> None:
        self.audits.append(kwargs)


class FakePipeline:
    """记录 query 入参，返回固定答案的假管线。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, kb: Any, question: str, **kwargs: Any) -> QueryResult:
        self.calls.append({"question": question, "kwargs": kwargs})
        return QueryResult(answer="需要提前审批", citations=[], model="fake-llm", chunks=[])

    def query_stream(self, kb: Any, question: str, **kwargs: Any) -> QueryStream:
        self.calls.append({"question": question, "kwargs": kwargs})
        return QueryStream(
            citations=[], model="fake-llm", tokens=iter(["需要", "提前审批"]), chunks=[]
        )


class HeaderAuthProvider(AuthProvider):
    """按请求头切换用户，便于构造跨用户越权场景。"""

    def authenticate(self, request: Any) -> CurrentUser:
        return CurrentUser(
            tenant_id=_TENANT,
            user_id=request.headers.get("x-test-user", "dev-user"),
        )


def _parse_sse(body: str) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        event = next(line[len("event: ") :] for line in lines if line.startswith("event: "))
        data = next(line[len("data: ") :] for line in lines if line.startswith("data: "))
        events.append((event, json.loads(data)))
    return events


def _seed_conversation(
    repo: FakeRepo, *, conversation_id: str, user_id: str, kb_id: str = _KB_ID
) -> None:
    repo.conversations[conversation_id] = ConversationInfo(
        conversation_id=conversation_id,
        tenant_id=_TENANT,
        kb_id=kb_id,
        user_id=user_id,
        title=None,
    )
    repo.messages[conversation_id] = [
        ConversationMessage(role="user", content="差旅报销流程是什么"),
        ConversationMessage(role="assistant", content="需要提前审批"),
    ]


# ------------------------------------------------------------------ 非流式会话 API
def test_chat_creates_conversation_and_returns_id() -> None:
    app = create_app()
    with TestClient(app) as client:
        repo = FakeRepo()
        pipeline = FakePipeline()
        app.state.db_repo = repo
        app.state.pipeline = pipeline

        response = client.post("/api/v1/kbs/finance/chat", json={"question": "报销流程？"})

    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    assert conversation_id
    assert pipeline.calls[0]["kwargs"]["history"] == []
    # 会话已登记，且本轮问答已落库（user + assistant 两条）
    assert repo.conversations[conversation_id].user_id == "dev-user"
    assert [m.role for m in repo.messages[conversation_id]] == ["user", "assistant"]


def test_chat_with_conversation_id_injects_history() -> None:
    app = create_app()
    with TestClient(app) as client:
        repo = FakeRepo()
        pipeline = FakePipeline()
        app.state.db_repo = repo
        app.state.pipeline = pipeline
        _seed_conversation(repo, conversation_id="c-1", user_id="dev-user")

        response = client.post(
            "/api/v1/kbs/finance/chat",
            json={"question": "要提前几天？", "conversation_id": "c-1"},
        )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == "c-1"
    history = pipeline.calls[0]["kwargs"]["history"]
    assert history == [
        ("user", "差旅报销流程是什么"),
        ("assistant", "需要提前审批"),
    ]
    # 新问答追加在历史之后
    assert [m.role for m in repo.messages["c-1"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_chat_rejects_conversation_of_another_user() -> None:
    app = create_app()
    with TestClient(app) as client:
        repo = FakeRepo()
        app.state.db_repo = repo
        app.state.pipeline = FakePipeline()
        app.state.auth_provider = HeaderAuthProvider()
        _seed_conversation(repo, conversation_id="c-other", user_id="other-user")

        response = client.post(
            "/api/v1/kbs/finance/chat",
            json={"question": "续聊", "conversation_id": "c-other"},
            headers={"x-test-user": "dev-user"},
        )

    assert response.status_code == 403
    assert response.json()["code"] == "permission_denied"


def test_chat_rejects_conversation_of_another_kb() -> None:
    app = create_app()
    with TestClient(app) as client:
        repo = FakeRepo()
        app.state.db_repo = repo
        app.state.pipeline = FakePipeline()
        _seed_conversation(
            repo, conversation_id="c-hr", user_id="dev-user", kb_id="kb-hr-001"
        )

        response = client.post(
            "/api/v1/kbs/finance/chat",
            json={"question": "续聊", "conversation_id": "c-hr"},
        )

    assert response.status_code == 403
    assert response.json()["code"] == "permission_denied"


def test_chat_with_unknown_conversation_returns_404() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.db_repo = FakeRepo()
        app.state.pipeline = FakePipeline()

        response = client.post(
            "/api/v1/kbs/finance/chat",
            json={"question": "续聊", "conversation_id": "missing"},
        )

    assert response.status_code == 404
    assert response.json()["code"] == "conversation_not_found"


# ------------------------------------------------------------------ 流式会话 API
def test_chat_stream_done_carries_conversation_id_and_records_turn() -> None:
    app = create_app()
    with TestClient(app) as client:
        repo = FakeRepo()
        pipeline = FakePipeline()
        app.state.db_repo = repo
        app.state.pipeline = pipeline

        response = client.post(
            "/api/v1/kbs/finance/chat:stream", json={"question": "报销流程？"}
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    # v2.1（清单 P1-03）：done 前新增最终引用校验事件 validation
    # v2.2：validation 前新增 answer 事件（用清理后的最终正文替换前端 token）
    assert [event for event, _ in events] == [
        "citations",
        "token",
        "token",
        "answer",
        "validation",
        "done",
    ]
    done = events[-1][1]
    conversation_id = done["conversation_id"]
    assert conversation_id
    assert pipeline.calls[0]["kwargs"]["history"] == []
    # 流式答案完整拼接后落库，与一次性路径保持一致
    assert [m.role for m in repo.messages[conversation_id]] == ["user", "assistant"]
    assert repo.messages[conversation_id][1].content == "需要提前审批"


def test_chat_stream_continues_existing_conversation() -> None:
    app = create_app()
    with TestClient(app) as client:
        repo = FakeRepo()
        pipeline = FakePipeline()
        app.state.db_repo = repo
        app.state.pipeline = pipeline
        _seed_conversation(repo, conversation_id="c-1", user_id="dev-user")

        response = client.post(
            "/api/v1/kbs/finance/chat:stream",
            json={"question": "还有呢？", "conversation_id": "c-1"},
        )

    assert response.status_code == 200
    history = pipeline.calls[0]["kwargs"]["history"]
    assert history == [
        ("user", "差旅报销流程是什么"),
        ("assistant", "需要提前审批"),
    ]
    assert _parse_sse(response.text)[-1][1]["conversation_id"] == "c-1"


# ------------------------------------------------------------------ 会话历史查询
def test_conversation_history_returns_messages_in_order() -> None:
    app = create_app()
    with TestClient(app) as client:
        repo = FakeRepo()
        app.state.db_repo = repo
        _seed_conversation(repo, conversation_id="c-1", user_id="dev-user")

        response = client.get("/api/v1/kbs/finance/conversations/c-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == "c-1"
    assert payload["kb_id"] == _KB_ID
    assert [item["role"] for item in payload["messages"]] == ["user", "assistant"]
    assert payload["messages"][0]["content"] == "差旅报销流程是什么"


def test_conversation_history_rejects_other_user_and_missing() -> None:
    app = create_app()
    with TestClient(app) as client:
        repo = FakeRepo()
        app.state.db_repo = repo
        app.state.auth_provider = HeaderAuthProvider()
        _seed_conversation(repo, conversation_id="c-other", user_id="other-user")

        forbidden = client.get(
            "/api/v1/kbs/finance/conversations/c-other", headers={"x-test-user": "dev-user"}
        )
        missing = client.get("/api/v1/kbs/finance/conversations/missing")

    assert forbidden.status_code == 403
    assert forbidden.json()["code"] == "permission_denied"
    assert missing.status_code == 404
    assert missing.json()["code"] == "conversation_not_found"
