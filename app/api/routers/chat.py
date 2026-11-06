"""知识库问答：混合检索 + 精排 + 生成，支持一次性返回与 SSE 流式两种形态。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.api.deps import CurrentUserDep, DBRepoDep, KBConfigDep, PipelineDep, rate_limit_dep
from app.api.schemas import (
    AnswerVerificationModel,
    ChatRequest,
    ChatResponse,
    CitationModel,
    ConversationHistoryResponse,
    ConversationMessageModel,
    EvidenceModel,
)
from app.core.errors import (
    AppError,
    ConversationNotFoundError,
    PermissionDeniedError,
)
from app.core.metrics import metrics
from app.core.settings import get_settings
from app.db.models import (
    AUDIT_ACTION_CHAT_QUERY,
    CONVERSATION_ROLE_ASSISTANT,
    CONVERSATION_ROLE_USER,
)
from app.db.repo import ConversationInfo, ConversationMessage
from app.generation.chain import UNSUPPORTED_ANSWER
from app.generation.citation_validator import (
    ANSWER_REJECTED,
    clean_answer_refs,
    validate_citations,
    verify_answer_support,
)

router = APIRouter(prefix="/kbs", tags=["chat"])

_settings = get_settings()
# 模块级构建限流依赖（limit 取自配置），在路由装饰器的 dependencies 中复用
_CHAT_RATE_LIMIT = rate_limit_dep("chat", limit=_settings.rate_limit_chat_rpm)


def _sse(event: str, data: Any) -> str:
    """SSE 帧：data 统一 JSON 编码，前端按 event 字段分发。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _check_owner(conversation: ConversationInfo, kb: KBConfigDep, user: CurrentUserDep) -> None:
    """会话归属校验：租户 / 知识库 / 用户三者必须一致，防止拿到 id 越权续聊。"""
    if (
        conversation.tenant_id != kb.tenant_id
        or conversation.kb_id != kb.kb_id
        or conversation.user_id != user.user_id
    ):
        raise PermissionDeniedError(
            "当前用户无权访问该会话",
            details={"conversation_id": conversation.conversation_id},
        )


async def _ensure_conversation(
    repo: DBRepoDep,
    kb: KBConfigDep,
    user: CurrentUserDep,
    conversation_id: str | None,
) -> tuple[str, list[tuple[str, str]]]:
    """解析会话：带 id 则校验归属并加载历史；不带 id 则新建，历史为空。

    回归兼容：未传 conversation_id 的单轮调用会自动分配会话 id 并写入问答消息，
    前端可把它带回下一轮，从而平滑升级到多轮。
    """
    if conversation_id:
        conversation = await repo.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(
                "会话不存在", details={"conversation_id": conversation_id}
            )
        _check_owner(conversation, kb, user)
        messages = await repo.list_conversation_messages(
            conversation_id, limit=_settings.conversation_history_limit
        )
        return conversation_id, [(item.role, item.content) for item in messages]

    new_id = uuid4().hex
    await repo.create_conversation(
        new_id,
        tenant_id=kb.tenant_id,
        kb_id=kb.kb_id,
        user_id=user.user_id,
        title=None,
    )
    return new_id, []


async def _record_turn(
    repo: DBRepoDep, conversation_id: str, question: str, answer: str
) -> None:
    """落一轮问答：用户问题与助手答案同一事务写入。"""
    await repo.append_conversation_messages(
        conversation_id,
        [
            ConversationMessage(role=CONVERSATION_ROLE_USER, content=question),
            ConversationMessage(role=CONVERSATION_ROLE_ASSISTANT, content=answer),
        ],
    )


@router.post(
    "/{kb_key}/chat",
    response_model=ChatResponse,
    summary="知识库问答（非流式）",
    dependencies=[_CHAT_RATE_LIMIT],
)
async def chat(
    payload: ChatRequest,
    kb: KBConfigDep,
    user: CurrentUserDep,
    pipeline: PipelineDep,
    repo: DBRepoDep,
) -> ChatResponse:
    """非流式问答：等待完整答案后一次性返回。

    关键设计：
    - allowed_groups 透传给检索层做文档级 ACL 过滤（用户只见自己组可见的片段）；
    - excluded_doc_ids 是"检索不可见集合"（已被新版本替代 + DELETING 删除中），
      检索时下推排除，避免答案引用过期内容或已进入删除流程的残留索引；
    - 全链路完成后写审计日志。
    """
    # 检索不可见集合：已被替代 + 删除中（DELETING）的文档，一次查询权威裁决
    excluded = await repo.list_hidden_doc_ids(kb_id=kb.kb_id, tenant_id=kb.tenant_id)
    # 解析或新建会话；history 只影响检索 query 的追问改写
    conversation_id, history = await _ensure_conversation(
        repo, kb, user, payload.conversation_id
    )
    # 问答链路是同步阻塞 IO（检索与生成客户端均为同步），放到线程池执行
    result = await run_in_threadpool(
        pipeline.query,
        kb,
        payload.question,
        top_k=payload.top_k,
        use_rewrite=payload.rewrite,
        use_rerank=payload.rerank,
        # ACL 过滤：把用户的组列表传下去，检索时过滤掉无权访问的文档片段
        allowed_groups=user.allowed_groups,
        excluded_doc_ids=excluded,
        history=history,
        # v2.0 查询路由：auto=结构化可用时按问题语义分流，否则纯文档检索
        mode=payload.mode,
    )
    # 落一轮问答（用户问 + 助手答），供下一轮追问改写使用
    await _record_turn(repo, conversation_id, payload.question, result.answer)
    await repo.write_audit(
        tenant_id=kb.tenant_id,
        kb_id=kb.kb_id,
        user_id=user.user_id,
        action=AUDIT_ACTION_CHAT_QUERY,
        detail={
            "question": payload.question[:200],
            "retrieved_count": len(result.chunks),
            "conversation_id": conversation_id,
        },
    )
    return ChatResponse(
        kb_id=kb.kb_id,
        answer=result.answer,
        model=result.model,
        retrieved_count=len(result.chunks),
        citations=[CitationModel.from_citation(citation) for citation in result.citations],
        conversation_id=conversation_id,
        route=result.route,
        sql=result.sql,
        # v2.1 统一证据与降级标记：SQL/API 来源可追溯；降级时前端显式提示
        evidence=[EvidenceModel.from_evidence(item) for item in result.evidence],
        degraded=result.degraded,
        degrade_reason=result.degrade_reason,
        # v2.2 独立答案验证状态：与非流式同一口径返回给前端标注
        verification=(
            AnswerVerificationModel(**result.verification.to_dict())
            if result.verification is not None
            else None
        ),
    )


@router.post(
    "/{kb_key}/chat:stream",
    summary="知识库问答（SSE 流式）",
    dependencies=[_CHAT_RATE_LIMIT],
)
async def chat_stream(
    payload: ChatRequest,
    kb: KBConfigDep,
    user: CurrentUserDep,
    pipeline: PipelineDep,
    repo: DBRepoDep,
) -> StreamingResponse:
    """SSE 流式问答：先推引用，再逐 token 推答案，最后 done 收尾。

    分两段的原因：检索是同步阻塞 IO（先在线程池完成拿到 citations）；
    token 生成是惰性迭代器，只有在 events() 被 StreamingResponse 消费时才真正请求 LLM，
    避免端点函数里提前触发昂贵的生成调用。
    """
    # 检索是同步阻塞 IO，先在线程池完成；token 生成是惰性迭代器，推送阶段才真正请求 LLM
    excluded = await repo.list_hidden_doc_ids(kb_id=kb.kb_id, tenant_id=kb.tenant_id)
    conversation_id, history = await _ensure_conversation(
        repo, kb, user, payload.conversation_id
    )
    stream = await run_in_threadpool(
        pipeline.query_stream,
        kb,
        payload.question,
        top_k=payload.top_k,
        use_rewrite=payload.rewrite,
        use_rerank=payload.rerank,
        # ACL 过滤：同非流式端点，检索层据此过滤无权访问的文档片段
        allowed_groups=user.allowed_groups,
        excluded_doc_ids=excluded,
        history=history,
        mode=payload.mode,
    )

    async def events() -> AsyncIterator[str]:
        """SSE 事件流：citations → token（多次）→ answer → validation → done；失败以 error 收尾。

        answer 事件（v2.2）：token 是增量推送、无法撤回，因此这里再发一份"最终正文"
        （清理失效 [n]、必要时替换为拒答文案），前端按 replace 直接覆盖展示文本。
        validation 事件（v2.1）：流式输出无法像非流式那样"先校验再返回"，因此
        在 token 推完后对完整答案补一次引用校验，把最终生效的引用列表与剔除数
        告知前端——流式与非流式的答案质量口径由此对齐。
        """
        yield _sse(
            "citations",
            [CitationModel.from_citation(c).model_dump(mode="json") for c in stream.citations],
        )
        parts: list[str] = []
        try:
            # 同步 token 迭代器包成异步迭代器，推送期间不阻塞事件循环
            async for token in iterate_in_threadpool(stream.tokens):
                parts.append(token)
                yield _sse("token", {"content": token})
        except AppError as exc:
            # 流式响应无法中途改 HTTP 状态码，业务错误只能通过 error 事件告知前端
            yield _sse("error", {"code": exc.code, "message": exc.message})
            return
        # 流式答案的最终引用校验：与非流式同一套 validate_citations 口径
        validation = validate_citations("".join(parts), stream.citations)
        if validation.out_of_range_refs or validation.unverified_citations:
            metrics.inc(
                "citation_issues_total",
                validation.out_of_range_refs + validation.unverified_citations,
            )
        final_citations = [
            CitationModel.from_citation(c).model_dump(mode="json")
            for c in validation.cleaned_citations
        ]
        # 最终正文：把已失效的 [n] 清掉，并基于实际送入模型的证据做独立答案验证。
        # 流式已推送的 token 无法撤回，因此用 answer 事件让前端替换为清理后的正文；
        # 落会话记录的也必须是同一份最终文本（否则历史里会残留无效引用标记）。
        final_answer = clean_answer_refs("".join(parts), validation.cleaned_citations)
        verification = verify_answer_support(
            final_answer,
            validation.cleaned_citations,
            stream.chunks,
            min_coverage=_settings.answer_min_citation_coverage,
            fact_conflicts=validation.fact_conflicts,
        )
        metrics.inc("answer_verification_total", status=verification.status)
        if verification.status == ANSWER_REJECTED:
            final_answer = UNSUPPORTED_ANSWER
            final_citations = []
        yield _sse("answer", {"content": final_answer, "replace": True})
        yield _sse(
            "validation",
            {
                "citations": final_citations,
                "total_answer_refs": validation.total_answer_refs,
                "unverified_citations": validation.unverified_citations,
                "out_of_range_refs": validation.out_of_range_refs,
                "fact_conflicts": validation.fact_conflicts,
                "verification": verification.to_dict(),
            },
        )
        yield _sse(
            "done",
            {
                "model": stream.model,
                "retrieved_count": len(stream.chunks),
                "conversation_id": conversation_id,
                "route": stream.route,
                "sql": stream.sql,
                "degraded": stream.degraded,
                "degrade_reason": stream.degrade_reason,
                "evidence": [
                    EvidenceModel.from_evidence(item).model_dump(mode="json")
                    for item in stream.evidence
                ],
            },
        )
        # 流式答案完整拼好后落会话；客户端中途断开（生成器被取消）则不落。
        # 必须落"最终清理后的正文"：与非流式响应、前端展示保持同一文本，
        # 否则历史会话里会残留已被判无效的 [n] 标记。
        await _record_turn(repo, conversation_id, payload.question, final_answer)
        # 问答完成后写审计；客户端中途断开（生成器被取消）则不计入
        await repo.write_audit(
            tenant_id=kb.tenant_id,
            kb_id=kb.kb_id,
            user_id=user.user_id,
            action=AUDIT_ACTION_CHAT_QUERY,
            detail={
                "question": payload.question[:200],
                "retrieved_count": len(stream.chunks),
                "stream": True,
                "conversation_id": conversation_id,
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # 禁止中间层缓存/缓冲，保证 token 即时下发
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{kb_key}/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
    summary="会话历史",
)
async def get_conversation_history(
    conversation_id: str,
    kb: KBConfigDep,
    user: CurrentUserDep,
    repo: DBRepoDep,
) -> ConversationHistoryResponse:
    """读取本会话的历史消息（仅创建者本人可读，跨用户/跨库一律 403）。"""
    conversation = await repo.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFoundError(
            "会话不存在", details={"conversation_id": conversation_id}
        )
    _check_owner(conversation, kb, user)
    messages = await repo.list_conversation_messages(conversation_id, limit=100)
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        kb_id=conversation.kb_id,
        title=conversation.title,
        messages=[
            ConversationMessageModel(role=item.role, content=item.content)
            for item in messages
        ],
    )


@router.delete(
    "/{kb_key}/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话",
)
async def delete_conversation(
    conversation_id: str,
    kb: KBConfigDep,
    user: CurrentUserDep,
    repo: DBRepoDep,
) -> None:
    """删除会话及全部消息（仅创建者本人可删，跨用户/跨库一律 403）。

    清单第四优先级第 20 项：会话过期和删除机制——用户侧主动删除入口；
    到期自动清理由 rag.purge_expired_conversations 周期任务负责。
    """
    conversation = await repo.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFoundError(
            "会话不存在", details={"conversation_id": conversation_id}
        )
    _check_owner(conversation, kb, user)
    await repo.delete_conversation(conversation_id)
