"""异步入库任务状态查询与死信（DLQ）管理。状态的唯一权威来源是 PG 的 ingest_tasks 表。"""

from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Query, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import (
    ContributorUserDep,
    CurrentUserDep,
    DBRepoDep,
    KBConfigDep,
    SettingsDep,
    has_kb_contributor_role,
)
from app.api.schemas import (
    DeadLetterListResponse,
    DeadLetterReplayResponse,
    DeadLetterSummary,
    TaskStatusResponse,
)
from app.core.errors import PermissionDeniedError, TaskNotFoundError
from app.core.logging import get_logger
from app.db.models import AUDIT_ACTION_DEAD_LETTER_REPLAY, TASK_STATUS_PENDING
from app.tasks.ingest import ingest_document_task

logger = get_logger(__name__)
router = APIRouter(tags=["tasks"])


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="查询异步入库任务状态（pending/running/success/failed）",
)
async def get_task_status(
    task_id: str, user: CurrentUserDep, repo: DBRepoDep, settings: SettingsDep
) -> TaskStatusResponse:
    """查询任务状态；不存在返回 404，跨租户访问返回 403。"""
    info = await repo.get_task(task_id)
    if info is None:
        raise TaskNotFoundError(f"任务不存在：{task_id}", details={"task_id": task_id})
    # 租户隔离：任务记录带 tenant_id，其他租户一律无权查看
    if info.tenant_id != user.tenant_id:
        raise PermissionDeniedError(
            "当前用户无权查看该任务", details={"task_id": task_id}
        )
    # 知识库隔离：任务里带有 kb_id，用户没有该库授权时同样不允许查看任务详情
    if user.allowed_kb_ids is not None and info.kb_id not in user.allowed_kb_ids:
        raise PermissionDeniedError(
            "当前用户无权查看该任务", details={"task_id": task_id}
        )
    # 属主校验：他人创建的任务只有贡献者及以上角色可查看
    # （任务详情含 doc_id/error 等内部信息，普通用户只能看自己的任务）
    if (
        info.created_by is not None
        and info.created_by != user.user_id
        and not has_kb_contributor_role(user, settings)
    ):
        raise PermissionDeniedError(
            "当前用户无权查看该任务", details={"task_id": task_id}
        )
    return TaskStatusResponse(
        task_id=info.task_id,
        status=info.status,
        kb_id=info.kb_id,
        doc_id=info.doc_id,
        error=info.error,
    )


@router.get(
    "/kbs/{kb_key}/dead-letters",
    response_model=DeadLetterListResponse,
    summary="分页列出知识库的入库死信（重试耗尽仍失败的任务）",
)
async def list_dead_letters(
    kb: KBConfigDep,
    repo: DBRepoDep,
    user: ContributorUserDep,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DeadLetterListResponse:
    """列出死信；贡献者及以上角色可见，payload 供排查与重放参考。"""
    items, total = await repo.list_dead_letters(
        kb_id=kb.kb_id, offset=(page - 1) * size, limit=size
    )
    return DeadLetterListResponse(
        total=total,
        page=page,
        size=size,
        items=[
            DeadLetterSummary(
                event_id=item.event_id,
                kb_id=item.kb_id,
                doc_id=item.doc_id,
                error=item.error,
                created_at=item.created_at,
                payload=item.payload,
            )
            for item in items
        ],
    )


@router.post(
    "/kbs/{kb_key}/dead-letters/{event_id}/replay",
    response_model=DeadLetterReplayResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="重放死信：按原任务参数重新投递异步入库",
)
async def replay_dead_letter(
    event_id: str,
    kb: KBConfigDep,
    repo: DBRepoDep,
    user: ContributorUserDep,
) -> DeadLetterReplayResponse:
    """重放一条死信：生成新任务行 + Outbox 事件并投递，成功后移除死信。

    新任务使用新的 task_id（不覆盖原失败任务的排查现场）；若原始暂存对象
    已被清理，重放任务会再次失败并重新落死信，此时应重新上传原文。
    """
    info = await repo.get_dead_letter(event_id)
    # 越库/越租户一律按 404 处理，不暴露死信存在性
    if info is None or info.kb_id != kb.kb_id or info.tenant_id != kb.tenant_id:
        raise TaskNotFoundError(
            f"死信不存在：{event_id}", details={"event_id": event_id}
        )

    task_kwargs = dict(info.payload)
    # 以当前库为准，防止 payload 里的 kb_key 与路径不一致
    task_kwargs["kb_key"] = kb.kb_code
    new_task_id = uuid4().hex
    await repo.create_task_with_outbox(
        new_task_id,
        kb_id=kb.kb_id,
        tenant_id=kb.tenant_id,
        created_by=user.user_id,
        status=TASK_STATUS_PENDING,
        payload=task_kwargs,
    )
    try:
        # 线程池内投递：即便误开 eager 模式，任务也不会嵌进应用事件循环
        await run_in_threadpool(
            ingest_document_task.apply_async, kwargs=task_kwargs, task_id=new_task_id
        )
    except Exception as exc:
        # 即时投递失败不动死信（Outbox 仍为 pending，dispatcher 会补投）
        await repo.mark_outbox_failed(new_task_id, error=str(exc))
        logger.warning("死信重放即时投递失败，等待 Outbox 补投 event=%s", event_id)
        raise
    await repo.mark_outbox_dispatched(new_task_id)
    await repo.delete_dead_letter(event_id)
    await repo.write_audit(
        tenant_id=kb.tenant_id,
        kb_id=kb.kb_id,
        user_id=user.user_id,
        action=AUDIT_ACTION_DEAD_LETTER_REPLAY,
        doc_id=info.doc_id,
        detail={"event_id": event_id, "new_task_id": new_task_id},
    )
    return DeadLetterReplayResponse(
        event_id=event_id, task_id=new_task_id, status=TASK_STATUS_PENDING
    )
