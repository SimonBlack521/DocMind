"""Transactional Outbox 投递器（v1.4）。

背景：API 先写 PG 任务行、再向 broker 发消息，两步之间的崩溃会让任务永久
pending。Outbox 把"写任务"和"待投递事件"放进同一事务，再由本模块周期补投，
把"至少一次投递"落到数据库事实层面。

投递语义：
- 事件用 `FOR UPDATE SKIP LOCKED` 原子领取（pending → dispatching），单次默认
  100 条；多 dispatcher 并发不会重复投递同一条事件；
- 投递成功 → dispatched；失败退回 pending 等下轮重投，达到上限置 failed 供人工重放；
- dispatcher 崩溃留下的 dispatching 事件按 lease 超时自动回收；
- 重复投递由任务侧幂等兜底（同一 doc_id 覆盖写，不产生重复索引）。
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import OUTBOX_EVENT_INGEST
from app.tasks.celery_app import celery_app
from app.tasks.ingest import _get_resources, ingest_document_task

logger = get_logger(__name__)

# 事件类型 → 投递函数；新增事件类型（如 Connector 同步）在这里登记即可
_DISPATCHERS: dict[str, Any] = {
    OUTBOX_EVENT_INGEST: ingest_document_task,
}

# 领取租约：dispatcher 崩溃后，事件在该时长后重新变为 pending 供其他实例领取
_CLAIM_LEASE_SECONDS = 300


@celery_app.task(name="rag.dispatch_outbox")
def dispatch_outbox(limit: int = 100) -> dict[str, int]:
    """原子领取并补投 pending 事件；返回扫描/成功/失败计数便于观测。"""
    resources = _get_resources()
    try:
        events = resources.run(
            resources.repo.claim_pending_outbox(
                limit=limit, lease_seconds=_CLAIM_LEASE_SECONDS
            )
        )
    except Exception:
        logger.exception("Outbox 扫描失败，等待下个周期重试")
        return {"scanned": 0, "dispatched": 0, "failed": 0}

    dispatched = 0
    failed = 0
    for event in events:
        task = _DISPATCHERS.get(event.event_type)
        if task is None:
            logger.warning(
                "Outbox 事件类型未知，跳过 event=%s type=%s",
                event.event_id,
                event.event_type,
            )
            resources.run(
                resources.repo.mark_outbox_failed(
                    event.event_id, error=f"unknown event type: {event.event_type}"
                )
            )
            failed += 1
            continue

        kwargs = {key: value for key, value in event.payload.items() if key != "task_id"}
        try:
            task.apply_async(kwargs=kwargs, task_id=event.event_id)
        except (AppError, RuntimeError, ValueError) as exc:
            message = getattr(exc, "message", str(exc))
            logger.warning("Outbox 投递失败 event=%s：%s", event.event_id, message)
            resources.run(
                resources.repo.mark_outbox_failed(event.event_id, error=message)
            )
            failed += 1
        else:
            resources.run(resources.repo.mark_outbox_dispatched(event.event_id))
            dispatched += 1

    logger.info(
        "Outbox 投递完成 scanned=%d dispatched=%d failed=%d",
        len(events),
        dispatched,
        failed,
    )
    return {"scanned": len(events), "dispatched": dispatched, "failed": failed}
