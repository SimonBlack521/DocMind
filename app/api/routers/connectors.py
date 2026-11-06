"""连接器管理（v2.0）：查看已配置知识源、手动触发同步。

配置本身由 KB YAML 维护（connections 属于基础设施声明，不适合开放给普通
管理接口热改），本路由只提供只读视图与"触发一次同步"的运维入口。
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import (
    ContributorUserDep,
    DBRepoDep,
    KBConfigDep,
    ObjectStoreDep,
    PipelineDep,
)
from app.api.schemas import (
    ConnectorSummary,
    ConnectorSyncResponse,
    ConnectorWebhookPayload,
    ConnectorWebhookResponse,
)
from app.core.errors import KBValidationError
from app.core.logging import get_logger
from app.orchestration.deletion import DocumentDeletionService
from app.tasks.connectors import sync_connector_task

logger = get_logger(__name__)
router = APIRouter(prefix="/kbs/{kb_key}/connectors", tags=["connectors"])


@router.get("", response_model=list[ConnectorSummary])
async def list_connectors(kb: KBConfigDep, user: ContributorUserDep) -> list[ConnectorSummary]:
    """列出知识库已配置的连接器（需要写入权限，避免泄露内部路径）。"""
    return [
        ConnectorSummary(
            connector_id=item.connector_id,
            type=item.type,
            enabled=item.enabled,
            path=item.path,
            owner=item.owner,
            authority_level=item.authority_level,
        )
        for item in kb.connectors
    ]


@router.post(
    "/{connector_id}/sync",
    response_model=ConnectorSyncResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def sync_connector(
    connector_id: str, kb: KBConfigDep, user: ContributorUserDep
) -> ConnectorSyncResponse:
    """异步触发一次连接器同步；读取/解析/入库全部在 worker 内完成。"""
    if not any(item.connector_id == connector_id for item in kb.connectors):
        raise KBValidationError(
            f"知识库 {kb.kb_code} 未配置连接器 {connector_id}",
            details={"kb_code": kb.kb_code, "connector_id": connector_id},
        )
    task_id = uuid4().hex
    await run_in_threadpool(
        sync_connector_task.apply_async,
        args=[kb.kb_code, connector_id],
        task_id=task_id,
    )
    return ConnectorSyncResponse(
        task_id=task_id,
        connector_id=connector_id,
        status="queued",
    )


@router.post(
    "/{connector_id}/webhook",
    response_model=ConnectorWebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def connector_webhook(
    connector_id: str,
    payload: ConnectorWebhookPayload,
    kb: KBConfigDep,
    user: ContributorUserDep,
    pipeline: PipelineDep,
    repo: DBRepoDep,
    object_store: ObjectStoreDep,
) -> ConnectorWebhookResponse:
    """源系统 Webhook 推送入口：上报变更/删除的 source_uri。

    - deleted_uris：直接走统一删除状态机（索引 + 原始文件 + 元数据），
      不必等待下一次全量同步，源端删除即时生效；
    - source_uris：变更通过触发一次全量幂等同步收口（连接器协议本身就是
      全量扫描 + 内容指纹幂等，Webhook 只负责"提前唤醒"，避免为它单独
      维护一套增量入库路径）。
    """
    if not any(item.connector_id == connector_id for item in kb.connectors):
        raise KBValidationError(
            f"知识库 {kb.kb_code} 未配置连接器 {connector_id}",
            details={"kb_code": kb.kb_code, "connector_id": connector_id},
        )

    # 删除事件即时处理：按 source_uri 找到已入库文档，走统一删除状态机
    deleted = 0
    if payload.deleted_uris:
        known = await repo.list_documents_by_source(
            kb_id=kb.kb_id, tenant_id=kb.tenant_id, source_id=connector_id
        )
        deleted_set = set(payload.deleted_uris)
        service = DocumentDeletionService(
            pipeline=pipeline, repo=repo, object_store=object_store
        )
        for doc_id, source_uri in known:
            if source_uri not in deleted_set:
                continue
            try:
                await service.delete(kb, doc_id, user_id=user.user_id)
                deleted += 1
            except Exception as exc:
                # 单篇失败留在 DELETING，由 rag.resume_pending_deletions 重试
                logger.warning(
                    "Webhook 删除失败 connector=%s doc_id=%s uri=%s err=%s",
                    connector_id,
                    doc_id,
                    source_uri,
                    exc,
                )

    # 变更事件：唤醒一次全量幂等同步（内容未变的文档会被治理层判重跳过）
    task_id: str | None = None
    if payload.source_uris:
        task_id = uuid4().hex
        await run_in_threadpool(
            sync_connector_task.apply_async,
            args=[kb.kb_code, connector_id],
            task_id=task_id,
        )

    return ConnectorWebhookResponse(
        connector_id=connector_id,
        task_id=task_id,
        deleted=deleted,
        status="accepted",
    )
