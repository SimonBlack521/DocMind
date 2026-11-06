"""连接器同步任务（v2.0）。

同步只做"发现 + 投递 + 删除收口"：把连接器产出的文档写入对象存储暂存，再为
每篇文档创建"任务行 + Outbox 事件"并投递现有入库任务；同步结束后对比
"上次已入库的 source_uri 集合"与"本次扫描到的集合"，源端缺失的文档走统一
删除服务（索引 + 原始文件 + 元数据）。这样同步不复制第二套入库/删除逻辑，
治理裁决、质量门禁、增量 Diff、双写索引全部复用，安全与质量约束不打折。

Checkpoint 语义：不依赖 mtime 水位（文件时间戳可能被复制/恢复导致漏检），
每次全量扫描目录，以 source_uri 稳定映射 + 内容指纹做幂等；删除差异用
PG 中已入库的 source_uri 集合做差集，因此源端删除不会留下可检索的旧内容。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import redis

from app.connectors.registry import build_connector
from app.core.errors import AppError
from app.core.logging import get_logger
from app.core.settings import get_settings
from app.db.models import OUTBOX_EVENT_INGEST, TASK_STATUS_PENDING
from app.orchestration.deletion import DocumentDeletionService
from app.tasks.celery_app import celery_app
from app.tasks.ingest import _get_resources, ingest_document_task

logger = get_logger(__name__)


@celery_app.task(name="rag.sync_connector", bind=True)
def sync_connector_task(self: Any, kb_key: str, connector_id: str) -> dict[str, Any]:
    """拉取一个连接器的全部（或增量）文档并交给入库任务。

    幂等：source_uri 由连接器稳定生成，重复同步会落到同一 doc_id，
    governance 判定"内容未变"即复用上一版向量，不产生重复文档。
    """
    resources = _get_resources()
    kb = resources.resolve_kb(kb_key)
    config = next(
        (item for item in kb.connectors if item.connector_id == connector_id), None
    )
    if config is None:
        raise AppError(f"知识库 {kb_key} 未配置连接器 {connector_id}")
    if not config.enabled:
        return {"connector_id": connector_id, "queued": 0, "skipped": "disabled"}

    connector = build_connector(config)
    known = resources.run(
        resources.repo.list_documents_by_source(
            kb_id=kb.kb_id, tenant_id=kb.tenant_id, source_id=connector_id
        )
    )
    fetched_uris: set[str] = set()
    queued = 0
    skipped = 0
    for document in connector.fetch():
        fetched_uris.add(document.source_uri)
        task_id = uuid4().hex
        staging_key = f"staging/{task_id}"
        resources.object_store.put_bytes(
            staging_key, document.content, content_type="application/octet-stream"
        )
        task_kwargs: dict[str, Any] = {
            "kb_key": kb.kb_code,
            "payload": None,
            "staging_key": staging_key,
            "filename": document.filename,
            "title": document.title or document.filename,
            "source_uri": document.source_uri,
            "acl_groups": None,
            "source_id": connector_id,
            "source_type": config.source_type,
            "owner": config.owner,
            "source_version": document.source_version,
            "authority_level": config.authority_level,
            "valid_from": None,
            "valid_to": None,
        }
        try:
            resources.run(
                resources.repo.create_task_with_outbox(
                    task_id,
                    kb_id=kb.kb_id,
                    tenant_id=kb.tenant_id,
                    status=TASK_STATUS_PENDING,
                    event_type=OUTBOX_EVENT_INGEST,
                    payload=task_kwargs,
                )
            )
            ingest_document_task.apply_async(kwargs=task_kwargs, task_id=task_id)
            resources.run(resources.repo.mark_outbox_dispatched(task_id))
            queued += 1
        except AppError as exc:
            # 单篇投递失败不阻断其余文档：Outbox 保持 pending，由 dispatcher 补投
            logger.warning(
                "连接器文档投递失败 connector=%s uri=%s err=%s",
                connector_id,
                document.source_uri,
                exc.message,
            )
            skipped += 1

    # 源端删除同步：已入库但本次扫描不到的文档，走统一删除状态机
    deletion_service = DocumentDeletionService(
        pipeline=resources.pipeline,
        repo=resources.repo,
        object_store=resources.object_store,
    )
    deleted = 0
    for doc_id, source_uri in known:
        if source_uri in fetched_uris:
            continue
        try:
            resources.run(deletion_service.delete(kb, doc_id))
            deleted += 1
        except Exception as exc:
            # 单篇删除失败留在 DELETING，由 rag.resume_pending_deletions 重试
            logger.warning(
                "连接器删除同步失败 connector=%s doc_id=%s uri=%s err=%s",
                connector_id,
                doc_id,
                source_uri,
                exc,
            )

    logger.info(
        "连接器同步完成 kb=%s connector=%s queued=%d skipped=%d deleted=%d",
        kb_key,
        connector_id,
        queued,
        skipped,
        deleted,
    )
    return {
        "connector_id": connector_id,
        "queued": queued,
        "skipped": skipped,
        "deleted": deleted,
    }


@celery_app.task(name="rag.sync_due_connectors", bind=True)
def sync_due_connectors_task(self: Any) -> dict[str, Any]:
    """周期扫描全部知识库的连接器配置，对到达同步间隔的连接器派发同步任务。

    到期判定用 Redis 键 `connector:due:{kb_id}:{connector_id}` 做 SET NX EX：
    设置成功 = 已到期（键同时充当"间隔期内不再重复派发"的令牌），多 beat /
    多 worker 实例并发也只会派发一次；Redis 不可用时本轮整体跳过（下次 beat
    周期再试），不让调度器因为缓存故障瘫痪。
    sync_interval_minutes=0 的连接器不参与自动调度（仅手动 /sync 或 Webhook）。
    """
    resources = _get_resources()
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    due = 0
    scanned = 0
    try:
        for kb in resources.kb_registry.list_enabled():
            for cfg in kb.connectors:
                if not cfg.enabled or cfg.sync_interval_minutes <= 0:
                    continue
                scanned += 1
                key = f"connector:due:{kb.kb_id}:{cfg.connector_id}"
                try:
                    acquired = client.set(key, "1", nx=True, ex=cfg.sync_interval_minutes * 60)
                except redis.RedisError as exc:
                    logger.warning("连接器到期判定失败（本轮跳过）：%s", exc)
                    return {"scanned": scanned, "due": due, "skipped": "redis_error"}
                if not acquired:
                    continue
                sync_connector_task.apply_async(args=[kb.kb_code, cfg.connector_id])
                due += 1
                logger.info(
                    "连接器定时同步已派发 kb=%s connector=%s interval=%dm",
                    kb.kb_code,
                    cfg.connector_id,
                    cfg.sync_interval_minutes,
                )
    finally:
        client.close()
    return {"scanned": scanned, "due": due}
