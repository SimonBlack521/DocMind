"""后台维护任务：保留期清理 + 索引对账（reconciler）。

保留期清理：按文档采集时间删除超期文档，满足数据最小化/留存期限要求。

索引对账（rag.reconcile_index）：事件驱动的增量索引可能因消息丢失、消费乱序、
补偿删除失败产生数据污染（孤儿向量：chunk 已删但索引还在；缺失：该写的没写上）。
对账任务以 PG 中 ready 文档的 chunk_manifest 为"应该存在"的权威清单，
周期性与 Milvus / ES 实际内容核对：
- 孤儿：索引里有、清单没有 → 按 id 精确删除；
- 缺失：清单有、索引没有 → 从对象存储归档取原文重灌该文档（版本不变，幂等）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from app.core.logging import get_logger
from app.core.settings import get_settings
from app.domain.kb_config import KBConfig
from app.orchestration.deletion import DocumentDeletionService
from app.tasks.celery_app import celery_app
from app.tasks.ingest import _get_resources, _WorkerResources

logger = get_logger(__name__)


@celery_app.task(bind=True, name="rag.purge_expired_documents")
def purge_expired_documents_task(
    self, *, retention_days: int | None = None, limit: int = 500
) -> dict[str, object]:
    """扫描所有启用知识库，删除采集时间早于 cutoff 的文档。

    返回统计值便于定时任务日志/告警核对；retention_days<=0 时直接跳过。
    """
    settings = get_settings()
    days = settings.document_retention_days if retention_days is None else retention_days
    if days <= 0:
        return {"deleted": 0, "skipped": True, "retention_days": days}

    resources = _get_resources()
    cutoff = datetime.now(UTC) - timedelta(days=days)
    deleted = 0
    for kb in resources.kb_registry.list_enabled():
        doc_ids = resources.run(
            resources.repo.list_expired_document_ids(
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                cutoff=cutoff,
                limit=limit,
            )
        )
        # 与 API 单篇/批量删除共用同一服务：索引、原始文件、元数据按状态机推进，
        # 失败保留 DELETING，由 resume_pending_deletions 继续重试
        service = DocumentDeletionService(
            pipeline=resources.pipeline,
            repo=resources.repo,
            object_store=resources.object_store,
        )
        for doc_id in doc_ids:
            try:
                resources.run(service.delete(kb, doc_id))
            except Exception as exc:
                logger.warning(
                    "保留期清理删除失败 kb=%s doc_id=%s：%s", kb.kb_code, doc_id, exc
                )
                continue
            deleted += 1
            logger.info("保留期清理删除文档 kb=%s doc_id=%s", kb.kb_code, doc_id)

    return {
        "deleted": deleted,
        "skipped": False,
        "retention_days": days,
        "cutoff": cutoff.isoformat(),
    }


@celery_app.task(bind=True, name="rag.purge_superseded_documents")
def purge_superseded_documents_task(
    self, *, retention_days: int | None = None, limit: int = 200
) -> dict[str, object]:
    """延迟清理被新版本替代的旧文档（Active Version 的"延迟清理队列"）。

    Active Version 切换只标记 superseded_by（检索侧即时排除旧版，查询透明），
    旧版本本身保留一个观察窗供审计/回滚；超过保留窗口后由本任务走统一删除
    状态机清理（索引、原始文件、元数据一致推进，失败留 DELETING 由重试收口）。
    """
    settings = get_settings()
    days = settings.superseded_retention_days if retention_days is None else retention_days
    if days <= 0:
        return {"deleted": 0, "skipped": True, "retention_days": days}

    resources = _get_resources()
    cutoff = datetime.now(UTC) - timedelta(days=days)
    deleted = 0
    for kb in resources.kb_registry.list_enabled():
        doc_ids = resources.run(
            resources.repo.list_superseded_document_ids(
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                cutoff=cutoff,
                limit=limit,
            )
        )
        service = DocumentDeletionService(
            pipeline=resources.pipeline,
            repo=resources.repo,
            object_store=resources.object_store,
        )
        for doc_id in doc_ids:
            try:
                resources.run(service.delete(kb, doc_id))
            except Exception as exc:
                logger.warning(
                    "superseded 清理删除失败 kb=%s doc_id=%s：%s", kb.kb_code, doc_id, exc
                )
                continue
            deleted += 1
            logger.info("superseded 旧版清理完成 kb=%s doc_id=%s", kb.kb_code, doc_id)

    return {
        "deleted": deleted,
        "skipped": False,
        "retention_days": days,
        "cutoff": cutoff.isoformat(),
    }


# ---------------------------------------------------------------- 索引对账


@celery_app.task(bind=True, name="rag.purge_expired_conversations")
def purge_expired_conversations_task(
    self, *, retention_days: int | None = None, limit: int = 500
) -> dict[str, object]:
    """过期会话清理（清单第四优先级第 20 项：会话过期和删除机制）。

    会话包含用户问答内容，长期保留有合规风险；按 updated_at 不活跃窗口
    （settings.conversation_retention_days）删除整段会话（消息连带删除）。
    """
    settings = get_settings()
    days = settings.conversation_retention_days if retention_days is None else retention_days
    if days <= 0:
        return {"deleted": 0, "skipped": True, "retention_days": days}
    resources = _get_resources()
    cutoff = datetime.now(UTC) - timedelta(days=days)
    conversation_ids = resources.run(
        resources.repo.list_expired_conversation_ids(cutoff=cutoff, limit=limit)
    )
    deleted = 0
    for conversation_id in conversation_ids:
        removed = resources.run(resources.repo.delete_conversation(conversation_id))
        if removed:
            deleted += 1
    return {
        "deleted": deleted,
        "skipped": False,
        "retention_days": days,
        "cutoff": cutoff.isoformat(),
    }


@celery_app.task(bind=True, name="rag.resume_pending_deletions")
def resume_pending_deletions_task(self, *, limit: int = 100) -> dict[str, int]:
    """删除失败收口任务：重试所有 DELETING 文档，保证原件最终被清理。

    删除流程任何阶段失败都会把文档留在 DELETING；本任务周期运行，
    让"原始文件暂时删不掉（对象存储抖动）"这类问题可以自愈。
    """
    resources = _get_resources()
    report: dict[str, int] = {}
    for kb in resources.kb_registry.list_enabled():
        service = DocumentDeletionService(
            pipeline=resources.pipeline,
            repo=resources.repo,
            object_store=resources.object_store,
        )
        try:
            report[kb.kb_code] = resources.run(
                service.resume_pending(kb, limit=limit)
            )
        except Exception as exc:
            logger.warning("删除重试任务失败 kb=%s：%s", kb.kb_code, exc)
            report[kb.kb_code] = 0
    return report


@celery_app.task(bind=True, name="rag.reconcile_index")
def reconcile_index_task(
    self, *, kb_key: str | None = None, repair: bool = True
) -> dict[str, object]:
    """索引对账兜底：清理孤儿 chunk、补齐缺失 chunk。

    kb_key 为空时对全部启用知识库对账；repair=False 只检测不修复（出报告）。
    """
    resources = _get_resources()
    if kb_key is not None:
        kbs = [resources.resolve_kb(kb_key)]
    else:
        kbs = list(resources.kb_registry.list_enabled())

    report: dict[str, object] = {}
    for kb in kbs:
        stats = _reconcile_kb(resources, kb, repair=repair)
        report[kb.kb_code] = stats
        if any(
            stats[key]
            for key in (
                "orphan_vectors_deleted",
                "orphan_keywords_deleted",
                "missing_vectors",
                "missing_keywords",
                "docs_unrepairable",
            )
        ):
            logger.warning("索引对账发现不一致 kb=%s stats=%s", kb.kb_code, stats)
        else:
            logger.info("索引对账一致 kb=%s stats=%s", kb.kb_code, stats)
    return {"repair": repair, "kbs": report}


def _reconcile_kb(
    resources: _WorkerResources, kb: KBConfig, *, repair: bool
) -> dict[str, int]:
    """单个知识库的对账流程：孤儿清理 → 缺失检测 → 按文档重灌修复。

    v1.5 起改为真正的流式对账（清单第二优先级第 10 项）：
    - 孤儿检测：索引侧分批产 chunk_id，PG 按批回查存在性（filter_existing_chunk_ids），
      不再把全库 Manifest 装进内存构建完整字典；
    - 缺失检测：PG 侧 document_chunks 主键游标分页（scan_kb_chunks），按批问索引，
      内存占用恒定，支持百万级 chunk。
    """
    runtime = resources.runtime_registry.runtime(kb)

    stats = {
        "expected_chunks": 0,
        "orphan_vectors_detected": 0,
        "orphan_keywords_detected": 0,
        "orphan_vectors_deleted": 0,
        "orphan_keywords_deleted": 0,
        "orphans_deferred": 0,
        "missing_vectors": 0,
        "missing_keywords": 0,
        "docs_repaired": 0,
        "docs_unrepairable": 0,
        "docs_pending_repair": 0,
        "index_params_reconciled": 0,
    }

    # 0) 索引参数对账（清单第五优先级第 22 项）：漂移原地重建 + 标量索引补齐。
    # 只在 repair 模式执行——drop_index 期间查询会降级，必须限制在维护窗口。
    if repair and runtime.vector_store.reconcile_index_params():
        stats["index_params_reconciled"] = 1

    # 1) 孤儿清理：索引里有、PG 清单里没有（已删文档残留 / 失效旧版清理失败的残留）
    # 只读模式（repair=False）只统计不删除：用户复现过 repair=False 仍删索引。
    # 在途保护：入库先写索引、后提交 PG 清单，窗口内的新块在 PG 查不到；
    # 存在 ingesting 文档时暂缓孤儿删除，删除前再做一次"清单复检 + 在途复检"。
    if hasattr(resources.repo, "list_inflight_document_ids"):
        inflight = resources.run(
            resources.repo.list_inflight_document_ids(
                kb_id=kb.kb_id, tenant_id=kb.tenant_id
            )
        )
    else:  # 兼容测试替身 / 旧仓储实现：视为无在途入库
        inflight = []

    def _clean_orphans(
        chunk_ids: list[str], *, deleted_key: str, detected_key: str, store: object
    ) -> None:
        if not chunk_ids:
            return
        stats[detected_key] += len(chunk_ids)
        if not repair:
            return
        if inflight:
            stats["orphans_deferred"] += len(chunk_ids)
            logger.warning(
                "检测到在途入库，暂缓孤儿删除 kb=%s ingesting=%s orphans=%s",
                kb.kb_code,
                len(inflight),
                len(chunk_ids),
            )
            return
        # 删除前复检：孤儿判定与删除之间可能有入库刚提交清单，二次确认避免误删
        still_absent = resources.run(
            resources.repo.filter_existing_chunk_ids(kb_id=kb.kb_id, chunk_ids=chunk_ids)
        )
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in still_absent]
        if ids:
            stats[deleted_key] += store.delete_by_ids(ids)  # type: ignore[attr-defined]

    for batch in runtime.vector_store.list_chunk_ids():
        existing = resources.run(
            resources.repo.filter_existing_chunk_ids(kb_id=kb.kb_id, chunk_ids=batch)
        )
        _clean_orphans(
            [chunk_id for chunk_id in batch if chunk_id not in existing],
            deleted_key="orphan_vectors_deleted",
            detected_key="orphan_vectors_detected",
            store=runtime.vector_store,
        )
    for batch in runtime.keyword_store.list_chunk_ids():
        existing = resources.run(
            resources.repo.filter_existing_chunk_ids(kb_id=kb.kb_id, chunk_ids=batch)
        )
        _clean_orphans(
            [chunk_id for chunk_id in batch if chunk_id not in existing],
            deleted_key="orphan_keywords_deleted",
            detected_key="orphan_keywords_detected",
            store=runtime.keyword_store,
        )

    # 2) 缺失检测：清单里有、索引里没有（写入失败 / 数据丢失）；游标分页流式扫描
    async def _scan_missing() -> tuple[set[str], list[str]]:
        missing_docs: set[str] = set()
        missing_chunk_ids: list[str] = []
        async for batch in resources.repo.scan_kb_chunks(kb_id=kb.kb_id):
            stats["expected_chunks"] += len(batch)
            # 子块（Milvus 侧清单）：父块不进向量库，不参与向量缺失判定
            child = [(chunk_id, doc_id) for doc_id, chunk_id, role in batch if role != "parent"]
            if child:
                found_vectors = runtime.vector_store.get_vectors_by_ids(
                    [chunk_id for chunk_id, _ in child]
                )
                for chunk_id, doc_id in child:
                    if chunk_id not in found_vectors:
                        stats["missing_vectors"] += 1
                        missing_docs.add(doc_id)
                        missing_chunk_ids.append(chunk_id)
            # 全量（ES 侧清单，含父块）
            found_keywords = {
                hit.chunk_id
                for hit in runtime.keyword_store.get_by_ids(
                    [chunk_id for _, chunk_id, _ in batch]
                )
            }
            for doc_id, chunk_id, _role in batch:
                if chunk_id not in found_keywords:
                    stats["missing_keywords"] += 1
                    missing_docs.add(doc_id)
                    missing_chunk_ids.append(chunk_id)
        return missing_docs, missing_chunk_ids

    missing_docs, missing_chunk_ids = resources.run(_scan_missing())
    if missing_chunk_ids and repair:
        # 索引状态落库：missing 标记供运维检索"哪些 chunk 待修复"；
        # 修复重灌走 mark_document_ready（清单先删后插）自动回到 indexed
        resources.run(resources.repo.mark_chunks_index_status(missing_chunk_ids, status="missing"))

    # 3) 按文档修复：从对象存储归档取原文重灌（版本不变，chunk_id 稳定，幂等）
    for doc_id in sorted(missing_docs):
        if repair:
            if _repair_document(resources, kb, doc_id):
                stats["docs_repaired"] += 1
            else:
                stats["docs_unrepairable"] += 1
        else:
            stats["docs_pending_repair"] += 1
    return stats


def _repair_document(resources: _WorkerResources, kb: KBConfig, doc_id: str) -> bool:
    """重灌单篇缺失文档：原文必须已归档到对象存储，否则只能人工重传。"""
    info = resources.run(resources.repo.get_document(doc_id))
    if info is None:
        return False
    if not info.raw_object_key:
        logger.warning("缺失文档无归档原文，无法自动修复 doc_id=%s", doc_id)
        return False
    try:
        data = resources.object_store.get_bytes(info.raw_object_key)
        filename = Path(urlsplit(info.source_uri).path).name or f"{doc_id}.txt"
        result = resources.pipeline.ingest_bytes(
            kb,
            data=data,
            filename=filename,
            source_uri=info.source_uri,
            title=info.title,
            version=info.version,
            acl_groups=info.acl_groups,
            governance={
                "source_id": info.source_id,
                "source_type": info.source_type,
                "owner": info.owner,
                "source_version": info.source_version,
                "authority_level": info.authority_level,
                "valid_from": info.valid_from,
                "valid_to": info.valid_to,
            },
            old_manifest=resources.run(resources.repo.get_chunk_manifest(info.doc_id)),
        )
        resources.run(
            resources.repo.mark_document_ready(
                info.doc_id,
                chunk_count=result.chunk_count,
                collection=result.collection,
                es_index=result.index,
                raw_object_key=result.raw_object_key,
                version=result.version,
                content_hash=result.content_hash,
                parse_report=result.quality,
                collected_at=result.collected_at,
                chunk_manifest=result.chunk_manifest,
            )
        )
        logger.info("缺失文档重灌修复完成 doc_id=%s chunks=%s", doc_id, result.chunk_count)
        return True
    except Exception as exc:
        logger.warning("缺失文档重灌修复失败 doc_id=%s err=%s", doc_id, exc)
        return False
