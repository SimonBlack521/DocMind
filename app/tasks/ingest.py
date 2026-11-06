"""异步入库任务：worker 进程内复用与 API 完全相同的 RAGPipeline，不做逻辑分叉。

RuntimeRegistry / S3ObjectStore / PG engine 等重组件按 worker 进程缓存（模块级懒加载单例），
进程退出（worker_shutdown）时统一关闭。元数据层是 asyncpg 异步驱动，
worker 线程内没有运行中的事件循环，用进程内专用循环串行执行，避免逐条新建连接。
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from celery import Task
from celery.signals import worker_shutdown

from app.core.errors import (
    DocumentParseError,
    DocumentQualityError,
    KBNotFoundError,
    KeywordStoreError,
    ModelUnavailableError,
    ObjectStoreError,
    VectorStoreError,
)
from app.core.logging import get_logger
from app.core.settings import get_settings
from app.db.models import (
    TASK_STATUS_FAILED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCESS,
)
from app.db.repo import MetadataRepo
from app.db.session import build_engine, build_session_factory
from app.domain.config_sync import ConfigSync
from app.domain.governance import compute_config_hash, resolve_version
from app.domain.kb_config import KBConfig, KBRegistry
from app.ingestion.governance import decide as governance_decide
from app.ingestion.loader import DocumentLoader, compute_doc_id, normalize_source_uri
from app.orchestration.pipeline import RAGPipeline
from app.orchestration.registry import RuntimeRegistry
from app.retrieval.cache import RetrievalCache
from app.storage.object_store import S3ObjectStore
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)

_T = TypeVar("_T")


class _WorkerResources:
    """worker 进程共享的重组件；首次任务执行时构建，避免 import 期建立连接。"""

    def __init__(self) -> None:
        settings = get_settings()
        self.kb_registry = KBRegistry.from_directory(settings.kb_config_dir)
        self.runtime_registry = RuntimeRegistry(settings)
        self.object_store = S3ObjectStore(
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            use_ssl=settings.s3_use_ssl,
        )
        # 与 API 进程共享同一 Redis：worker 入库成功后由 pipeline 内部失效检索缓存
        self.retrieval_cache = RetrievalCache(
            settings.redis_url, ttl_s=settings.retrieval_cache_ttl_s
        )
        # 元数据仓库先于 pipeline 构建：pipeline 的血缘归属/ACL 调和依赖 repo
        self.engine = build_engine(settings.database_url)
        self.repo = MetadataRepo(build_session_factory(self.engine))
        self.pipeline = RAGPipeline(
            settings=settings,
            registry=self.runtime_registry,
            object_store=self.object_store,
            cache=self.retrieval_cache,
            repo=self.repo,
        )
        # 多副本配置同步：API 建/改/删库后 INCR Redis 版本号，worker 节流检查后重扫
        self.config_sync = ConfigSync(settings.redis_url, enabled=not settings.celery_eager)
        self._config_sync_state: tuple[float, int | None] = (0.0, None)
        self.loop = asyncio.new_event_loop()

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        return self.loop.run_until_complete(coro)

    def refresh_kb_if_changed(self) -> None:
        """节流检查 Redis 配置版本号；变化时重扫知识库配置目录。

        与 API 侧 `_refresh_if_config_changed` 同策略：每 kb_config_sync_interval_s
        秒最多查一次 Redis；Redis 不可用时保持现状，不影响任务执行。
        """
        interval = get_settings().kb_config_sync_interval_s
        if not self.config_sync.enabled or interval <= 0:
            return
        now = time.monotonic()
        last_checked, last_version = self._config_sync_state
        if now - last_checked < interval:
            return
        version = self.config_sync.version()
        self._config_sync_state = (now, version)
        if version is None or version == last_version:
            return
        logger.info("检测到配置版本变化（%s -> %s），重扫知识库配置", last_version, version)
        self.kb_registry = KBRegistry.from_directory(get_settings().kb_config_dir)

    def resolve_kb(self, key: str) -> KBConfig:
        """知识库由 API 进程运行时创建时，worker 首次未命中就重扫配置目录。"""
        self.refresh_kb_if_changed()
        try:
            return self.kb_registry.resolve(key)
        except KBNotFoundError:
            self.kb_registry = KBRegistry.from_directory(get_settings().kb_config_dir)
            return self.kb_registry.resolve(key)

    def close(self) -> None:
        self.runtime_registry.close()
        self.object_store.close()
        self.retrieval_cache.close()
        self.config_sync.close()
        self.run(self.engine.dispose())
        self.loop.close()


_resources: _WorkerResources | None = None
_resources_lock = threading.Lock()

# 瞬时错误（对象存储/向量库/关键词库/模型服务超时抖动）才自动重试；
# 解析、质量、权限类错误重试无意义，直接落 failed。
# 重试耗尽后写死信表（event_id=task id 幂等），人工排查后可重放，不阻塞消费链路。
_TRANSIENT_ERRORS = (
    ObjectStoreError,
    VectorStoreError,
    KeywordStoreError,
    ModelUnavailableError,
)


def _get_resources() -> _WorkerResources:
    global _resources
    if _resources is None:
        with _resources_lock:
            if _resources is None:
                _resources = _WorkerResources()
    return _resources


@worker_shutdown.connect
def _on_worker_shutdown(**kwargs: Any) -> None:
    global _resources
    with _resources_lock:
        if _resources is not None:
            _resources.close()
            _resources = None


def _mark_failed(
    resources: _WorkerResources,
    task_id: str,
    doc_id: str | None,
    error: str,
    *,
    mark_document: bool = True,
) -> None:
    """失败落库；元数据库本身异常时只记日志，不掩盖原始错误。

    mark_document=False 用于质量门禁场景：文档应保持 quarantined，
    不能被通用的 failed 覆盖掉（两者后续的处理方式不同）。
    """
    try:
        resources.run(
            resources.repo.mark_task(task_id, status=TASK_STATUS_FAILED, error=error[-500:])
        )
        if doc_id is not None and mark_document:
            resources.run(resources.repo.mark_document_failed(doc_id))
    except Exception:
        logger.exception("失败状态写入元数据库失败 task=%s", task_id)


def _put_dead_letter(
    resources: _WorkerResources,
    *,
    event_id: str,
    kb_key: str,
    doc_id: str | None,
    payload: dict[str, Any],
    error: str,
) -> None:
    """死信落库；死信表本身不可用时只记日志，不掩盖原始异常。"""
    try:
        kb = resources.resolve_kb(kb_key)
        resources.run(
            resources.repo.put_dead_letter(
                event_id=event_id,
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                doc_id=doc_id,
                payload=payload,
                error=error,
            )
        )
        logger.error("入库任务重试耗尽，已落死信 event=%s doc_id=%s", event_id, doc_id)
    except Exception:
        logger.exception("死信写入失败 event=%s", event_id)


@celery_app.task(
    bind=True,
    max_retries=3,
    autoretry_for=_TRANSIENT_ERRORS,
    retry_backoff=True,
)
def ingest_document_task(
    self: Task,
    *,
    kb_key: str,
    payload: bytes | None,
    staging_key: str | None,
    filename: str,
    title: str,
    source_uri: str | None,
    acl_groups: list[str] | None = None,
    source_id: str | None = None,
    source_type: str = "file",
    owner: str | None = None,
    source_version: str | None = None,
    authority_level: int = 0,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> dict[str, Any]:
    """文件 bytes → 解析 → 治理裁决 → 质量门禁 → 切块 → 向量化 → 双写。

    治理裁决与 API 同步端点保持一致（内容未变复用 / 跨来源重复登记 / 内容变化升版本），
    差异只是执行位置在 worker 内。质量门禁未通过时文档置 quarantined 而非 failed，
    task 本身仍记 failed，便于前端提示"文件已收到但解析质量不达标"。
    """
    resources = _get_resources()
    task_id: str = self.request.id
    doc_id: str | None = None
    keep_staging = False
    try:
        # 解析失败（如 kb 不存在）也要走统一失败落库，因此放在 try 内
        kb = resources.resolve_kb(kb_key)
        resources.run(resources.repo.mark_task(task_id, status=TASK_STATUS_RUNNING))

        data = payload
        if data is None:
            if not staging_key:
                raise DocumentParseError("payload 与 staging_key 必须提供一个")
            data = resources.object_store.get_bytes(staging_key)

        resolved_source_uri = source_uri or f"kb://{kb.kb_code}/{filename}"
        loader = DocumentLoader(
            tenant_id=kb.tenant_id,
            kb_id=kb.kb_id,
            redact_pii=get_settings().privacy_redaction_enabled,
        )
        document = loader.load_bytes(
            data,
            source_uri=resolved_source_uri,
            title=title or Path(filename).stem,
            suffix=Path(filename).suffix,
        )
        doc_id = compute_doc_id(kb.tenant_id, kb.kb_id, resolved_source_uri)
        normalized_uri = normalize_source_uri(resolved_source_uri)

        existing = resources.run(resources.repo.get_document(doc_id))
        valid_from_dt = datetime.fromisoformat(valid_from) if valid_from else None
        valid_to_dt = datetime.fromisoformat(valid_to) if valid_to else None
        # 切块配置 + Embedding 版本指纹：任一变化都必须重建，不能直接复用旧索引
        parse_config_hash = compute_config_hash(
            {**kb.chunking.model_dump(), "embedding_version": get_settings().embedding_version}
        )
        # 与同步路径共用同一份治理裁决（ACL + 治理字段 + 内容 + 状态 + 有效期 +
        # 切块/Embedding 指纹），避免异步 worker 的简化判断漏掉变更导致旧索引被复用
        decision = governance_decide(
            new_content_hash=document.content_hash,
            new_acl=acl_groups,
            new_source_id=(source_id or None),
            new_source_type=(source_type or "file"),
            new_owner=(owner or None),
            new_source_version=(source_version or None),
            new_authority_level=authority_level,
            existing_status=(existing.status if existing else None),
            existing_content_hash=(existing.content_hash if existing else None),
            existing_acl=(existing.acl_groups if existing else None),
            existing_source_id=(existing.source_id if existing else None),
            existing_source_type=(existing.source_type if existing else None),
            existing_owner=(existing.owner if existing else None),
            existing_source_version=(existing.source_version if existing else None),
            existing_authority_level=(existing.authority_level if existing else 0),
            new_valid_from=valid_from_dt,
            new_valid_to=valid_to_dt,
            existing_valid_from=(existing.valid_from if existing else None),
            existing_valid_to=(existing.valid_to if existing else None),
            new_config_hash=parse_config_hash,
            existing_config_hash=(existing.parse_config_hash if existing else None),
        )
        if decision.decision == "reuse" and existing is not None:
            resources.run(
                resources.repo.mark_task(
                    task_id, status=TASK_STATUS_SUCCESS, doc_id=existing.doc_id
                )
            )
            return {
                "task_id": task_id,
                "status": TASK_STATUS_SUCCESS,
                "doc_id": existing.doc_id,
                "chunk_count": existing.chunk_count,
                "version": existing.version,
                "reused": True,
            }

        governance: dict[str, Any] = {
            "source_id": source_id or None,
            "source_type": source_type or "file",
            "owner": owner or None,
            "source_version": source_version or None,
            "authority_level": authority_level,
            "valid_from": valid_from_dt,
            "valid_to": valid_to_dt,
        }

        # 场景 2：同库已有相同内容 → 只登记血缘
        duplicate = resources.run(
            resources.repo.find_document_by_content_hash(
                kb_id=kb.kb_id,
                content_hash=document.content_hash,
                exclude_doc_id=doc_id,
            )
        )
        if duplicate is not None:
            resources.run(
                resources.repo.upsert_document_ingesting(
                    doc_id=doc_id,
                    kb_id=kb.kb_id,
                    tenant_id=kb.tenant_id,
                    source_uri=normalized_uri,
                    title=title or Path(filename).stem,
                    version=1,
                    acl_groups=acl_groups,
                    content_hash=document.content_hash,
                    parser_name=document.parser_name,
                    parser_version=document.parser_version,
                    parse_config_hash=parse_config_hash,
                    **governance,
                )
            )
            resources.run(
                resources.repo.mark_document_duplicate(
                    doc_id,
                    duplicate_of=duplicate.doc_id,
                    content_hash=document.content_hash,
                    parse_report={},
                )
            )
            # 血缘并集 ACL 调和：与 API 同步路径同一语义（清单第 9 项），
            # 共享内容的任一来源更宽松时刷新承载者索引可见性，避免其用户漏招
            resources.pipeline.reconcile_lineage_acl(
                kb, content_hash=document.content_hash, carrier_doc_id=duplicate.doc_id
            )
            resources.run(
                resources.repo.mark_task(task_id, status=TASK_STATUS_SUCCESS, doc_id=doc_id)
            )
            logger.info("异步入库重复内容 doc_id=%s duplicate_of=%s", doc_id, duplicate.doc_id)
            return {
                "task_id": task_id,
                "status": TASK_STATUS_SUCCESS,
                "doc_id": doc_id,
                "chunk_count": duplicate.chunk_count,
                "duplicate_of": duplicate.doc_id,
            }

        # 场景 3：内容变化 → 自动升版本，整篇覆盖重建
        version = resolve_version(
            existing.content_hash if existing else None,
            existing.version if existing else 1,
            document.content_hash,
        )
        resources.run(
            resources.repo.upsert_document_ingesting(
                doc_id=doc_id,
                kb_id=kb.kb_id,
                tenant_id=kb.tenant_id,
                source_uri=normalized_uri,
                title=title or Path(filename).stem,
                version=version,
                acl_groups=acl_groups,
                content_hash=document.content_hash,
                parser_name=document.parser_name,
                parser_version=document.parser_version,
                parse_config_hash=parse_config_hash,
                **governance,
            )
        )

        result = resources.pipeline.ingest_document(
            kb,
            document,
            raw_bytes=data,
            version=version,
            acl_groups=acl_groups,
            governance=governance,
            old_manifest=(
                resources.run(resources.repo.get_chunk_manifest(existing.doc_id))
                if existing is not None
                else None
            ),
        )

        resources.run(
            resources.repo.mark_document_ready(
                result.doc_id,
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
        # Active Version 切换：与 API 同步路径一致，新版本 ready 后替代同 source_id 旧文档
        try:
            superseded_count = resources.run(
                resources.repo.supersede_previous_versions(
                    doc_id=result.doc_id,
                    kb_id=kb.kb_id,
                    tenant_id=kb.tenant_id,
                    source_id=source_id or None,
                )
            )
        except Exception as exc:
            logger.warning("Active Version 切换失败 doc_id=%s：%s", result.doc_id, exc)
            superseded_count = 0
        if superseded_count:
            logger.info(
                "Active Version 切换 doc_id=%s 替代旧版本 %d 篇",
                result.doc_id,
                superseded_count,
            )
        resources.run(
            resources.repo.mark_task(task_id, status=TASK_STATUS_SUCCESS, doc_id=result.doc_id)
        )
        logger.info(
            "异步入库成功 task=%s doc_id=%s chunks=%s version=%s",
            task_id,
            result.doc_id,
            result.chunk_count,
            result.version,
        )
        return {
            "task_id": task_id,
            "status": TASK_STATUS_SUCCESS,
            "doc_id": result.doc_id,
            "chunk_count": result.chunk_count,
            "version": result.version,
        }
    except DocumentQualityError as exc:
        # 质量门禁：文档隔离保留，任务本身失败并带上可读原因
        if doc_id is not None:
            try:
                resources.run(
                    resources.repo.mark_document_quarantined(doc_id, parse_report=exc.details)
                )
            except Exception:
                logger.exception("隔离状态写入失败 doc_id=%s", doc_id)
        _mark_failed(resources, task_id, doc_id, str(exc), mark_document=False)
        raise
    except _TRANSIENT_ERRORS as exc:
        # 还有重试额度时交给 autoretry：保留 staging 对象，任务保持 running
        if self.request.retries < (self.max_retries or 0):
            keep_staging = True
            logger.warning(
                "瞬时错误，等待自动重试 task=%s retries=%s err=%s",
                task_id,
                self.request.retries,
                exc,
            )
            raise
        # 重试耗尽：落死信（人工排查后可按 payload 重放），文档与任务置 failed
        _put_dead_letter(
            resources,
            event_id=task_id,
            kb_key=kb_key,
            doc_id=doc_id,
            payload={
                "kb_key": kb_key,
                "staging_key": staging_key,
                "filename": filename,
                "title": title,
                "source_uri": source_uri,
                "acl_groups": acl_groups,
                "source_id": source_id,
                "source_type": source_type,
                "owner": owner,
                "source_version": source_version,
                "authority_level": authority_level,
                "valid_from": valid_from,
                "valid_to": valid_to,
            },
            error=str(exc),
        )
        _mark_failed(resources, task_id, doc_id, f"瞬时错误且重试已耗尽：{exc}")
        raise
    except Exception as exc:
        _mark_failed(resources, task_id, doc_id, str(exc))
        raise
    finally:
        if staging_key and not keep_staging:
            try:
                resources.object_store.delete(staging_key)
            except ObjectStoreError as exc:
                logger.warning("清理暂存对象失败 key=%s：%s", staging_key, exc.message)
