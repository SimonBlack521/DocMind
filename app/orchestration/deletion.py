"""文档删除服务（v2.0.1）：单篇删除、按责任人批量删除、失败重试走同一套状态机。

删除不是"一次调用全部清掉"，而是分阶段推进，任何一步失败都保留可恢复状态：

    READY
      → DELETING（先落状态，失败也能被重试任务重新扫描到）
      → 索引已清理（Milvus + ES）
      → 原始文件已清理（对象存储）
      → 元数据已清理（PG documents + document_chunks）
      → 完成（行已删除，写审计）

关键语义：
- 原始文件删除失败时**不再删除 PG 元数据**：否则对象存储里的原件会失去追踪，
  违反"用户已删除但文件仍留存"的合规要求；
- 任何阶段失败都把文档留在 DELETING，由 `rag.resume_pending_deletions` 周期重试；
- 单篇与批量删除共用本服务，避免两条路径的行为漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.concurrency import run_in_threadpool

from app.core.errors import AppError, DocumentNotFoundError, ObjectStoreError
from app.core.logging import get_logger
from app.db.models import AUDIT_ACTION_DOCUMENT_DELETE
from app.db.repo import DocumentInfo, MetadataRepo
from app.domain.kb_config import KBConfig
from app.orchestration.pipeline import RAGPipeline
from app.storage.object_store import ObjectStore
from app.storage.vector_base import merge_lineage_acl

logger = get_logger(__name__)

AUDIT_ACTION_DOCUMENT_DELETE_FAILED = "document.delete_failed"


@dataclass(frozen=True, slots=True)
class DeleteOutcome:
    """单篇删除结果：索引删除计数 + 各阶段完成标记（便于审计与重试判断）。"""

    doc_id: str
    deleted_vectors: int
    deleted_keywords: int
    object_deleted: bool
    metadata_deleted: bool


@dataclass(slots=True)
class PurgeOutcome:
    """批量删除结果：成功数、失败文档与失败原因，失败项留给重试任务。"""

    owner: str
    deleted_documents: int = 0
    deleted_vectors: int = 0
    deleted_keywords: int = 0
    failures: dict[str, str] = field(default_factory=dict)


class DocumentDeletionService:
    """统一删除入口：API 单篇/批量与后台保留期清理都必须经过这里。"""

    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        repo: MetadataRepo,
        object_store: ObjectStore,
    ) -> None:
        self._pipeline = pipeline
        self._repo = repo
        self._object_store = object_store

    async def delete(self, kb: KBConfig, doc_id: str, *, user_id: str = "") -> DeleteOutcome:
        """按状态机删除单篇文档；失败保留 DELETING 状态并写审计。"""
        info = await self._repo.get_document(doc_id)
        if (
            info is None
            or info.kb_id != kb.kb_id
            or info.tenant_id != kb.tenant_id
        ):
            raise DocumentNotFoundError(
                f"文档不存在：{doc_id}", details={"doc_id": doc_id}
            )

        # 先落 DELETING：后续任何一步失败，重试任务都能从 PG 里重新发现它
        await self._repo.mark_document_deleting(doc_id)
        # 状态变更与缓存失效协同（清单第三轮评审 P0）：DELETING 文档的残留索引
        # 在清理完成前必须对检索不可见，这里立即失效本库缓存，不留 TTL 窗口
        await run_in_threadpool(self._pipeline.invalidate_retrieval_cache, kb)
        raw_key = info.raw_object_key

        # 重复内容血缘处理（必须在删除索引之前）：
        # - 删除承载者：把另一篇同内容文档提升为新承载者（重灌索引），并改指血缘；
        # - 删除任一来源：按剩余血缘重算并集 ACL，从承载者索引收缩可见性。
        lineage_target: tuple[str, list[str]] | None = None
        try:
            lineage_target = await self._prepare_lineage(kb, info)
        except Exception as exc:
            # 迁移失败时中止删除，避免"元数据还在、物理索引已消失"的半删除状态；
            # 文档保持 DELETING，由重试任务继续（原文仍在对象存储，可再次迁移）。
            await self._record_failure(
                kb, doc_id, user_id, stage="lineage", error=exc, raw_key=raw_key
            )
            raise

        try:
            result = await run_in_threadpool(
                self._pipeline.delete_document, kb, doc_id
            )
        except Exception as exc:
            await self._record_failure(
                kb, doc_id, user_id, stage="index", error=exc, raw_key=raw_key
            )
            raise

        object_deleted = False
        if raw_key:
            try:
                await run_in_threadpool(self._object_store.delete, raw_key)
            except ObjectStoreError as exc:
                # 原始文件没删掉：保留元数据，避免原件失去追踪；状态留在 DELETING 供重试
                await self._record_failure(
                    kb, doc_id, user_id, stage="object", error=exc, raw_key=raw_key
                )
                raise
            object_deleted = True

        try:
            await self._repo.delete_document(doc_id)
        except Exception as exc:
            await self._record_failure(
                kb, doc_id, user_id, stage="metadata", error=exc, raw_key=raw_key
            )
            raise

        # 元数据已删除，剩余血缘的并集 ACL 可以安全收敛（私有来源删除后，
        # 承载者索引不应继续保留更宽的可见性）；失败只告警，不影响删除结果，
        # 下次登记/删除时会再次调和。
        if lineage_target is not None:
            carrier_doc_id, merged_acl = lineage_target
            try:
                await run_in_threadpool(
                    self._pipeline.refresh_lineage_acl,
                    kb,
                    carrier_doc_id=carrier_doc_id,
                    merged_acl=merged_acl,
                )
            except Exception as exc:
                logger.warning(
                    "删除后血缘 ACL 收敛失败 carrier=%s：%s", carrier_doc_id, exc
                )

        await self._repo.write_audit(
            tenant_id=kb.tenant_id,
            kb_id=kb.kb_id,
            user_id=user_id,
            action=AUDIT_ACTION_DOCUMENT_DELETE,
            doc_id=doc_id,
            detail={
                "deleted_vectors": result.deleted_vectors,
                "deleted_keywords": result.deleted_keywords,
                "raw_object_key": raw_key,
                "object_deleted": object_deleted,
            },
        )
        return DeleteOutcome(
            doc_id=doc_id,
            deleted_vectors=result.deleted_vectors,
            deleted_keywords=result.deleted_keywords,
            object_deleted=object_deleted,
            metadata_deleted=True,
        )

    async def purge_owner(
        self, kb: KBConfig, owner: str, *, user_id: str = ""
    ) -> PurgeOutcome:
        """按责任人批量删除：逐篇走同一状态机，单篇失败不影响其余文档。"""
        doc_ids = await self._repo.list_document_ids_by_owner(
            kb_id=kb.kb_id, tenant_id=kb.tenant_id, owner=owner
        )
        outcome = PurgeOutcome(owner=owner)
        for doc_id in doc_ids:
            try:
                result = await self.delete(kb, doc_id, user_id=user_id)
            except AppError as exc:
                outcome.failures[doc_id] = exc.message
                logger.warning(
                    "批量删除单篇失败 owner=%s doc_id=%s：%s",
                    owner,
                    doc_id,
                    exc.message,
                )
                continue
            outcome.deleted_documents += 1
            outcome.deleted_vectors += result.deleted_vectors
            outcome.deleted_keywords += result.deleted_keywords
        return outcome

    async def resume_pending(self, kb: KBConfig, *, limit: int = 100) -> int:
        """重试 DELETING 状态文档：供后台任务周期调用，返回本次成功删除数。"""
        doc_ids = await self._repo.list_deleting_document_ids(
            kb_id=kb.kb_id, tenant_id=kb.tenant_id, limit=limit
        )
        deleted = 0
        for doc_id in doc_ids:
            try:
                await self.delete(kb, doc_id)
            except AppError as exc:
                logger.warning(
                    "删除重试仍失败 doc_id=%s：%s", doc_id, exc.message
                )
                continue
            deleted += 1
        return deleted

    async def _prepare_lineage(
        self, kb: KBConfig, info: DocumentInfo
    ) -> tuple[str, list[str]] | None:
        """删除前处理重复内容血缘，返回（删除后应刷新 ACL 的承载者, 并集 ACL）。

        - 被删文档是承载者且仍有同内容来源：先迁移承载者（重灌索引），再把
          其余 duplicate 的血缘指针改指新承载者，保证删除后索引不会消失；
        - 没有可迁移来源：只返回剩余血缘的并集 ACL，让承载者索引收缩可见性；
        - 无同内容来源（独占内容）：返回 None，不做额外处理。
        """
        content_hash = getattr(info, "content_hash", None)
        if not content_hash or self._repo is None:
            return None
        lineage = await self._repo.list_lineage_documents(
            kb_id=kb.kb_id, tenant_id=kb.tenant_id, content_hashes=[content_hash]
        )
        remaining = [
            doc
            for doc in lineage.get(content_hash, [])
            if doc.doc_id != info.doc_id
        ]
        if not remaining:
            return None
        merged = merge_lineage_acl([doc.acl_groups for doc in remaining])

        # 剩余来源里已有 ready 承载者（被删的是 duplicate / 另一份物理索引）：
        # 只需收缩该承载者的 ACL，无需迁移。
        surviving_carrier = next(
            (
                doc
                for doc in remaining
                if doc.duplicate_of is None and doc.chunk_count > 0
            ),
            None,
        )
        if surviving_carrier is not None:
            return surviving_carrier.doc_id, merged

        # 被删文档是唯一承载者：挑一个来源重灌为新的物理承载者
        is_carrier = info.duplicate_of is None and info.chunk_count > 0
        if not is_carrier:
            return None
        successor = remaining[0]
        await self._migrate_carrier(kb, carrier=info, successor=successor, merged_acl=merged)
        return successor.doc_id, merged

    async def _migrate_carrier(
        self,
        kb: KBConfig,
        *,
        carrier: DocumentInfo,
        successor: DocumentInfo,
        merged_acl: list[str],
    ) -> None:
        """把 duplicate 来源提升为新的物理承载者：原文重灌 + 血缘改指 + 元数据 ready。

        重复内容只保留一份物理索引，因此承载者删除时必须先把另一来源重建出来，
        否则会出现"元数据还在、实际索引已消失"。原文优先取来源自己的归档，
        没有时退回被删承载者的归档（内容指纹相同，字节等价）。
        """
        source_key = successor.raw_object_key or carrier.raw_object_key
        if not source_key:
            raise AppError(
                "无法迁移重复内容承载者：缺少归档原文",
                details={"doc_id": successor.doc_id},
            )
        data = await run_in_threadpool(self._object_store.get_bytes, source_key)
        filename = Path(urlsplit(successor.source_uri).path).name or f"{successor.doc_id}.txt"
        result = await run_in_threadpool(
            self._pipeline.ingest_bytes,
            kb,
            data=data,
            filename=filename,
            source_uri=successor.source_uri,
            title=successor.title,
            version=successor.version,
            acl_groups=merged_acl,
            governance={
                "source_id": successor.source_id,
                "source_type": successor.source_type,
                "owner": successor.owner,
                "source_version": successor.source_version,
                "authority_level": successor.authority_level,
                "valid_from": successor.valid_from,
                "valid_to": successor.valid_to,
            },
            old_manifest=await self._repo.get_chunk_manifest(successor.doc_id),
        )
        await self._repo.mark_document_ready(
            successor.doc_id,
            chunk_count=result.chunk_count,
            collection=result.collection,
            es_index=result.index,
            raw_object_key=result.raw_object_key,
            version=result.version,
            content_hash=result.content_hash,
            parse_report=result.quality,
            collected_at=result.collected_at,
            chunk_manifest=result.chunk_manifest,
            clear_duplicate=True,
        )
        # 其余 duplicate 仍指向已删除的承载者，改指新承载者
        await self._repo.repoint_duplicates(
            old_carrier_doc_id=carrier.doc_id,
            new_carrier_doc_id=successor.doc_id,
        )
        logger.info(
            "重复内容承载者迁移完成 kb=%s old=%s new=%s chunks=%s",
            kb.kb_code,
            carrier.doc_id,
            successor.doc_id,
            result.chunk_count,
        )

    async def _record_failure(
        self,
        kb: KBConfig,
        doc_id: str,
        user_id: str,
        *,
        stage: str,
        error: Exception,
        raw_key: str | None,
    ) -> None:
        """失败留痕：文档保持 DELETING，审计记录失败阶段，便于人工定位。"""
        message = getattr(error, "message", str(error))
        logger.warning(
            "文档删除失败 doc_id=%s stage=%s：%s", doc_id, stage, message
        )
        try:
            await self._repo.write_audit(
                tenant_id=kb.tenant_id,
                kb_id=kb.kb_id,
                user_id=user_id,
                action=AUDIT_ACTION_DOCUMENT_DELETE_FAILED,
                doc_id=doc_id,
                detail={"stage": stage, "error": message[:300], "raw_object_key": raw_key},
            )
        except Exception:
            # 审计失败不能掩盖原始删除错误
            logger.exception("删除失败审计写入失败 doc_id=%s", doc_id)
