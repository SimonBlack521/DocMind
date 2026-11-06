"""元数据仓库：文档与入库任务的状态读写、审计日志，API 层与 Celery 任务共用。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging import get_logger
from app.db.models import (
    DOC_STATUS_DELETING,
    DOC_STATUS_DUPLICATE,
    DOC_STATUS_FAILED,
    DOC_STATUS_INGESTING,
    DOC_STATUS_QUARANTINED,
    DOC_STATUS_READY,
    OUTBOX_EVENT_INGEST,
    OUTBOX_MAX_ATTEMPTS,
    OUTBOX_STATUS_DISPATCHED,
    OUTBOX_STATUS_DISPATCHING,
    OUTBOX_STATUS_FAILED,
    OUTBOX_STATUS_PENDING,
    TASK_STATUS_FAILED,
    TASK_STATUS_SUCCESS,
    AuditLogRow,
    ConversationMessageRow,
    ConversationRow,
    DeadLetterEventRow,
    DocumentChunkRow,
    DocumentRow,
    IngestTaskRow,
    OutboxEventRow,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class TaskInfo:
    """任务状态的只读视图，避免把 ORM 对象泄漏到会话外。"""

    task_id: str
    doc_id: str | None
    kb_id: str
    tenant_id: str
    status: str
    error: str | None
    # 任务创建者（None = 系统任务，如连接器同步）：任务详情接口的属主校验依据
    created_by: str | None = None


@dataclass(slots=True)
class OutboxEventInfo:
    """Outbox 待投递事件视图（dispatcher 只读所需字段）。"""

    event_id: str
    event_type: str
    kb_id: str
    tenant_id: str
    payload: dict
    attempts: int


@dataclass(slots=True)
class ConversationInfo:
    """会话元数据视图：路由层据此做归属校验（同租户 + 同用户 + 同知识库）。"""

    conversation_id: str
    tenant_id: str
    kb_id: str
    user_id: str
    title: str | None


@dataclass(slots=True)
class ConversationMessage:
    """会话消息视图（role: user / assistant）。"""

    role: str
    content: str


@dataclass(slots=True)
class DocumentInfo:
    """文档元数据的只读视图（含治理字段），供 API/任务层做版本与去重裁决。"""

    doc_id: str
    kb_id: str
    tenant_id: str
    source_uri: str
    title: str
    chunk_count: int
    version: int
    status: str
    created_at: datetime
    updated_at: datetime
    # 文档级 ACL：空列表 = 公开（全员可见）
    acl_groups: list[str] = field(default_factory=list)
    # ---- 知识治理字段 ----
    source_id: str | None = None
    source_type: str = "file"
    owner: str | None = None
    source_version: str | None = None
    content_hash: str | None = None
    collected_at: datetime | None = None
    authority_level: int = 0
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    superseded_by: str | None = None
    duplicate_of: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    parse_config_hash: str | None = None
    parse_report: dict = field(default_factory=dict)
    # 当前版本 chunk 清单：增量入库 Diff 与索引对账的权威依据
    chunk_manifest: list[dict] = field(default_factory=list)
    # 存储位置（同内容重复上传时用于复用，不必重新索引）
    collection: str = ""
    es_index: str = ""
    raw_object_key: str | None = None


@dataclass(slots=True)
class AuditLogInfo:
    """审计日志的只读视图。"""

    audit_id: str
    tenant_id: str
    kb_id: str
    user_id: str
    action: str
    doc_id: str | None
    detail: dict | None
    created_at: datetime


@dataclass(slots=True)
class DeadLetterInfo:
    """死信事件的只读视图：payload 是重放所需的原始任务参数。"""

    event_id: str
    kb_id: str
    tenant_id: str
    doc_id: str | None
    payload: dict
    error: str | None
    created_at: datetime


def _to_document_info(row: DocumentRow) -> DocumentInfo:
    return DocumentInfo(
        doc_id=row.doc_id,
        kb_id=row.kb_id,
        tenant_id=row.tenant_id,
        source_uri=row.source_uri,
        title=row.title,
        chunk_count=row.chunk_count,
        version=row.version,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
        acl_groups=list(row.acl_groups or []),
        source_id=row.source_id,
        source_type=row.source_type,
        owner=row.owner,
        source_version=row.source_version,
        content_hash=row.content_hash,
        collected_at=row.collected_at,
        authority_level=row.authority_level,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        superseded_by=row.superseded_by,
        duplicate_of=row.duplicate_of,
        parser_name=row.parser_name,
        parser_version=row.parser_version,
        parse_config_hash=row.parse_config_hash,
        parse_report=dict(row.parse_report or {}),
        chunk_manifest=list(row.chunk_manifest or []),
        collection=row.collection,
        es_index=row.es_index,
        raw_object_key=row.raw_object_key,
    )


def acl_visible(acl_groups: list[str], allowed_groups: frozenset[str] | None) -> bool:
    """文档可见性：allowed_groups 为 None 不限制；否则文档公开（空）或与用户组有交集。"""
    if allowed_groups is None:
        return True
    return not acl_groups or bool(set(acl_groups) & allowed_groups)


def _to_audit_log_info(row: AuditLogRow) -> AuditLogInfo:
    detail: dict | None = None
    if row.detail:
        try:
            parsed = json.loads(row.detail)
            if isinstance(parsed, dict):
                detail = parsed
        except ValueError:
            logger.warning("审计日志 detail 非 JSON，按空处理 audit_id=%s", row.audit_id)
    return AuditLogInfo(
        audit_id=row.audit_id,
        tenant_id=row.tenant_id,
        kb_id=row.kb_id,
        user_id=row.user_id,
        action=row.action,
        doc_id=row.doc_id,
        detail=detail,
        created_at=row.created_at,
    )


class MetadataRepo:
    """所有方法各自开启短事务，调用方无需管理会话。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    # ---------------------------------------------------------------- 文档
    async def upsert_document_ingesting(
        self,
        *,
        doc_id: str,
        kb_id: str,
        tenant_id: str,
        source_uri: str,
        title: str,
        version: int = 1,
        acl_groups: list[str] | None = None,
        source_id: str | None = None,
        source_type: str = "file",
        owner: str | None = None,
        source_version: str | None = None,
        content_hash: str | None = None,
        authority_level: int = 0,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        parser_name: str | None = None,
        parser_version: str | None = None,
        parse_config_hash: str | None = None,
    ) -> None:
        """入库开始时登记/覆盖文档行（覆盖式重建时回到 ingesting，ACL 同步重置为新值）。

        治理字段（来源标识/责任人/版本/时效/权威度）随每次入库刷新，
        content_hash 由调用方先算好传入，便于同库去重与内容漂移检测。
        """
        async with self._sessions() as session, session.begin():
            await session.merge(
                DocumentRow(
                    doc_id=doc_id,
                    kb_id=kb_id,
                    tenant_id=tenant_id,
                    source_uri=source_uri,
                    title=title,
                    version=version,
                    status=DOC_STATUS_INGESTING,
                    acl_groups=list(acl_groups or []),
                    source_id=source_id,
                    source_type=source_type,
                    owner=owner,
                    source_version=source_version,
                    content_hash=content_hash,
                    authority_level=authority_level,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    parser_name=parser_name,
                    parser_version=parser_version,
                    parse_config_hash=parse_config_hash,
                )
            )

    async def mark_document_ready(
        self,
        doc_id: str,
        *,
        chunk_count: int,
        collection: str,
        es_index: str,
        raw_object_key: str | None,
        version: int | None = None,
        content_hash: str | None = None,
        parse_report: dict | None = None,
        collected_at: datetime | None = None,
        chunk_manifest: list[dict] | None = None,
        clear_duplicate: bool = False,
    ) -> None:
        """入库成功后回填块数、存储位置、版本与质量报告，状态置 ready。

        clear_duplicate=True 用于重复内容承载者迁移：把原 duplicate 文档提升为
        新的物理承载者时，必须同时清除它指向已删除承载者的血缘指针。
        """
        async with self._sessions() as session, session.begin():
            row = await session.get(DocumentRow, doc_id)
            if row is None:
                return
            row.chunk_count = chunk_count
            row.collection = collection
            row.es_index = es_index
            row.raw_object_key = raw_object_key
            row.status = DOC_STATUS_READY
            if clear_duplicate:
                row.duplicate_of = None
            # 内容变化时任务层会把版本号上调，这里以实际写入版本为准
            if version is not None:
                row.version = version
            if content_hash is not None:
                row.content_hash = content_hash
            if parse_report is not None:
                row.parse_report = parse_report
            if collected_at is not None:
                row.collected_at = collected_at
            if chunk_manifest is not None:
                # v1.4：清单落独立表（先删后插，保证"当前版本"语义），
                # documents.chunk_manifest 只作为旧数据兼容列保留
                await session.execute(
                    delete(DocumentChunkRow).where(DocumentChunkRow.doc_id == doc_id)
                )
                session.add_all(
                    [
                        DocumentChunkRow(
                            doc_id=doc_id,
                            chunk_id=str(entry["chunk_id"]),
                            kb_id=row.kb_id,
                            version=int(version or row.version),
                            chunk_index=int(entry.get("chunk_index", 0)),
                            content_hash=str(entry.get("content_hash", "")),
                            section_path=(
                                str(entry["section_path"])
                                if entry.get("section_path") is not None
                                else None
                            ),
                            role=str(entry["role"]) if entry.get("role") else None,
                            embedding_version=(
                                str(entry["embedding_version"])
                                if entry.get("embedding_version")
                                else None
                            ),
                        )
                        for entry in chunk_manifest
                    ]
                )

    async def get_chunk_manifest(self, doc_id: str) -> list[dict]:
        """读取文档当前版本的 chunk 清单（替代已弃用的 JSONB 列）。"""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentChunkRow)
                    .where(DocumentChunkRow.doc_id == doc_id)
                    .order_by(DocumentChunkRow.chunk_index)
                )
            ).scalars().all()
        return [
            {
                "chunk_id": row.chunk_id,
                "chunk_index": row.chunk_index,
                "content_hash": row.content_hash,
                "section_path": row.section_path or "",
                "role": row.role,
                "embedding_version": row.embedding_version,
            }
            for row in rows
        ]

    async def list_ready_manifests(self, *, kb_id: str) -> dict[str, list[dict]]:
        """列出本知识库全部 ready 文档的 chunk 清单：索引对账任务的"应该存在"依据。"""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentChunkRow)
                    .join(DocumentRow, DocumentRow.doc_id == DocumentChunkRow.doc_id)
                    .where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.status == DOC_STATUS_READY,
                    )
                    .order_by(DocumentChunkRow.doc_id, DocumentChunkRow.chunk_index)
                )
            ).scalars().all()
        manifests: dict[str, list[dict]] = {}
        for row in rows:
            manifests.setdefault(row.doc_id, []).append(
                {
                    "chunk_id": row.chunk_id,
                    "chunk_index": row.chunk_index,
                    "content_hash": row.content_hash,
                    "section_path": row.section_path or "",
                    "role": row.role,
                    "embedding_version": row.embedding_version,
                }
            )
        return manifests

    async def mark_document_quarantined(
        self, doc_id: str, *, parse_report: dict, content_hash: str | None = None
    ) -> None:
        """质量门禁未通过：落 quarantined + 质量报告，保留原件供人工修复后重传。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(DocumentRow, doc_id)
            if row is not None:
                row.status = DOC_STATUS_QUARANTINED
                row.parse_report = parse_report
                if content_hash is not None:
                    row.content_hash = content_hash

    async def mark_document_duplicate(
        self, doc_id: str, *, duplicate_of: str, content_hash: str, parse_report: dict
    ) -> None:
        """内容与库内已有文档重复：只登记血缘，不再重复建立索引。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(DocumentRow, doc_id)
            if row is not None:
                row.status = DOC_STATUS_DUPLICATE
                row.duplicate_of = duplicate_of
                row.content_hash = content_hash
                row.parse_report = parse_report

    async def find_document_by_content_hash(
        self, *, kb_id: str, content_hash: str, exclude_doc_id: str | None = None
    ) -> DocumentInfo | None:
        """按内容指纹查同库已有文档（优先返回 ready 状态的规范版本）。"""
        async with self._sessions() as session:
            stmt = (
                select(DocumentRow)
                .where(
                    DocumentRow.kb_id == kb_id,
                    DocumentRow.content_hash == content_hash,
                    DocumentRow.status == DOC_STATUS_READY,
                )
                .order_by(DocumentRow.created_at.asc())
                .limit(1)
            )
            if exclude_doc_id is not None:
                stmt = stmt.where(DocumentRow.doc_id != exclude_doc_id)
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _to_document_info(row) if row is not None else None

    async def mark_document_superseded(self, doc_id: str, *, superseded_by: str) -> None:
        """标记文档已被新版本替代（检索层据此排除旧版本，原件与元数据保留备查）。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(DocumentRow, doc_id)
            if row is not None:
                row.superseded_by = superseded_by

    async def supersede_previous_versions(
        self, *, doc_id: str, kb_id: str, tenant_id: str, source_id: str | None
    ) -> int:
        """Active Version 切换：把同 source_id 的其它 ready 文档标记为被本篇替代。

        调用时机必须在 mark_document_ready 之后：新版本双写完成、检索可见后才
        切换，查询侧只会过滤到"真能用"的旧版本。source_id 为空（上游未提供原始
        ID）时不做替代判定，避免误伤无关文档；返回被替代的文档数。
        """
        if not source_id:
            return 0
        async with self._sessions() as session, session.begin():
            rows = (
                await session.execute(
                    select(DocumentRow).where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.source_id == source_id,
                        DocumentRow.doc_id != doc_id,
                        DocumentRow.status == DOC_STATUS_READY,
                        DocumentRow.superseded_by.is_(None),
                    )
                )
            ).scalars().all()
            for row in rows:
                row.superseded_by = doc_id
            return len(rows)

    async def list_superseded_doc_ids(self, *, kb_id: str, tenant_id: str) -> frozenset[str]:
        """返回本知识库内已被新版本替代的 doc_id 集合，供检索层下推过滤。

        只查 doc_id 一列，代价很低；返回 frozenset 便于直接作为过滤参数复用。
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow.doc_id).where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.superseded_by.is_not(None),
                    )
                )
            ).scalars().all()
        return frozenset(rows)

    async def list_hidden_doc_ids(self, *, kb_id: str, tenant_id: str) -> frozenset[str]:
        """检索层统一排除集合：已被替代 + DELETING（删除中、索引尚未清理完）。

        DELETING 是删除状态机的第一阶段：PG 状态先落库，Milvus/ES 残留索引要等
        删除（或后台重试）完成才消失。这个窗口期内残留索引仍可被召回，必须以
        PG 元数据为权威在查询期直接排除，保证"开始删除即对检索不可见"——
        否则用户会看到一份已请求删除的文档出现在答案引用里。
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow.doc_id).where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        or_(
                            DocumentRow.superseded_by.is_not(None),
                            DocumentRow.status == DOC_STATUS_DELETING,
                        ),
                    )
                )
            ).scalars().all()
        return frozenset(rows)

    async def list_inflight_document_ids(
        self, *, kb_id: str, tenant_id: str
    ) -> list[str]:
        """正在入库（ingesting）的文档 id：索引对账的"在途保护"依据。

        入库链路先写索引、后提交 PG chunk 清单与 ready 状态，这个窗口内新写的
        chunk 在 PG 里查不到，会被对账任务误判为孤儿。对账在存在 ingesting
        文档时暂缓孤儿删除，避免误删正在写入的数据。
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow.doc_id).where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.status == DOC_STATUS_INGESTING,
                    )
                )
            ).scalars().all()
        return list(rows)

    async def list_superseded_document_ids(
        self, *, kb_id: str, tenant_id: str, cutoff: datetime, limit: int = 500
    ) -> list[str]:
        """被新版本替代且超过保留窗口（supersede 时间早于 cutoff）的文档 id。

        延迟清理队列：Active Version 切换后旧版本只标记不删除，保留一个观察窗
        （便于回滚与审计），超窗后由周期任务走统一删除状态机清理。
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow.doc_id)
                    .where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.superseded_by.is_not(None),
                        DocumentRow.updated_at < cutoff,
                    )
                    .limit(limit)
                )
            ).scalars().all()
        return list(rows)

    async def repoint_duplicates(
        self, *, old_carrier_doc_id: str, new_carrier_doc_id: str
    ) -> int:
        """承载者被删除后，把其余 duplicate 文档的血缘指针改指新承载者。

        否则这些文档仍指向已删除的 doc_id，检索层血缘归属与后续 ACL 调和
        都会查不到承载者。返回受影响行数。
        """
        async with self._sessions() as session, session.begin():
            rows = (
                await session.execute(
                    select(DocumentRow).where(
                        DocumentRow.duplicate_of == old_carrier_doc_id
                    )
                )
            ).scalars().all()
            for row in rows:
                row.duplicate_of = new_carrier_doc_id
        return len(rows)

    async def list_lineage_documents(
        self, *, kb_id: str, tenant_id: str, content_hashes: Collection[str]
    ) -> dict[str, list[DocumentInfo]]:
        """按内容指纹批量查"共享同一物理内容"的逻辑文档（ready + duplicate 血缘）。

        每个指纹下的文档按（权威度降序、创建时间升序、doc_id 升序）排序：
        检索层据此为用户挑选"实际授权可见且最具权威"的引用来源，
        入库层据此合并血缘并集 ACL。
        """
        hashes = sorted({h for h in content_hashes if h})
        if not hashes:
            return {}
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow)
                    .where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.content_hash.in_(hashes),
                        DocumentRow.status.in_([DOC_STATUS_READY, DOC_STATUS_DUPLICATE]),
                    )
                    .order_by(
                        DocumentRow.authority_level.desc(),
                        DocumentRow.created_at.asc(),
                        DocumentRow.doc_id.asc(),
                    )
                )
            ).scalars().all()
        lineage: dict[str, list[DocumentInfo]] = {}
        for row in rows:
            if row.content_hash:
                lineage.setdefault(row.content_hash, []).append(_to_document_info(row))
        return lineage

    async def scan_kb_chunks(
        self, *, kb_id: str, batch_size: int = 1000
    ) -> AsyncIterator[list[tuple[str, str, str]]]:
        """主键游标分页流式扫描本库 chunk 清单，产出 (doc_id, chunk_id, role)。

        百万级 chunk 对账不能把全量清单装进内存（旧实现 list_ready_manifests
        一次性构建完整字典）；游标分页每批一个短事务，内存占用恒定。
        只扫描 ready 文档的当前版本清单（与旧实现对账口径一致）；
        role 为空时按 "child" 语义返回空串，由调用方自行兜底。
        """
        last_doc_id = ""
        last_chunk_id = ""
        while True:
            async with self._sessions() as session:
                stmt = (
                    select(
                        DocumentChunkRow.doc_id,
                        DocumentChunkRow.chunk_id,
                        DocumentChunkRow.role,
                    )
                    .join(DocumentRow, DocumentRow.doc_id == DocumentChunkRow.doc_id)
                    .where(
                        DocumentChunkRow.kb_id == kb_id,
                        DocumentRow.status == DOC_STATUS_READY,
                    )
                    .order_by(DocumentChunkRow.doc_id, DocumentChunkRow.chunk_id)
                    .limit(batch_size)
                )
                if last_doc_id:
                    stmt = stmt.where(
                        (DocumentChunkRow.doc_id > last_doc_id)
                        | (
                            (DocumentChunkRow.doc_id == last_doc_id)
                            & (DocumentChunkRow.chunk_id > last_chunk_id)
                        )
                    )
                rows = (await session.execute(stmt)).all()
            if not rows:
                return
            batch = [
                (str(doc_id), str(chunk_id), str(role or ""))
                for doc_id, chunk_id, role in rows
            ]
            yield batch
            last_doc_id, last_chunk_id = batch[-1][0], batch[-1][1]

    async def filter_existing_chunk_ids(
        self, *, kb_id: str, chunk_ids: Collection[str]
    ) -> set[str]:
        """返回给定 chunk_id 中在本库清单里真实存在的子集（对账孤儿检测用）。"""
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return set()
        found: set[str] = set()
        async with self._sessions() as session:
            for start in range(0, len(ids), 500):
                rows = (
                    await session.execute(
                        select(DocumentChunkRow.chunk_id).where(
                            DocumentChunkRow.kb_id == kb_id,
                            DocumentChunkRow.chunk_id.in_(ids[start : start + 500]),
                        )
                    )
                ).scalars().all()
                found.update(str(row) for row in rows)
        return found

    async def mark_chunks_index_status(
        self, chunk_ids: Collection[str], *, status: str
    ) -> int:
        """批量更新 chunk 清单的索引状态（如对账发现 missing）；修复重灌走
        mark_document_ready（清单先删后插）自动回到 indexed。"""
        ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return 0
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                update(DocumentChunkRow)
                .where(DocumentChunkRow.chunk_id.in_(ids))
                .values(index_status=status)
            )
            # UPDATE 的 execute 结果实际是 CursorResult（带 rowcount），类型桩按基类标注
            return int(getattr(result, "rowcount", 0) or 0)

    async def list_document_ids_by_owner(
        self, *, kb_id: str, tenant_id: str, owner: str
    ) -> list[str]:
        """按责任人列出文档 id：支撑"离职/撤回授权后删除该用户全部文档"的合规接口。"""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow.doc_id).where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.owner == owner,
                    )
                )
            ).scalars().all()
        return list(rows)

    async def list_documents_by_source(
        self, *, kb_id: str, tenant_id: str, source_id: str
    ) -> list[tuple[str, str]]:
        """列出某连接器来源已入库的 (doc_id, source_uri)：支撑源端删除同步。

        每次同步把"上次已入库的 URI 集合"与"本次扫描到的 URI 集合"做差，
        缺失的文档走统一删除服务，避免源文件删除后索引里仍可检索到旧内容。
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow.doc_id, DocumentRow.source_uri).where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.source_id == source_id,
                    )
                )
            ).all()
        return [(str(row[0]), str(row[1])) for row in rows]

    async def list_expired_document_ids(
        self, *, kb_id: str, tenant_id: str, cutoff: datetime, limit: int = 500
    ) -> list[str]:
        """采集时间早于 cutoff 的文档 id，供保留期清理任务分批删除。

        只取 collected_at 有值的行（历史数据没有采集时间，交由人工确认，不做自动删除）。
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow.doc_id)
                    .where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.collected_at.is_not(None),
                        DocumentRow.collected_at < cutoff,
                    )
                    .limit(limit)
                )
            ).scalars().all()
        return list(rows)

    async def mark_document_failed(self, doc_id: str) -> None:
        """入库失败置 failed，保留行便于排查与重传覆盖。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(DocumentRow, doc_id)
            if row is not None:
                row.status = DOC_STATUS_FAILED

    async def mark_document_deleting(self, doc_id: str) -> None:
        """删除开始前落 DELETING：失败时保留该状态供重试任务继续推进。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(DocumentRow, doc_id)
            if row is not None:
                row.status = DOC_STATUS_DELETING

    async def list_deleting_document_ids(
        self, *, kb_id: str, tenant_id: str, limit: int = 100
    ) -> list[str]:
        """扫描删除未完成的文档（tombstone），供后台重试任务收口。"""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(DocumentRow.doc_id)
                    .where(
                        DocumentRow.kb_id == kb_id,
                        DocumentRow.tenant_id == tenant_id,
                        DocumentRow.status == DOC_STATUS_DELETING,
                    )
                    .limit(limit)
                )
            ).scalars().all()
        return list(rows)

    async def delete_document(self, doc_id: str) -> None:
        async with self._sessions() as session, session.begin():
            # 先删清单行（无外键依赖，但保持与文档同事务清理，避免孤儿清单）
            await session.execute(
                delete(DocumentChunkRow).where(DocumentChunkRow.doc_id == doc_id)
            )
            row = await session.get(DocumentRow, doc_id)
            if row is not None:
                await session.delete(row)

    async def get_document(self, doc_id: str) -> DocumentInfo | None:
        async with self._sessions() as session:
            row = await session.get(DocumentRow, doc_id)
            return _to_document_info(row) if row is not None else None

    async def list_documents(
        self,
        *,
        kb_id: str,
        tenant_id: str,
        status: str | None = None,
        allowed_groups: frozenset[str] | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[DocumentInfo], int]:
        """按知识库分页列出文档（按更新时间倒序），返回 (items, total)。

        allowed_groups 非 None 时叠加文档级 ACL 过滤：公开（空数组）或与用户组有交集。
        """
        async with self._sessions() as session:
            base = select(DocumentRow).where(
                DocumentRow.kb_id == kb_id, DocumentRow.tenant_id == tenant_id
            )
            if status is not None:
                base = base.where(DocumentRow.status == status)
            if allowed_groups is not None:
                # JSONB：jsonb_array_length 判公开；`?` 逐个判组是否在数组内（字符串绑定参数）
                base = base.where(
                    or_(
                        func.jsonb_array_length(DocumentRow.acl_groups) == 0,
                        *[
                            DocumentRow.acl_groups.op("?")(group)
                            for group in sorted(allowed_groups)
                        ],
                    )
                )
            total = (
                await session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar_one()
            rows = (
                await session.execute(
                    base.order_by(DocumentRow.updated_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).scalars().all()
            return [_to_document_info(row) for row in rows], total

    # ---------------------------------------------------------------- 任务
    async def create_task(
        self,
        task_id: str,
        *,
        kb_id: str,
        tenant_id: str,
        doc_id: str | None = None,
        created_by: str | None = None,
        status: str,
    ) -> None:
        """登记任务行；task_id 复用 Celery 任务 id，便于 API 侧按 id 轮询。"""
        async with self._sessions() as session, session.begin():
            session.add(
                IngestTaskRow(
                    task_id=task_id,
                    kb_id=kb_id,
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    created_by=created_by,
                    status=status,
                )
            )

    async def create_task_with_outbox(
        self,
        task_id: str,
        *,
        kb_id: str,
        tenant_id: str,
        doc_id: str | None = None,
        created_by: str | None = None,
        status: str,
        event_type: str = OUTBOX_EVENT_INGEST,
        payload: dict | None = None,
    ) -> None:
        """同一事务写入任务行与 outbox 事件（v1.4 Transactional Outbox）。

        提交成功即代表"任务已登记、必须被投递"，即使进程在提交后、消息发出前
        崩溃，dispatcher 也能凭 pending 事件补投，不会留下永久 pending 任务。
        """
        event_payload = dict(payload or {})
        event_payload.setdefault("task_id", task_id)
        async with self._sessions() as session, session.begin():
            session.add(
                IngestTaskRow(
                    task_id=task_id,
                    kb_id=kb_id,
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    created_by=created_by,
                    status=status,
                )
            )
            session.add(
                OutboxEventRow(
                    event_id=task_id,
                    event_type=event_type,
                    kb_id=kb_id,
                    tenant_id=tenant_id,
                    payload=event_payload,
                    status=OUTBOX_STATUS_PENDING,
                )
            )

    async def list_pending_outbox(self, limit: int = 100) -> list[OutboxEventInfo]:
        """按创建时间扫描待投递事件；限制批量大小避免单次扫描拖垮 DB。"""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(OutboxEventRow)
                    .where(OutboxEventRow.status == OUTBOX_STATUS_PENDING)
                    .order_by(OutboxEventRow.created_at)
                    .limit(limit)
                )
            ).scalars().all()
            return [
                OutboxEventInfo(
                    event_id=row.event_id,
                    event_type=row.event_type,
                    kb_id=row.kb_id,
                    tenant_id=row.tenant_id,
                    payload=dict(row.payload or {}),
                    attempts=row.attempts,
                )
                for row in rows
            ]

    async def claim_pending_outbox(
        self, *, limit: int = 100, lease_seconds: int = 300
    ) -> list[OutboxEventInfo]:
        """原子领取待投递事件：多 dispatcher 并发也不会重复投递同一条事件。

        实现：同一事务内先用 `FOR UPDATE SKIP LOCKED` 锁定一批 pending 行，
        再统一置 dispatching。SQLite 会忽略 SKIP LOCKED（本地/测试场景没有并发
        dispatcher），PostgreSQL 下则是标准的队列领取语义。
        `dispatched_at` 复用作领取时间戳，超时未完成的事件由下一次领取回收。
        """
        now = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            # lease 回收：dispatcher 崩溃后事件不会永久卡在 dispatching
            await session.execute(
                update(OutboxEventRow)
                .where(
                    OutboxEventRow.status == OUTBOX_STATUS_DISPATCHING,
                    OutboxEventRow.dispatched_at.is_not(None),
                    OutboxEventRow.dispatched_at < now - timedelta(seconds=lease_seconds),
                )
                .values(
                    status=OUTBOX_STATUS_PENDING,
                    dispatched_at=None,
                    last_error="dispatch lease expired",
                )
            )
            rows = (
                await session.execute(
                    select(OutboxEventRow)
                    .where(OutboxEventRow.status == OUTBOX_STATUS_PENDING)
                    .order_by(OutboxEventRow.created_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).scalars().all()
            for row in rows:
                row.status = OUTBOX_STATUS_DISPATCHING
                # 领取只占位，不累加 attempts；尝试次数以实际投递结果为准（崩溃不计数）
                row.dispatched_at = now
            return [
                OutboxEventInfo(
                    event_id=row.event_id,
                    event_type=row.event_type,
                    kb_id=row.kb_id,
                    tenant_id=row.tenant_id,
                    payload=dict(row.payload or {}),
                    attempts=row.attempts,
                )
                for row in rows
            ]

    async def mark_outbox_dispatched(self, event_id: str) -> None:
        """投递成功：置 dispatched 并记录时间，dispatcher 不会再扫到它。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(OutboxEventRow, event_id)
            if row is None:
                return
            row.status = OUTBOX_STATUS_DISPATCHED
            row.dispatched_at = datetime.now(UTC)

    async def mark_outbox_failed(self, event_id: str, *, error: str) -> None:
        """投递失败：累加尝试次数；未到上限退回 pending 等下轮重投。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(OutboxEventRow, event_id)
            if row is None:
                return
            row.attempts += 1
            row.last_error = error[:2000]
            if row.attempts >= OUTBOX_MAX_ATTEMPTS:
                row.status = OUTBOX_STATUS_FAILED
                row.dispatched_at = None
            else:
                row.status = OUTBOX_STATUS_PENDING
                row.dispatched_at = None

    # ------------------------------------------------------- 多轮会话（v2.0）
    async def create_conversation(
        self,
        conversation_id: str,
        *,
        tenant_id: str,
        kb_id: str,
        user_id: str,
        title: str | None = None,
    ) -> None:
        """新建会话容器；标题缺省取首问前 100 字，便于前端列表展示。"""
        async with self._sessions() as session, session.begin():
            session.add(
                ConversationRow(
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    user_id=user_id,
                    title=title[:200] if title else None,
                )
            )

    async def get_conversation(self, conversation_id: str) -> ConversationInfo | None:
        async with self._sessions() as session:
            row = await session.get(ConversationRow, conversation_id)
            if row is None:
                return None
            return ConversationInfo(
                conversation_id=row.conversation_id,
                tenant_id=row.tenant_id,
                kb_id=row.kb_id,
                user_id=row.user_id,
                title=row.title,
            )

    async def append_conversation_messages(
        self,
        conversation_id: str,
        messages: list[ConversationMessage],
    ) -> None:
        """一次事务追加多条消息（用户问 + 助手答），并刷新会话 updated_at。"""
        if not messages:
            return
        async with self._sessions() as session, session.begin():
            for message in messages:
                session.add(
                    ConversationMessageRow(
                        message_id=uuid4().hex,
                        conversation_id=conversation_id,
                        role=message.role,
                        content=message.content,
                    )
                )
            row = await session.get(ConversationRow, conversation_id)
            if row is not None:
                # 显式赋值而非依赖 onupdate：批量写入时 onupdate 在某些方言下不触发
                row.updated_at = datetime.now(UTC)

    async def list_conversation_messages(
        self, conversation_id: str, *, limit: int = 20
    ) -> list[ConversationMessage]:
        """取最近 limit 条消息并按时间正序返回（追问改写只关心近邻上下文）。"""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(ConversationMessageRow)
                    .where(ConversationMessageRow.conversation_id == conversation_id)
                    .order_by(ConversationMessageRow.created_at.desc())
                    .limit(limit)
                )
            ).scalars().all()
        ordered = list(reversed(rows))
        return [
            ConversationMessage(role=row.role, content=row.content) for row in ordered
        ]

    async def delete_conversation(self, conversation_id: str) -> bool:
        """删除会话及其全部消息（用户主动删除路径）；返回是否删到行。

        归属校验在路由层完成（同租户 + 同用户 + 同知识库），这里只做物理删除。
        """
        async with self._sessions() as session, session.begin():
            row = await session.get(ConversationRow, conversation_id)
            if row is None:
                return False
            await session.execute(
                delete(ConversationMessageRow).where(
                    ConversationMessageRow.conversation_id == conversation_id
                )
            )
            await session.delete(row)
            return True

    async def list_expired_conversation_ids(
        self, *, cutoff: datetime, limit: int = 500
    ) -> list[str]:
        """长期未更新（updated_at 早于 cutoff）的会话 id：过期清理任务用。"""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(ConversationRow.conversation_id)
                    .where(ConversationRow.updated_at < cutoff)
                    .limit(limit)
                )
            ).scalars().all()
        return list(rows)

    async def mark_task(
        self,
        task_id: str,
        *,
        status: str,
        doc_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """推进任务状态；进入终态（success/failed）时自动写 finished_at。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(IngestTaskRow, task_id)
            if row is None:
                return
            row.status = status
            if doc_id is not None:
                row.doc_id = doc_id
            if error is not None:
                row.error = error
            if status in (TASK_STATUS_SUCCESS, TASK_STATUS_FAILED):
                row.finished_at = datetime.now(UTC)

    async def get_task(self, task_id: str) -> TaskInfo | None:
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(IngestTaskRow).where(IngestTaskRow.task_id == task_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return TaskInfo(
                task_id=row.task_id,
                doc_id=row.doc_id,
                kb_id=row.kb_id,
                tenant_id=row.tenant_id,
                status=row.status,
                error=row.error,
                created_by=row.created_by,
            )

    # ---------------------------------------------------------------- 审计
    async def write_audit(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        user_id: str,
        action: str,
        doc_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        """追加一条审计日志；写入失败只记日志，绝不阻断主链路（与缓存降级同原则）。"""
        try:
            async with self._sessions() as session, session.begin():
                session.add(
                    AuditLogRow(
                        audit_id=uuid4().hex,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        user_id=user_id,
                        action=action,
                        doc_id=doc_id,
                        detail=json.dumps(detail, ensure_ascii=False) if detail else None,
                    )
                )
        except Exception as exc:  # 审计是旁路，任何异常都不上抛
            logger.warning("审计日志写入失败 action=%s kb=%s：%s", action, kb_id, exc)

    async def list_audit_logs(
        self,
        *,
        kb_id: str,
        tenant_id: str,
        action: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[AuditLogInfo], int]:
        """按知识库分页查询审计日志（按时间倒序），返回 (items, total)。"""
        async with self._sessions() as session:
            base = select(AuditLogRow).where(
                AuditLogRow.kb_id == kb_id, AuditLogRow.tenant_id == tenant_id
            )
            if action is not None:
                base = base.where(AuditLogRow.action == action)
            total = (
                await session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar_one()
            rows = (
                await session.execute(
                    base.order_by(AuditLogRow.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).scalars().all()
            return [_to_audit_log_info(row) for row in rows], total

    # ---------------------------------------------------------------- 死信
    async def put_dead_letter(
        self,
        *,
        event_id: str,
        kb_id: str,
        tenant_id: str,
        doc_id: str | None,
        payload: dict,
        error: str,
    ) -> None:
        """落一条死信；event_id（=任务 id）幂等覆盖，同一任务重试多次只留最新一条。"""
        async with self._sessions() as session, session.begin():
            await session.merge(
                DeadLetterEventRow(
                    event_id=event_id,
                    kb_id=kb_id,
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    payload=payload,
                    error=error[-2000:],
                )
            )

    async def list_dead_letters(
        self, *, kb_id: str | None = None, offset: int = 0, limit: int = 50
    ) -> tuple[list[DeadLetterInfo], int]:
        """分页列出死信（按时间倒序）；kb_id 为空时跨知识库列出（运维视角）。"""
        async with self._sessions() as session:
            base = select(DeadLetterEventRow)
            if kb_id is not None:
                base = base.where(DeadLetterEventRow.kb_id == kb_id)
            total = (
                await session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar_one()
            rows = (
                await session.execute(
                    base.order_by(DeadLetterEventRow.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).scalars().all()
            items = [
                DeadLetterInfo(
                    event_id=row.event_id,
                    kb_id=row.kb_id,
                    tenant_id=row.tenant_id,
                    doc_id=row.doc_id,
                    payload=dict(row.payload or {}),
                    error=row.error,
                    created_at=row.created_at,
                )
                for row in rows
            ]
            return items, total

    async def delete_dead_letter(self, event_id: str) -> None:
        """死信被成功重放/人工确认后移除。"""
        async with self._sessions() as session, session.begin():
            row = await session.get(DeadLetterEventRow, event_id)
            if row is not None:
                await session.delete(row)

    async def get_dead_letter(self, event_id: str) -> DeadLetterInfo | None:
        """按 event_id 取单条死信（重放前的存在性与归属校验用）。"""
        async with self._sessions() as session:
            row = await session.get(DeadLetterEventRow, event_id)
            if row is None:
                return None
            return DeadLetterInfo(
                event_id=row.event_id,
                kb_id=row.kb_id,
                tenant_id=row.tenant_id,
                doc_id=row.doc_id,
                payload=dict(row.payload or {}),
                error=row.error,
                created_at=row.created_at,
            )
