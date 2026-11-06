"""文档入库与删除。重传同一 source_uri 即视为覆盖重建（doc_id 稳定不变）。

同步端点在请求内完成全部入库；异步端点（:async）投递 Celery 任务后立即返回 202，
任务进度走 GET /tasks/{task_id}。两种路径都会把文档状态写入 PG 元数据（documents 表），
pipeline 本身不依赖 DB，元数据写入只发生在本层与 worker 任务层。
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import (
    ContributorUserDep,
    CurrentUserDep,
    DBRepoDep,
    KBConfigDep,
    ObjectStoreDep,
    PipelineDep,
    rate_limit_dep,
)
from app.api.schemas import (
    AsyncIngestResponse,
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentSummary,
    OwnerPurgeResponse,
)
from app.core.errors import DocumentNotFoundError, DocumentParseError, DocumentQualityError
from app.core.logging import get_logger
from app.core.settings import get_settings
from app.db.models import (
    AUDIT_ACTION_DOCUMENT_UPLOAD,
    AUDIT_ACTION_DOCUMENT_UPLOAD_ASYNC,
    TASK_STATUS_PENDING,
)
from app.db.repo import acl_visible
from app.domain.governance import (
    compute_config_hash,
    normalize_source_type,
    resolve_version,
)
from app.ingestion.governance import decide as governance_decide
from app.ingestion.loader import (
    DocumentLoader,
    compute_doc_id,
    normalize_source_uri,
)
from app.orchestration.deletion import DocumentDeletionService
from app.tasks.ingest import ingest_document_task

logger = get_logger(__name__)
router = APIRouter(prefix="/kbs", tags=["documents"])

_settings = get_settings()
_UPLOAD_RATE_LIMIT = rate_limit_dep("upload", limit=_settings.rate_limit_upload_rpm)

MAX_FILE_BYTES = 32 * 1024 * 1024


def _validate_upload(data: bytes) -> None:
    """入库前的体积/空文件校验；超限按 400 级业务错误（DocumentParseError）拒绝。"""
    if not data:
        raise DocumentParseError("上传文件为空")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentParseError("单个文件不能超过 32MB")


_UPLOAD_CHUNK_BYTES = 1024 * 1024


async def _read_upload_chunked(file: UploadFile) -> bytes:
    """分块读取上传体：超过 MAX_FILE_BYTES 立即中止，不必读完整请求体。

    防止客户端发送超大 body 时先把全部内容缓冲进内存才拒绝。
    """
    buf = io.BytesIO()
    while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
        buf.write(chunk)
        if buf.tell() > MAX_FILE_BYTES:
            raise DocumentParseError("单个文件不能超过 32MB")
    return buf.getvalue()


async def _spool_upload_to_temp(file: UploadFile) -> Path:
    """把上传分块落到临时文件（而非内存），供异步路径流式写入对象存储。

    空文件/超限在读取阶段即中止；任何失败都会清理临时文件，不泄漏磁盘。
    """
    fd, tmp = tempfile.mkstemp(prefix="rag-upload-")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as fh:
            size = 0
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise DocumentParseError("单个文件不能超过 32MB")
                fh.write(chunk)
        if size == 0:
            raise DocumentParseError("上传文件为空")
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def _parse_acl_groups(raw: str) -> list[str] | None:
    """表单里 acl_groups 是逗号分隔字符串；空串/纯空白表示公开（None）。

    返回 None 而非空列表很关键：None 在检索层表示"无 ACL 限制"，
    空列表语义上是"谁都不可见"，二者不可混淆。
    """
    groups = [item.strip() for item in raw.split(",") if item.strip()]
    return groups or None


def _governance_fields(
    *,
    source_uri: str,
    suffix: str | None,
    source_id: str,
    source_type: str,
    owner: str,
    source_version: str,
    authority_level: int,
    valid_from: datetime | None,
    valid_to: datetime | None,
) -> dict[str, Any]:
    """校验并整理治理字段；非法取值转成 400 级业务错误而不是 500。

    source_type 为空时按后缀推断；有效时间窗口必须合法（start <= end）。
    """
    try:
        resolved_type = normalize_source_type(source_type, source_uri=source_uri, suffix=suffix)
    except ValueError as exc:
        raise DocumentParseError(str(exc)) from exc
    if valid_from and valid_to and valid_to < valid_from:
        raise DocumentParseError("valid_to 不能早于 valid_from")
    return {
        "source_id": source_id or None,
        "source_type": resolved_type,
        "owner": owner or None,
        "source_version": source_version or None,
        "authority_level": authority_level,
        "valid_from": valid_from,
        "valid_to": valid_to,
    }


def _parse_config_hash(kb: Any) -> str:
    """把切块参数 + Embedding 版本打成指纹，随文档留存以支持解析结果可复现。

    Embedding 版本必须进指纹：换模型（即使维度相同）后旧向量语义不同，
    不能因为内容/ACL 未变就直接复用旧索引。
    """
    return compute_config_hash(
        {**kb.chunking.model_dump(), "embedding_version": _settings.embedding_version}
    )


@router.get(
    "/{kb_key}/documents",
    response_model=DocumentListResponse,
    summary="分页列出知识库文档（按更新时间倒序）",
)
async def list_documents(
    kb: KBConfigDep,
    repo: DBRepoDep,
    user: CurrentUserDep,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
    status: Annotated[
        str | None,
        Query(pattern="^(ingesting|ready|failed|quarantined|duplicate)$"),
    ] = None,
) -> DocumentListResponse:
    """分页列出文档；allowed_groups 透传到 SQL 层做文档级 ACL 过滤。"""
    items, total = await repo.list_documents(
        kb_id=kb.kb_id,
        tenant_id=kb.tenant_id,
        status=status,
        # ACL 过滤：用户只能看到 acl 与其组交集非空（或公开）的文档
        allowed_groups=user.allowed_groups,
        offset=(page - 1) * size,
        limit=size,
    )
    return DocumentListResponse(
        total=total,
        page=page,
        size=size,
        items=[DocumentSummary.from_info(item) for item in items],
    )


@router.get(
    "/{kb_key}/documents/{doc_id}",
    response_model=DocumentSummary,
    summary="查询单个文档的元数据",
)
async def get_document(
    doc_id: str, kb: KBConfigDep, repo: DBRepoDep, user: CurrentUserDep
) -> DocumentSummary:
    """查询单个文档元数据；越库/越租户/越 ACL 一律按 404 处理。"""
    info = await repo.get_document(doc_id)
    # 无权访问的文档按 404 处理，不暴露文档存在性
    if (
        info is None
        or info.kb_id != kb.kb_id
        or info.tenant_id != kb.tenant_id
        or not acl_visible(info.acl_groups, user.allowed_groups)
    ):
        raise DocumentNotFoundError(f"文档不存在：{doc_id}", details={"doc_id": doc_id})
    return DocumentSummary.from_info(info)


@router.post(
    "/{kb_key}/documents",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档并同步完成解析、切块、向量化与双写",
    dependencies=[_UPLOAD_RATE_LIMIT],
)
async def upload_document(
    kb: KBConfigDep,
    pipeline: PipelineDep,
    repo: DBRepoDep,
    user: ContributorUserDep,
    file: Annotated[UploadFile, File(...)],
    title: Annotated[str, Form()] = "",
    source_uri: Annotated[str | None, Form()] = None,
    acl_groups: Annotated[str, Form(description="逗号分隔的可见组，留空为公开")] = "",
    source_id: Annotated[str, Form(description="上游系统原始 ID，如工单号")] = "",
    source_type: Annotated[str, Form(description="wiki/ticket/api/table/file 等")] = "",
    owner: Annotated[str, Form(description="内容责任人")] = "",
    source_version: Annotated[str, Form(description="上游版本串，如 v2.3")] = "",
    authority_level: Annotated[int, Form(ge=0, le=100, description="来源权威级别 0-100")] = 0,
    valid_from: Annotated[datetime | None, Form(description="生效时间")] = None,
    valid_to: Annotated[datetime | None, Form(description="失效时间")] = None,
) -> DocumentIngestResponse:
    """同步上传入库：解析 → 治理裁决 → 质量门禁 → 切块 → 向量化 → 双写。

    治理裁决三选一（都在写索引前完成，避免做无用功）：
    1. 内容未变化且已 ready → 直接复用，不重复索引；
    2. 内容与库内另一篇完全相同 → 只登记 duplicate_of 血缘，不重复索引；
    3. 内容变化 → 自动升版本，整篇覆盖重建。
    """
    data = await _read_upload_chunked(file)
    _validate_upload(data)

    filename = file.filename or "document.txt"
    suffix = Path(filename).suffix
    acl = _parse_acl_groups(acl_groups)
    resolved_source_uri = source_uri or f"kb://{kb.kb_code}/{filename}"
    normalized_uri = normalize_source_uri(resolved_source_uri)
    doc_id = compute_doc_id(kb.tenant_id, kb.kb_id, resolved_source_uri)
    governance = _governance_fields(
        source_uri=normalized_uri,
        suffix=suffix,
        source_id=source_id,
        source_type=source_type,
        owner=owner,
        source_version=source_version,
        authority_level=authority_level,
        valid_from=valid_from,
        valid_to=valid_to,
    )

    # 解析一次即可拿到正文与内容指纹，后续去重/版本裁决都基于它，不重复解析
    loader = DocumentLoader(
        tenant_id=kb.tenant_id,
        kb_id=kb.kb_id,
        redact_pii=get_settings().privacy_redaction_enabled,
    )
    document = await run_in_threadpool(
        loader.load_bytes,
        data,
        source_uri=resolved_source_uri,
        title=title or Path(filename).stem,
        suffix=suffix,
    )

    existing = await repo.get_document(doc_id)
    # 业务语义（清单第 5 项）：同一 source_uri 即同一文档。
    # 内容变化由治理裁决走"升版本覆盖重建"，不再另造 doc_id——
    # (kb_id, source_uri) 有唯一约束，同 uri 多 doc_id 会直接违反约束；
    # 不同上传主体/不同来源需要独立身份时应显式指定不同的 source_uri。

    # 场景 1：同一来源、内容未变、ACL/治理未变、上次已成功 → 幂等复用
    # 使用与异步 worker 完全相同的裁决函数，避免两边判断漂移导致权限/治理更新失效
    decision = governance_decide(
        new_content_hash=document.content_hash,
        new_acl=acl,
        new_source_id=governance["source_id"],
        new_source_type=governance["source_type"],
        new_owner=governance["owner"],
        new_source_version=governance["source_version"],
        new_authority_level=governance["authority_level"],
        existing_status=(existing.status if existing else None),
        existing_content_hash=(existing.content_hash if existing else None),
        existing_acl=(existing.acl_groups if existing else None),
        existing_source_id=(existing.source_id if existing else None),
        existing_source_type=(existing.source_type if existing else None),
        existing_owner=(existing.owner if existing else None),
        existing_source_version=(existing.source_version if existing else None),
        existing_authority_level=(existing.authority_level if existing else 0),
        new_valid_from=governance["valid_from"],
        new_valid_to=governance["valid_to"],
        existing_valid_from=(existing.valid_from if existing else None),
        existing_valid_to=(existing.valid_to if existing else None),
        new_config_hash=_parse_config_hash(kb),
        existing_config_hash=(existing.parse_config_hash if existing else None),
    )
    if decision.decision == "reuse" and existing is not None:
        return DocumentIngestResponse(
            doc_id=existing.doc_id,
            source_uri=existing.source_uri,
            title=existing.title,
            chunk_count=existing.chunk_count,
            collection=existing.collection,
            index=existing.es_index,
            raw_object_key=existing.raw_object_key,
            version=existing.version,
            content_hash=document.content_hash,
            reused=True,
            quality=existing.parse_report,
        )

    # 场景 2：同一知识库内已有另一篇内容完全相同的文档 → 只登记血缘
    duplicate = await repo.find_document_by_content_hash(
        kb_id=kb.kb_id, content_hash=document.content_hash, exclude_doc_id=doc_id
    )
    if duplicate is not None:
        await repo.upsert_document_ingesting(
            doc_id=doc_id,
            kb_id=kb.kb_id,
            tenant_id=kb.tenant_id,
            source_uri=normalized_uri,
            title=title or Path(filename).stem,
            version=1,
            acl_groups=acl,
            content_hash=document.content_hash,
            parser_name=document.parser_name,
            parser_version=document.parser_version,
            parse_config_hash=_parse_config_hash(kb),
            **governance,
        )
        await repo.mark_document_duplicate(
            doc_id,
            duplicate_of=duplicate.doc_id,
            content_hash=document.content_hash,
            parse_report={},
        )
        # 血缘并集 ACL 调和（第 9 项）：共享内容的任一来源 ACL 更宽松时，
        # 刷新承载者物理索引的可见性并集，避免该来源用户漏招；失败不阻断登记
        await run_in_threadpool(
            pipeline.reconcile_lineage_acl,
            kb,
            content_hash=document.content_hash,
            carrier_doc_id=duplicate.doc_id,
        )
        await repo.write_audit(
            tenant_id=kb.tenant_id,
            kb_id=kb.kb_id,
            user_id=user.user_id,
            action=AUDIT_ACTION_DOCUMENT_UPLOAD,
            doc_id=doc_id,
            detail={"filename": filename, "duplicate_of": duplicate.doc_id},
        )
        return DocumentIngestResponse(
            doc_id=doc_id,
            source_uri=normalized_uri,
            title=title or Path(filename).stem,
            chunk_count=duplicate.chunk_count,
            collection=duplicate.collection,
            index=duplicate.es_index,
            raw_object_key=duplicate.raw_object_key,
            version=1,
            content_hash=document.content_hash,
            reused=True,
            duplicate_of=duplicate.doc_id,
        )

    # 场景 3：内容变化 → 自动升版本；同内容的重试沿用原版本，保证 chunk_id 稳定
    version = resolve_version(
        existing.content_hash if existing else None,
        existing.version if existing else 1,
        document.content_hash,
    )
    await repo.upsert_document_ingesting(
        doc_id=doc_id,
        kb_id=kb.kb_id,
        tenant_id=kb.tenant_id,
        source_uri=normalized_uri,
        title=title or Path(filename).stem,
        version=version,
        acl_groups=acl,
        content_hash=document.content_hash,
        parser_name=document.parser_name,
        parser_version=document.parser_version,
        parse_config_hash=_parse_config_hash(kb),
        **governance,
    )
    try:
        # 入库链路是同步阻塞 IO（模型与存储客户端均为同步），放到线程池执行
        result = await run_in_threadpool(
            pipeline.ingest_document,
            kb,
            document,
            raw_bytes=data,
            version=version,
            acl_groups=acl,
            governance=governance,
            old_manifest=(
                await repo.get_chunk_manifest(existing.doc_id) if existing is not None else None
            ),
        )
    except DocumentQualityError as exc:
        # 质量门禁未通过：隔离而不是失败，保留行与原件供修复后重传
        await repo.mark_document_quarantined(
            doc_id,
            parse_report=exc.details,
            content_hash=document.content_hash,
        )
        raise
    except Exception:
        await repo.mark_document_failed(doc_id)
        raise
    await repo.mark_document_ready(
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
    # Active Version 切换：新版本 ready（双写完成、检索可见）后才把同 source_id
    # 的旧文档标记为 superseded，检索层据此排除旧版本。切换失败不影响本次入库结果
    try:
        superseded_count = await repo.supersede_previous_versions(
            doc_id=result.doc_id,
            kb_id=kb.kb_id,
            tenant_id=kb.tenant_id,
            source_id=governance["source_id"],
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
    await repo.write_audit(
        tenant_id=kb.tenant_id,
        kb_id=kb.kb_id,
        user_id=user.user_id,
        action=AUDIT_ACTION_DOCUMENT_UPLOAD,
        doc_id=result.doc_id,
        detail={
            "filename": filename,
            "title": result.title,
            "chunk_count": result.chunk_count,
            "acl_groups": acl or [],
            "version": result.version,
            "content_hash": result.content_hash,
            "quality": result.quality,
        },
    )
    return DocumentIngestResponse(
        doc_id=result.doc_id,
        source_uri=result.source_uri,
        title=result.title,
        chunk_count=result.chunk_count,
        collection=result.collection,
        index=result.index,
        raw_object_key=result.raw_object_key,
        version=result.version,
        content_hash=result.content_hash,
        quality=result.quality,
    )


@router.post(
    "/{kb_key}/documents:async",
    response_model=AsyncIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="上传文档并异步入库（Celery worker 执行），返回任务 ID",
    dependencies=[_UPLOAD_RATE_LIMIT],
)
async def upload_document_async(
    kb: KBConfigDep,
    repo: DBRepoDep,
    object_store: ObjectStoreDep,
    user: ContributorUserDep,
    file: Annotated[UploadFile, File(...)],
    title: Annotated[str, Form()] = "",
    source_uri: Annotated[str | None, Form()] = None,
    acl_groups: Annotated[str, Form(description="逗号分隔的可见组，留空为公开")] = "",
    source_id: Annotated[str, Form(description="上游系统原始 ID，如工单号")] = "",
    source_type: Annotated[str, Form(description="wiki/ticket/api/table/file 等")] = "",
    owner: Annotated[str, Form(description="内容责任人")] = "",
    source_version: Annotated[str, Form(description="上游版本串，如 v2.3")] = "",
    authority_level: Annotated[int, Form(ge=0, le=100, description="来源权威级别 0-100")] = 0,
    valid_from: Annotated[datetime | None, Form(description="生效时间")] = None,
    valid_to: Annotated[datetime | None, Form(description="失效时间")] = None,
) -> AsyncIngestResponse:
    """异步上传入库：登记任务 → 投递 Celery → 立即返回 202。

    文件传输策略按大小分流：≤4MB 直接塞进 Celery 消息（省一次 S3 往返），
    更大的先 PUT 到 S3 暂存、任务内拉回（避免撑爆 Redis 消息体）。
    治理裁决（去重/版本/质量门禁）在 worker 内完成——解析本身就是重活，
    放在 API 进程会抵消异步入库的意义。
    """
    filename = file.filename or "document.txt"
    suffix = Path(filename).suffix
    resolved_source_uri = source_uri or f"kb://{kb.kb_code}/{filename}"
    # 先做一次参数校验，把非法 source_type/时间窗挡在投递之前
    governance = _governance_fields(
        source_uri=normalize_source_uri(resolved_source_uri),
        suffix=suffix,
        source_id=source_id,
        source_type=source_type,
        owner=owner,
        source_version=source_version,
        authority_level=authority_level,
        valid_from=valid_from,
        valid_to=valid_to,
    )
    # task_id 由服务端生成（而非 Celery 自动生成），便于先落库再投递时对齐
    task_id = uuid4().hex
    task_kwargs: dict[str, Any] = {
        "kb_key": kb.kb_code,
        # v1.4：异步任务一律先落对象存储，消息与 Outbox 事件只带 staging_key，
        # 避免 broker 消息携带二进制（同时满足 JSON 序列化要求）
        "payload": None,
        "staging_key": None,
        "filename": filename,
        "title": title,
        "source_uri": source_uri,
        "acl_groups": _parse_acl_groups(acl_groups),
        "source_id": governance["source_id"],
        "source_type": governance["source_type"],
        "owner": governance["owner"],
        "source_version": governance["source_version"],
        "authority_level": governance["authority_level"],
        # Celery 使用 JSON 序列化：时间统一转 ISO 字符串，任务内再解析
        "valid_from": governance["valid_from"].isoformat() if valid_from else None,
        "valid_to": governance["valid_to"].isoformat() if valid_to else None,
    }
    staging_key = f"staging/{task_id}"
    # 分块落临时文件再流式写入对象存储：API 进程内存占用与文件大小解耦；
    # 空文件/超限在读取阶段即中止，无论成败临时文件都会被清理
    tmp_path = await _spool_upload_to_temp(file)
    try:
        await run_in_threadpool(
            object_store.put_fileobj,
            staging_key,
            tmp_path,
            content_type="application/octet-stream",
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    task_kwargs["staging_key"] = staging_key

    # 任务行与 Outbox 事件同事务写入：提交即代表"必须被投递"
    await repo.create_task_with_outbox(
        task_id,
        kb_id=kb.kb_id,
        tenant_id=kb.tenant_id,
        created_by=user.user_id,
        status=TASK_STATUS_PENDING,
        payload=task_kwargs,
    )

    try:
        # 线程池内投递：即便误开 eager 模式，任务也不会嵌进应用事件循环
        await run_in_threadpool(
            ingest_document_task.apply_async, kwargs=task_kwargs, task_id=task_id
        )
    except Exception as exc:
        # 即时投递失败不动 staging（Outbox 仍为 pending，dispatcher 会补投）
        await repo.mark_outbox_failed(task_id, error=str(exc))
        logger.warning("异步入库即时投递失败，等待 Outbox 补投 task=%s", task_id)
        raise
    else:
        await repo.mark_outbox_dispatched(task_id)

    await repo.write_audit(
        tenant_id=kb.tenant_id,
        kb_id=kb.kb_id,
        user_id=user.user_id,
        action=AUDIT_ACTION_DOCUMENT_UPLOAD_ASYNC,
        detail={"filename": filename, "title": title, "task_id": task_id},
    )
    return AsyncIngestResponse(task_id=task_id, status=TASK_STATUS_PENDING)


@router.delete(
    "/{kb_key}/documents/{doc_id}",
    response_model=DocumentDeleteResponse,
    summary="删除文档：向量 + ES + 原始文件 + 元数据，全部清理",
)
async def delete_document(
    doc_id: str,
    kb: KBConfigDep,
    pipeline: PipelineDep,
    repo: DBRepoDep,
    object_store: ObjectStoreDep,
    user: ContributorUserDep,
) -> DocumentDeleteResponse:
    """删除状态机：DELETING → 清索引 → 清原文 → 删 PG → 审计。

    原始文件删除失败时保留 PG 元数据（不让原件失去追踪），文档留在 DELETING，
    由 `rag.resume_pending_deletions` 周期重试。
    """
    service = DocumentDeletionService(
        pipeline=pipeline, repo=repo, object_store=object_store
    )
    outcome = await service.delete(kb, doc_id, user_id=user.user_id)
    return DocumentDeleteResponse(
        doc_id=outcome.doc_id,
        deleted_vectors=outcome.deleted_vectors,
        deleted_keywords=outcome.deleted_keywords,
    )


@router.delete(
    "/{kb_key}/owners/{owner}/documents",
    response_model=OwnerPurgeResponse,
    summary="按责任人删除其全部文档（离职/授权撤回/数据主体删除）",
)
async def delete_owner_documents(
    owner: str,
    kb: KBConfigDep,
    pipeline: PipelineDep,
    repo: DBRepoDep,
    object_store: ObjectStoreDep,
    user: ContributorUserDep,
) -> OwnerPurgeResponse:
    """合规删除接口：批量清理某责任人名下的全部文档及其索引。

    与单篇删除共用 DocumentDeletionService：原始文件也会被清理，单篇失败留在
    DELETING 供重试，不会出现"批量删除漏删原件"的行为差异。
    """
    service = DocumentDeletionService(
        pipeline=pipeline, repo=repo, object_store=object_store
    )
    outcome = await service.purge_owner(kb, owner, user_id=user.user_id)
    return OwnerPurgeResponse(
        owner=owner,
        deleted_documents=outcome.deleted_documents,
        deleted_vectors=outcome.deleted_vectors,
        deleted_keywords=outcome.deleted_keywords,
    )
