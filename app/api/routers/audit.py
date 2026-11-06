"""审计日志查询：按知识库分页返回审计记录（时间倒序）。

审计写入分散在各业务端点（上传/删除/问答），此处只提供只读查询。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import AuditorUserDep, DBRepoDep, KBConfigDep
from app.api.schemas import AuditLogEntry, AuditLogListResponse

router = APIRouter(prefix="/kbs", tags=["audit"])


@router.get(
    "/{kb_key}/audit-logs",
    response_model=AuditLogListResponse,
    summary="分页查询知识库审计日志（按时间倒序）",
)
async def list_audit_logs(
    # KBConfigDep 依赖内部已完成"库存在 + 已启用 + 当前用户可访问"三项校验
    kb: KBConfigDep,
    repo: DBRepoDep,
    user: AuditorUserDep,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
    # action 白名单过滤：正则限制取值范围，防止任意字符串进入 SQL 查询条件
    action: Annotated[
        str | None,
        Query(
            pattern="^(document\\.upload|document\\.upload_async|document\\.delete"
            "|document\\.delete_failed|conflict_doc_id|chat\\.query"
            "|kb\\.create|kb\\.update|kb\\.delete)$"
        ),
    ] = None,
) -> AuditLogListResponse:
    """分页查询指定知识库的审计日志；租户隔离沿用了 kb 解析出的 tenant_id。"""
    items, total = await repo.list_audit_logs(
        kb_id=kb.kb_id,
        tenant_id=kb.tenant_id,
        action=action,
        offset=(page - 1) * size,
        limit=size,
    )
    return AuditLogListResponse(
        total=total,
        page=page,
        size=size,
        items=[AuditLogEntry.from_info(item) for item in items],
    )
