"""知识库管理：列表 / 详情 / 运行时创建、更新、删除。

配置以 YAML 落在 KB_CONFIG_DIR，写操作先落盘再热加载内存注册表：
- 创建：文件名固定为 {kb_code}.yaml，kb_code / kb_id 全局唯一；
- 更新：kb_code / kb_id / tenant_id 不可变（集合与索引命名依赖 kb_code，租户变更会串数据）；
- 删除：库内仍有文档时拒绝（409），需先删除文档，避免留下孤儿向量/元数据。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.deps import (
    AdminUserDep,
    CacheDep,
    ConfigSyncDep,
    CurrentUserDep,
    DBRepoDep,
    KBConfigDep,
    KBWriteConfigDep,
    RegistryDep,
    RuntimeRegistryDep,
    SettingsDep,
    can_access_kb,
)
from app.api.schemas import (
    KBCreateRequest,
    KBDeleteResponse,
    KBSummary,
    KBUpdateRequest,
    KBWriteRequest,
)
from app.core.errors import (
    KBConfigError,
    KBConflictError,
    KBInUseError,
    KBNotFoundError,
    KBValidationError,
    PermissionDeniedError,
)
from app.db.models import (
    AUDIT_ACTION_KB_CREATE,
    AUDIT_ACTION_KB_DELETE,
    AUDIT_ACTION_KB_UPDATE,
)
from app.domain.kb_config import (
    KBConfig,
    KBRegistry,
    delete_kb_config,
    discover_kb_configs,
    find_kb_config_path,
    save_kb_config,
    validation_details,
)

router = APIRouter(prefix="/kbs", tags=["knowledge-bases"])


def _build_config(
    payload: KBWriteRequest, *, kb_id: str, kb_code: str, tenant_id: str
) -> KBConfig:
    """把 API 入参组装成内部 KBConfig 并跑一遍完整校验；
    pydantic ValidationError 统一翻译成业务错误 KBValidationError（400 级）。"""
    data = {
        "kb_id": kb_id,
        "kb_code": kb_code,
        "name": payload.name,
        "description": payload.description,
        "sample_questions": payload.sample_questions,
        "tenant_id": tenant_id,
        "chunking": payload.chunking.model_dump(),
        "embedding": payload.embedding.model_dump(),
        "retrieval": payload.retrieval.model_dump(),
        "enabled": payload.enabled,
    }
    try:
        return KBConfig.model_validate(data)
    except ValidationError as exc:
        raise KBValidationError(
            "知识库配置校验失败", details={"errors": validation_details(exc)}
        ) from exc


def _load_configs(directory: Path) -> list[KBConfig]:
    """重扫配置目录（写操作前以磁盘为准，避免多副本部署下的内存脏读）。"""
    return discover_kb_configs(directory)


def _swap_registry(request: Request, configs: list[KBConfig]) -> None:
    """整体替换内存注册表（按 kb_code 排序保证遍历顺序稳定），实现配置热加载。"""
    request.app.state.kb_registry = KBRegistry(sorted(configs, key=lambda c: c.kb_code))


def _check_tenant(payload: KBWriteRequest, tenant_id: str) -> None:
    """跨租户防护：显式指定 tenant_id 时必须与当前用户租户一致。"""
    if payload.tenant_id is not None and payload.tenant_id != tenant_id:
        raise PermissionDeniedError(
            "不能在其他租户下管理知识库",
            details={"tenant_id": payload.tenant_id, "user_tenant_id": tenant_id},
        )


@router.get("", response_model=list[KBSummary], summary="列出当前用户可见的知识库")
def list_kbs(user: CurrentUserDep, registry: RegistryDep) -> list[KBSummary]:
    """列出当前用户可见的知识库：只含已启用库，且逐一过 can_access_kb 过滤。"""
    return [
        KBSummary.from_config(kb)
        for kb in registry.list_enabled()
        if can_access_kb(kb, user)
    ]


@router.post(
    "",
    response_model=KBSummary,
    status_code=status.HTTP_201_CREATED,
    summary="创建知识库（写入 YAML 配置并热加载）",
)
async def create_kb(
    payload: KBCreateRequest,
    request: Request,
    settings: SettingsDep,
    user: AdminUserDep,
    repo: DBRepoDep,
    config_sync: ConfigSyncDep,
) -> KBSummary:
    """创建知识库：三重唯一性检查（kb_code / kb_id / 配置文件名）→ 落盘 → 热加载。"""
    _check_tenant(payload, user.tenant_id)
    config = _build_config(
        payload,
        kb_id=payload.kb_id or f"kb-{payload.kb_code}",
        kb_code=payload.kb_code,
        tenant_id=user.tenant_id,
    )

    # 以磁盘为准做冲突判断，避免其他副本刚写入的配置被覆盖
    configs = await run_in_threadpool(_load_configs, settings.kb_config_dir)
    if any(item.kb_code == config.kb_code for item in configs):
        raise KBConflictError(
            f"知识库标识已存在：{config.kb_code}", details={"kb_code": config.kb_code}
        )
    if any(item.kb_id == config.kb_id for item in configs):
        raise KBConflictError(
            f"知识库 ID 已存在：{config.kb_id}", details={"kb_id": config.kb_id}
        )
    target = settings.kb_config_dir / f"{config.kb_code}.yaml"
    if target.exists():
        raise KBConflictError(
            f"知识库配置文件已存在：{target.name}", details={"path": str(target)}
        )

    await run_in_threadpool(save_kb_config, settings.kb_config_dir, config)
    _swap_registry(request, [*configs, config])
    # 广播失效：其他副本下一个检查周期即可看到新库（Redis 不可用时为无操作）
    if config_sync is not None:
        config_sync.publish(config.kb_code)
    await repo.write_audit(
        tenant_id=config.tenant_id,
        kb_id=config.kb_id,
        user_id=user.user_id,
        action=AUDIT_ACTION_KB_CREATE,
        detail={"kb_code": config.kb_code, "name": config.name},
    )
    return KBSummary.from_config(config)


@router.put(
    "/{kb_key}",
    response_model=KBSummary,
    summary="更新知识库配置（kb_code / kb_id / 租户不可变）",
)
async def update_kb(
    kb_key: str,
    payload: KBUpdateRequest,
    request: Request,
    user: AdminUserDep,
    kb: KBWriteConfigDep,
    settings: SettingsDep,
    runtime_registry: RuntimeRegistryDep,
    cache: CacheDep,
    repo: DBRepoDep,
    config_sync: ConfigSyncDep,
) -> KBSummary:
    """更新知识库：先落盘再热加载；同时丢弃旧运行时客户端并失效检索缓存，
    保证新配置（切块/检索参数/Embedding 模型）立即生效。"""
    configs = await run_in_threadpool(_load_configs, settings.kb_config_dir)
    current = next((item for item in configs if item.kb_code == kb.kb_code), None)
    if current is None:
        raise KBNotFoundError(f"知识库不存在：{kb_key}", details={"kb": kb_key})
    _check_tenant(payload, current.tenant_id)

    updated = _build_config(
        payload, kb_id=current.kb_id, kb_code=current.kb_code, tenant_id=current.tenant_id
    )
    # 连接器由 YAML 人工维护、不在更新接口入参里；更新其余字段时原样保留，
    # 避免一次常规配置修改把已接好的外部知识源悄悄清空
    updated = updated.model_copy(
        update={
            "connectors": current.connectors,
            "doc_acl_enabled": current.doc_acl_enabled,
        }
    )
    path = await run_in_threadpool(
        find_kb_config_path, settings.kb_config_dir, current.kb_code
    )
    if path is None:
        raise KBConfigError(
            f"知识库配置文件缺失：{current.kb_code}", details={"kb_code": current.kb_code}
        )
    await run_in_threadpool(
        save_kb_config, settings.kb_config_dir, updated, path=path
    )

    _swap_registry(
        request, [item for item in configs if item.kb_code != updated.kb_code] + [updated]
    )
    # 通知其他副本重扫；本副本已经完成热加载与运行时驱逐
    if config_sync is not None:
        config_sync.publish(updated.kb_code)
    # 配置变更（切块/检索/Embedding）后丢弃旧运行时并失效检索缓存
    runtime_registry.evict(current.kb_id)
    # 先 bump 版本号再 SCAN+DELETE：SCAN 失败时版本号也足以让旧 key 自然失效
    await run_in_threadpool(cache.bump_version, current.tenant_id, current.kb_id)
    await run_in_threadpool(cache.invalidate_kb, current.tenant_id, current.kb_id)
    await repo.write_audit(
        tenant_id=current.tenant_id,
        kb_id=current.kb_id,
        user_id=user.user_id,
        action=AUDIT_ACTION_KB_UPDATE,
        detail={"kb_code": current.kb_code, "name": updated.name},
    )
    return KBSummary.from_config(updated)


@router.delete(
    "/{kb_key}",
    response_model=KBDeleteResponse,
    summary="删除知识库配置（库内仍有文档时拒绝）",
)
async def delete_kb(
    kb_key: str,
    request: Request,
    user: AdminUserDep,
    kb: KBWriteConfigDep,
    settings: SettingsDep,
    runtime_registry: RuntimeRegistryDep,
    cache: CacheDep,
    repo: DBRepoDep,
    config_sync: ConfigSyncDep,
) -> KBDeleteResponse:
    """删除知识库：库内仍有文档时拒绝（409），防止留下孤儿向量/元数据。"""
    _, total = await repo.list_documents(
        kb_id=kb.kb_id, tenant_id=kb.tenant_id, limit=1
    )
    if total:
        raise KBInUseError(
            f"知识库下仍有 {total} 个文档，请先删除文档",
            details={"kb_code": kb.kb_code, "documents": total},
        )

    configs = await run_in_threadpool(_load_configs, settings.kb_config_dir)
    path = await run_in_threadpool(find_kb_config_path, settings.kb_config_dir, kb.kb_code)
    if path is None:
        raise KBConfigError(
            f"知识库配置文件缺失：{kb.kb_code}", details={"kb_code": kb.kb_code}
        )
    await run_in_threadpool(delete_kb_config, path)

    _swap_registry(request, [item for item in configs if item.kb_code != kb.kb_code])
    if config_sync is not None:
        config_sync.publish(kb.kb_code)
    runtime_registry.evict(kb.kb_id)
    await run_in_threadpool(cache.bump_version, kb.tenant_id, kb.kb_id)
    await run_in_threadpool(cache.invalidate_kb, kb.tenant_id, kb.kb_id)
    await repo.write_audit(
        tenant_id=kb.tenant_id,
        kb_id=kb.kb_id,
        user_id=user.user_id,
        action=AUDIT_ACTION_KB_DELETE,
        detail={"kb_code": kb.kb_code, "name": kb.name},
    )
    return KBDeleteResponse(kb_id=kb.kb_id, kb_code=kb.kb_code)


@router.get("/{kb_key}", response_model=KBSummary, summary="知识库详情")
def get_kb(kb: KBConfigDep) -> KBSummary:
    """知识库详情：解析与权限校验全部由 KBConfigDep 依赖完成，这里只做模型转换。"""
    return KBSummary.from_config(kb)
