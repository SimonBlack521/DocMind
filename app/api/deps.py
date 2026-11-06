"""FastAPI 依赖：身份解析、知识库解析与权限校验、运行时对象获取。

身份统一由 app.state.auth_provider（AuthProvider）产出：dev 占位 / OIDC+JWT / LDAP（预留）。
权限单元是知识库：租户必须匹配，且用户绑定的知识库集合允许访问。
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.auth import AuthProvider, CurrentUser, build_auth_provider
from app.core.errors import ConfigurationError, KBNotFoundError, PermissionDeniedError
from app.core.logging import get_logger
from app.core.ratelimit import RateLimiter, build_rate_limiter
from app.core.settings import Settings, get_settings
from app.db.repo import MetadataRepo
from app.domain.config_sync import ConfigSync
from app.domain.kb_config import KBConfig, KBRegistry
from app.orchestration.pipeline import RAGPipeline
from app.orchestration.registry import RuntimeRegistry
from app.retrieval.cache import RetrievalCache
from app.storage.object_store import S3ObjectStore

logger = get_logger(__name__)

# CurrentUser 已迁移到 app.auth.base，此处 re-export 兼容既有 import 路径

# 依赖别名约定：Annotated[类型, Depends(函数)] 起一个 XxxDep 别名，
# 端点签名里写一个别名即完成注入，避免重复的 Annotated 样板代码
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_auth_provider(request: Request, settings: SettingsDep) -> AuthProvider:
    """从 app.state 取认证提供者；正常请求必命中 lifespan 注入的单例。"""
    provider = getattr(request.app.state, "auth_provider", None)
    if provider is None:
        # 未走 lifespan 的场景（如直接调依赖的单元测试）按配置即时构建
        provider = build_auth_provider(settings)
    return provider


AuthProviderDep = Annotated[AuthProvider, Depends(get_auth_provider)]


def get_current_user(request: Request, provider: AuthProviderDep) -> CurrentUser:
    """把 HTTP 请求委托给 AuthProvider 解析出当前用户（dev 占位 / JWT 校验等）。"""
    return provider.authenticate(request)


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]


def get_kb_registry(request: Request) -> KBRegistry:
    """取知识库注册表单例；未初始化说明 app 没走 lifespan，属于部署错误。"""
    registry = getattr(request.app.state, "kb_registry", None)
    if registry is None:
        raise ConfigurationError("知识库注册表尚未初始化")
    return registry


RegistryDep = Annotated[KBRegistry, Depends(get_kb_registry)]


def get_config_sync(request: Request) -> ConfigSync | None:
    """取配置失效通知单例；无 lifespan 场景返回 None（调用方跳过检查）。"""
    return getattr(request.app.state, "config_sync", None)


ConfigSyncDep = Annotated[ConfigSync | None, Depends(get_config_sync)]


def get_pipeline(request: Request) -> RAGPipeline:
    """取 RAG 管线单例（检索 + 生成的编排入口）。"""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise ConfigurationError("RAG 管线尚未初始化")
    return pipeline


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


def get_db_repo(request: Request) -> MetadataRepo:
    """取 PG 元数据仓库单例（documents / ingest_tasks / audit_logs 表的操作入口）。"""
    repo = getattr(request.app.state, "db_repo", None)
    if repo is None:
        raise ConfigurationError("元数据仓库尚未初始化")
    return repo


DBRepoDep = Annotated[MetadataRepo, Depends(get_db_repo)]


def get_object_store(request: Request) -> S3ObjectStore:
    """取对象存储单例（原文归档 + 异步任务大文件暂存）。"""
    store = getattr(request.app.state, "object_store", None)
    if store is None:
        raise ConfigurationError("对象存储尚未初始化")
    return store


ObjectStoreDep = Annotated[S3ObjectStore, Depends(get_object_store)]


def get_runtime_registry(request: Request) -> RuntimeRegistry:
    """取运行时注册表单例（按 kb_id 缓存向量库/关键词库/Embedding 等客户端）。"""
    registry = getattr(request.app.state, "runtime_registry", None)
    if registry is None:
        raise ConfigurationError("运行时注册表尚未初始化")
    return registry


RuntimeRegistryDep = Annotated[RuntimeRegistry, Depends(get_runtime_registry)]


def get_retrieval_cache(request: Request) -> RetrievalCache:
    """取检索结果缓存单例（Redis；禁用时为 no-op 实现）。"""
    cache = getattr(request.app.state, "retrieval_cache", None)
    if cache is None:
        raise ConfigurationError("检索缓存尚未初始化")
    return cache


CacheDep = Annotated[RetrievalCache, Depends(get_retrieval_cache)]


def can_access_kb(kb: KBConfig, user: CurrentUser) -> bool:
    """知识库级 ACL：租户必须一致；用户带 allowed_kb_ids 白名单时库必须在名单内。"""
    if kb.tenant_id != user.tenant_id:
        return False
    # allowed_kb_ids 为 None 表示不受限（可访问本租户全部库）
    return user.allowed_kb_ids is None or kb.kb_id in user.allowed_kb_ids


def _in_group(user: CurrentUser, group: str) -> bool:
    """用户是否属于指定授权组（组名为空 / 用户组为 None 都视为不匹配）。"""
    return bool(group) and bool(user.allowed_groups) and group in user.allowed_groups


def has_platform_admin_role(user: CurrentUser, settings: Settings) -> bool:
    """平台管理员：配置了平台管理员组时只认组成员，否则沿用未受限用户兜底。"""
    if _in_group(user, settings.platform_admin_group):
        return True
    if settings.rbac_strict or settings.platform_admin_group:
        return False
    return user.allowed_kb_ids is None


def has_kb_admin_role(user: CurrentUser, settings: Settings) -> bool:
    """知识库管理员：平台管理员 / 管理员组成员；未配置组时非严格模式允许未受限用户。"""
    if has_platform_admin_role(user, settings) or _in_group(user, settings.kb_admin_group):
        return True
    if settings.rbac_strict or settings.kb_admin_group:
        return False
    return user.allowed_kb_ids is None


def has_kb_contributor_role(user: CurrentUser, settings: Settings) -> bool:
    """贡献者：可上传/删除文档；管理员天然具备该权限。"""
    if has_kb_admin_role(user, settings) or _in_group(user, settings.kb_contributor_group):
        return True
    if settings.rbac_strict or settings.kb_contributor_group:
        return False
    return user.allowed_kb_ids is None


def has_auditor_role(user: CurrentUser, settings: Settings) -> bool:
    """审计员：可查看审计日志；管理员天然具备该权限。"""
    if has_kb_admin_role(user, settings) or _in_group(user, settings.kb_auditor_group):
        return True
    if settings.rbac_strict or settings.kb_auditor_group:
        return False
    return user.allowed_kb_ids is None


def ensure_kb_admin(user: CurrentUserDep, settings: SettingsDep) -> CurrentUser:
    """知识库写权限：受限用户（IdP 下发 allowed_kb_ids）不具备管理权；
    配置 KB_ADMIN_GROUP 时还要求用户属于该组（留空表示不限制）。"""
    if not has_kb_admin_role(user, settings):
        raise PermissionDeniedError(
            "当前用户没有知识库管理权限",
            details={"user_id": user.user_id, "required_group": settings.kb_admin_group},
        )
    return user


AdminUserDep = Annotated[CurrentUser, Depends(ensure_kb_admin)]


def ensure_kb_contributor(user: CurrentUserDep, settings: SettingsDep) -> CurrentUser:
    """文档写权限（上传/删除）：要求贡献者及以上角色，防止普通读者直接改知识库内容。"""
    if not has_kb_contributor_role(user, settings):
        raise PermissionDeniedError(
            "当前用户没有文档写入权限",
            details={
                "user_id": user.user_id,
                "required_group": settings.kb_contributor_group,
            },
        )
    return user


ContributorUserDep = Annotated[CurrentUser, Depends(ensure_kb_contributor)]


def ensure_auditor(user: CurrentUserDep, settings: SettingsDep) -> CurrentUser:
    """审计查看权限：与知识库管理权分离，便于只读审计角色独立授权。"""
    if not has_auditor_role(user, settings):
        raise PermissionDeniedError(
            "当前用户没有审计日志查看权限",
            details={"user_id": user.user_id, "required_group": settings.kb_auditor_group},
        )
    return user


AuditorUserDep = Annotated[CurrentUser, Depends(ensure_auditor)]


def ensure_platform_admin(user: CurrentUserDep, settings: SettingsDep) -> CurrentUser:
    """平台管理员权限：跨知识库运维动作（预留，当前用于任务越权查看兜底）。"""
    if not has_platform_admin_role(user, settings):
        raise PermissionDeniedError(
            "当前用户不是平台管理员",
            details={"user_id": user.user_id, "required_group": settings.platform_admin_group},
        )
    return user


PlatformAdminUserDep = Annotated[CurrentUser, Depends(ensure_platform_admin)]


def _refresh_registry(request: Request, settings: Settings) -> KBRegistry:
    """重扫配置目录并替换 app.state 中的注册表（多副本部署下的配置最终一致）。"""
    registry = KBRegistry.from_directory(settings.kb_config_dir)
    request.app.state.kb_registry = registry
    return registry


def _refresh_if_config_changed(
    request: Request,
    settings: Settings,
    sync: ConfigSync | None,
) -> None:
    """检查 Redis 配置版本号；变化时重扫目录。

    限频：每 kb_config_sync_interval_s 秒最多查一次 Redis，避免每个请求都多一次
    网络往返。版本号读取失败/未设置时保持现有注册表，不影响请求。
    """
    interval = settings.kb_config_sync_interval_s
    if sync is None or not sync.enabled or interval <= 0:
        return
    now = time.monotonic()
    last_checked, last_version = (
        getattr(request.app.state, "config_sync_state", None) or (0.0, None)
    )
    if now - last_checked < interval:
        return
    version = sync.version()
    request.app.state.config_sync_state = (now, version)
    if version is None or version == last_version:
        return
    logger.info("检测到配置版本变化（%s -> %s），重扫知识库配置", last_version, version)
    _refresh_registry(request, settings)


def resolve_kb(
    kb_key: str,
    request: Request,
    user: CurrentUserDep,
    settings: SettingsDep,
    config_sync: ConfigSyncDep,
) -> KBConfig:
    """路径上的 kb_id / kb_code 都能解析，统一在这里做启用状态与权限校验。"""
    _refresh_if_config_changed(request, settings, config_sync)
    registry = get_kb_registry(request)
    try:
        kb = registry.resolve(kb_key)
    except KBNotFoundError:
        # 其他实例可能刚写入新配置：重扫一次目录再判定，避免多副本部署下新建库不可见
        registry = _refresh_registry(request, settings)
        kb = registry.resolve(kb_key)
    if not kb.enabled:
        raise KBNotFoundError(f"知识库未启用：{kb_key}", details={"kb": kb_key})
    if not can_access_kb(kb, user):
        raise PermissionDeniedError(
            f"当前用户无权访问知识库：{kb_key}", details={"kb": kb_key, "user": user.user_id}
        )
    return kb


KBConfigDep = Annotated[KBConfig, Depends(resolve_kb)]


def resolve_kb_for_write(
    kb_key: str,
    request: Request,
    user: CurrentUserDep,
    settings: SettingsDep,
    config_sync: ConfigSyncDep,
) -> KBConfig:
    """管理端点用：不校验 enabled（否则禁用后就无法再启用），但租户/访问范围仍然校验。"""
    _refresh_if_config_changed(request, settings, config_sync)
    registry = get_kb_registry(request)
    try:
        kb = registry.resolve(kb_key)
    except KBNotFoundError:
        registry = _refresh_registry(request, settings)
        kb = registry.resolve(kb_key)
    if not can_access_kb(kb, user):
        raise PermissionDeniedError(
            f"当前用户无权访问知识库：{kb_key}", details={"kb": kb_key, "user": user.user_id}
        )
    return kb


KBWriteConfigDep = Annotated[KBConfig, Depends(resolve_kb_for_write)]


def get_rate_limiter(request: Request, settings: SettingsDep) -> RateLimiter:
    """取限流器单例；无 lifespan 场景即时构建（禁用时为零成本的 DisabledRateLimiter）。"""
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        return limiter
    # 直接调 router（无 lifespan）的场景：按设置即时构建（DisabledRateLimiter 零成本）
    limiter = build_rate_limiter(
        enabled=settings.rate_limit_enabled,
        redis_url=settings.redis_url,
    )
    return limiter


def rate_limit_dep(action: str, *, limit: int):
    """返回一个 FastAPI dependency：检查 (action, user_id) 窗口内计数，超限抛 429。

    用法：
        @router.post("/chat")
        async def chat(
            user: CurrentUserDep,
            _: Annotated[None, Depends(rate_limit_dep("chat", limit=30))],
        ): ...
    """

    def _check(
        user: CurrentUserDep,
        limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    ) -> None:
        result = limiter.check(action, user.user_id, limit=limit)
        if not result.allowed:
            # 429 带 Retry-After / X-RateLimit-Limit 头，方便客户端做退避重试
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate_limit_exceeded",
                headers={
                    "Retry-After": str(int(result.reset_after_s)),
                    "X-RateLimit-Limit": str(limit),
                },
            )

    return Depends(_check)
