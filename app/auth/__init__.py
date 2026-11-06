"""鉴权入口：按 AUTH_MODE 构建认证提供者。

在整体链路中的角色：API 层依赖注入（Depends）拿到 build_auth_provider 产出的
AuthProvider，每个请求调用 authenticate() 得到 CurrentUser，再进入租户/知识库
两级权限校验。oidc / ldap 的重依赖（PyJWT、ldap3）在对应分支内 lazy import：
dev 模式下不装这两个包也能起服务，也缩短测试冷启动。
"""

from __future__ import annotations

from app.auth.base import AuthProvider, CurrentUser, DevAuthProvider
from app.core.errors import ConfigurationError
from app.core.settings import Settings

__all__ = [
    "AuthProvider",
    "CurrentUser",
    "DevAuthProvider",
    "build_auth_provider",
]


def build_auth_provider(settings: Settings) -> AuthProvider:
    """工厂函数：按 settings.auth_mode 返回对应的认证提供者。

    ldap 模式下会把 Settings.ldap_group_kb_map（JSON 字符串）解析成
    {组名: [kb_id, ...]} 映射表交给 LDAPAuthProvider；JSON 非法或不是对象时
    抛 ConfigurationError（启动期 fail-fast，而非等首个请求才发现映射表写错）。
    """
    if settings.auth_mode == "dev":
        return DevAuthProvider(tenant_id=settings.dev_tenant_id, user_id=settings.dev_user_id)
    if settings.auth_mode == "oidc":
        from app.auth.oidc import OIDCAuthProvider

        return OIDCAuthProvider(
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
            jwks_url=settings.oidc_jwks_url,
            tenant_claim=settings.oidc_tenant_claim,
            kb_claim=settings.oidc_kb_claim,
            group_claim=settings.oidc_group_claim,
            default_tenant_id=settings.default_tenant_id,
            allow_missing_groups=settings.oidc_allow_missing_groups,
            allow_missing_kb_ids=settings.oidc_allow_missing_kb_ids,
        )
    if settings.auth_mode == "ldap":
        import json

        from app.auth.ldap import LDAPAuthProvider

        kb_map: dict[str, list[str]] | None = None
        if settings.ldap_group_kb_map.strip():
            try:
                kb_map = json.loads(settings.ldap_group_kb_map)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    "LDAP_GROUP_KB_MAP 必须是合法 JSON，如 {\"finance\":[\"kb-1\"]}",
                    details={"error": str(exc)},
                ) from exc
            if not isinstance(kb_map, dict):
                raise ConfigurationError("LDAP_GROUP_KB_MAP 必须是 JSON 对象（组名 → kb_id 列表）")
        return LDAPAuthProvider(
            server_url=settings.ldap_server_url,
            user_search_base=settings.ldap_user_search_base,
            bind_dn=settings.ldap_bind_dn,
            bind_password=settings.ldap_bind_password,
            user_search_filter=settings.ldap_user_search_filter,
            group_search_base=settings.ldap_group_search_base,
            group_search_filter=settings.ldap_group_search_filter,
            group_name_attribute=settings.ldap_group_name_attribute,
            group_kb_map=kb_map,
            default_tenant_id=settings.default_tenant_id,
            timeout_s=settings.ldap_timeout_s,
        )
    raise ConfigurationError(f"未知 AUTH_MODE：{settings.auth_mode}")
