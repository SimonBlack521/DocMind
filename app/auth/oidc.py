"""OIDC + JWT 鉴权：校验 Bearer Token 的签名 / iss / aud / exp，并映射租户与知识库权限。

- JWKS 通过 PyJWKClient 拉取并缓存，首次认证时才联网，启动不依赖 IdP 可达；
- claim 名可配置：sub → user_id；OIDC_TENANT_CLAIM → tenant_id（缺省回落 default_tenant）；
  OIDC_KB_CLAIM → 可访问知识库集合（list 或逗号分隔字符串，缺省表示不限制）；
  OIDC_GROUP_CLAIM → 用户组集合（同上两种形态），用于文档级 ACL。
"""

from __future__ import annotations

from typing import Any, Protocol

import jwt
from fastapi import Request
from jwt import PyJWKClient
from jwt.exceptions import PyJWTError

from app.auth.base import AuthProvider, CurrentUser
from app.core.errors import AuthenticationError, ConfigurationError

# 只接受非对称签名算法，杜绝 HS256 密钥混淆与 alg=none 攻击
_ALLOWED_ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]


class _SigningKeySource(Protocol):
    """便于测试注入本地密钥源，生产为 PyJWKClient。"""

    def get_signing_key_from_jwt(self, token: str) -> Any: ...


class OIDCAuthProvider(AuthProvider):
    """企业 SSO 认证：Bearer JWT → PyJWKClient 取公钥验签 → claims 映射 CurrentUser。

    构造期只校验配置齐全性，不联网（JWKS 首次使用时才拉取并被 PyJWKClient
    缓存），因此 IdP 短暂不可达不影响服务启动，只影响启动后第一批请求。

    :param issuer/audience/jwks_url: IdP 三要素，缺一即 ConfigurationError；
    :param tenant_claim/kb_claim/group_claim: JWT 中租户 / 知识库 / 组的 claim 名，
        按企业 IdP 实际下发字段配置；
    :param default_tenant_id: token 未携带租户 claim 时的回落租户；
    :param allow_missing_groups/allow_missing_kb_ids: fail-open 开关。IdP 未下发
        相应 claim 时：False（默认）→ 空集合（仅公开文档 / 无 KB 访问权限）；
        True → None（不限权）。生产环境强烈建议保持 False，否则 IdP 配置漏下发
        claim 会导致受限资源被任意登录用户看到；
    :param jwks_client: 测试注入的本地密钥源（满足 _SigningKeySource 协议即可），
        避免测试依赖真实 IdP。
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        tenant_claim: str = "tenant_id",
        kb_claim: str = "allowed_kb_ids",
        group_claim: str = "groups",
        default_tenant_id: str = "default",
        allow_missing_groups: bool = False,
        allow_missing_kb_ids: bool = False,
        jwks_client: _SigningKeySource | None = None,
    ) -> None:
        if not issuer or not audience or not jwks_url:
            raise ConfigurationError(
                "AUTH_MODE=oidc 时 OIDC_ISSUER / OIDC_AUDIENCE / OIDC_JWKS_URL 必须全部配置"
            )
        self._issuer = issuer
        self._audience = audience
        self._tenant_claim = tenant_claim
        self._kb_claim = kb_claim
        self._group_claim = group_claim
        self._default_tenant_id = default_tenant_id
        # 默认 fail-closed：缺少 claim 不代表"不限权"
        self._allow_missing_groups = allow_missing_groups
        self._allow_missing_kb_ids = allow_missing_kb_ids
        self._jwks: _SigningKeySource = jwks_client or PyJWKClient(jwks_url)

    def authenticate(self, request: Request) -> CurrentUser:
        """验签并映射身份；任何一步失败统一抛 AuthenticationError（401）。

        jwt.decode 同时校验签名、iss、aud、exp（require 强制 exp/sub 必须存在）。
        注意算法白名单在模块级固定为非对称算法——即使 IdP 配置错误下发 HS256
        token 也会被拒，杜绝"用公钥当 HMAC 密钥"的算法混淆攻击。
        """
        token = self._extract_bearer(request)
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
            claims: dict[str, Any] = jwt.decode(
                token,
                signing_key.key,
                algorithms=_ALLOWED_ALGORITHMS,
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "sub"]},
            )
        except AuthenticationError:
            raise
        except PyJWTError as exc:
            raise AuthenticationError("访问令牌校验失败", details={"error": str(exc)}) from exc
        except Exception as exc:  # JWKS 拉取失败、token 结构非法等
            raise AuthenticationError("访问令牌校验失败", details={"error": str(exc)}) from exc

        user_id = str(claims.get("sub", "")).strip()
        if not user_id:
            raise AuthenticationError("访问令牌缺少 sub claim")

        tenant_raw = claims.get(self._tenant_claim)
        tenant_id = str(tenant_raw).strip() if tenant_raw else self._default_tenant_id
        groups = self._parse_str_set(claims.get(self._group_claim))
        if groups is None and not self._allow_missing_groups:
            groups = frozenset()
        kb_ids = self._parse_str_set(claims.get(self._kb_claim))
        if kb_ids is None and not self._allow_missing_kb_ids:
            # 缺少 KB 授权 claim：按"无可访问知识库"处理，而不是 None（不限权）。
            # 否则 can_access_kb 会把 None 解释成"本租户所有库都能访问"，
            # 再叠加 KB_ADMIN_GROUP 为空等默认行为，普通登录用户可能拿到管理权限
            kb_ids = frozenset()
        return CurrentUser(
            tenant_id=tenant_id,
            user_id=user_id,
            allowed_kb_ids=kb_ids,
            allowed_groups=groups,
        )

    @staticmethod
    def _extract_bearer(request: Request) -> str:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthenticationError("缺少 Bearer 访问令牌")
        return token.strip()

    @staticmethod
    def _parse_str_set(value: Any) -> frozenset[str] | None:
        """list[str] 或逗号分隔字符串都接受；缺省 / 空表示不限制。"""
        if isinstance(value, list):
            return frozenset(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, str) and value.strip():
            return frozenset(item.strip() for item in value.split(",") if item.strip())
        return None
