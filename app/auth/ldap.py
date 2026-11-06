"""LDAP 鉴权：HTTP Basic 凭证 → 目录服务两阶段 bind 验证 → 组 / 知识库映射。

认证流程：
    1. 服务账号 bind（LDAP_BIND_DN / LDAP_BIND_PASSWORD，留空则匿名 bind）；
    2. 按 LDAP_USER_SEARCH_FILTER 在 LDAP_USER_SEARCH_BASE 下搜索用户，得到 user_dn；
    3. 用 user_dn + 用户密码 rebind，验证密码（空密码会被 LDAP 当匿名 bind 放行，必须拒绝）；
    4. 按 LDAP_GROUP_SEARCH_FILTER 搜索用户所属组，取组名属性（默认 cn）→ allowed_groups；
    5. allowed_kb_ids 由 LDAP_GROUP_KB_MAP（组名 → kb_id 列表）并集得出；
       映射表为空表示不限制知识库范围；映射表非空但用户无匹配组 → 无可访问知识库。

测试可注入 FakeGateway 替代真实 ldap3 连接（同 OIDC 的 jwks_client 注入模式）。
"""

from __future__ import annotations

import base64
from typing import Protocol

from fastapi import Request

from app.auth.base import AuthProvider, CurrentUser
from app.core.errors import AuthenticationError, ConfigurationError


class _Gateway(Protocol):
    """目录访问抽象：生产为 LDAPGateway（ldap3），测试注入假实现。"""

    def verify_user(self, username: str, password: str) -> frozenset[str]:
        """验证用户名密码，返回用户所属组集合；失败抛 AuthenticationError。"""
        ...


class LDAPGateway:
    """真实 LDAP 目录访问（ldap3）。lazy import：未安装 ldap3 时给出清晰报错。"""

    def __init__(
        self,
        *,
        server_url: str,
        bind_dn: str = "",
        bind_password: str = "",
        user_search_base: str,
        user_search_filter: str = "(uid={username})",
        group_search_base: str = "",
        group_search_filter: str = "(member={user_dn})",
        group_name_attribute: str = "cn",
        timeout_s: float = 10.0,
    ) -> None:
        try:
            from ldap3 import ALL, Server
        except ImportError as exc:  # pragma: no cover - ldap3 已在依赖中
            raise ConfigurationError("AUTH_MODE=ldap 需要 ldap3 依赖：pip install ldap3") from exc
        self._server = Server(server_url, get_info=ALL, connect_timeout=timeout_s)
        self._bind_dn = bind_dn
        self._bind_password = bind_password
        self._user_search_base = user_search_base
        self._user_search_filter = user_search_filter
        self._group_search_base = group_search_base or user_search_base
        self._group_search_filter = group_search_filter
        self._group_name_attribute = group_name_attribute
        self._timeout_s = timeout_s

    def verify_user(self, username: str, password: str) -> frozenset[str]:
        """两阶段 bind 验证用户名密码，返回用户所属组名集合。

        为什么要两阶段而不是直接用用户 DN 拼密码 bind：多数企业目录里 DN 含有
        用户不可见的层级（ou=...），客户端只输入 uid/sAMAccountName，必须先用
        服务账号把 uid 解析成 user_dn，再用该 DN 验密码。
        组搜索失败（无权限/无组）不视为错误，返回空集合——语义是"该用户不属于
        任何组"，交由上层按空集合=仅公开资源处理。

        :raises AuthenticationError: 密码错误、用户不存在或空密码（统一话术，防用户枚举）；
        :raises ConfigurationError: 目录服务本身不可达（部署问题，非用户凭证问题）。
        """
        # 空密码在 LDAP simple bind 下会被当匿名访问放行，必须显式拒绝
        if not password:
            raise AuthenticationError("用户名或密码错误")

        from ldap3 import Connection
        from ldap3.core.exceptions import LDAPException
        from ldap3.utils.conv import escape_filter_chars

        # 1. 服务账号 bind（bind_dn 为空则匿名），用于搜索
        try:
            conn = Connection(
                self._server,
                user=self._bind_dn or None,
                password=self._bind_password or None,
                auto_bind=True,
                receive_timeout=self._timeout_s,
            )
        except LDAPException as exc:
            raise ConfigurationError(
                "LDAP 目录服务不可用", details={"error": str(exc)}
            ) from exc

        try:
            # 2. 搜索用户得到 user_dn
            user_filter = self._user_search_filter.format(username=escape_filter_chars(username))
            if not conn.search(self._user_search_base, user_filter) or not conn.entries:
                raise AuthenticationError("用户名或密码错误")
            user_dn = conn.entries[0].entry_dn

            # 3. 用户 rebind 验证密码
            try:
                Connection(
                    self._server,
                    user=user_dn,
                    password=password,
                    auto_bind=True,
                    receive_timeout=self._timeout_s,
                ).unbind()
            except LDAPException as exc:
                raise AuthenticationError("用户名或密码错误") from exc

            # 4. 搜索用户所属组
            group_filter = self._group_search_filter.format(
                user_dn=user_dn, username=escape_filter_chars(username)
            )
            groups: set[str] = set()
            if conn.search(
                self._group_search_base,
                group_filter,
                attributes=[self._group_name_attribute],
            ):
                for entry in conn.entries:
                    for value in entry.entry_attributes_as_dict.get(
                        self._group_name_attribute, []
                    ):
                        if str(value).strip():
                            groups.add(str(value).strip())
            return frozenset(groups)
        finally:
            conn.unbind()


class LDAPAuthProvider(AuthProvider):
    """HTTP Basic + LDAP 目录验证，产出 CurrentUser（组 / 知识库两级权限）。"""

    def __init__(
        self,
        *,
        server_url: str,
        user_search_base: str,
        bind_dn: str = "",
        bind_password: str = "",
        user_search_filter: str = "(uid={username})",
        group_search_base: str = "",
        group_search_filter: str = "(member={user_dn})",
        group_name_attribute: str = "cn",
        group_kb_map: dict[str, list[str]] | None = None,
        default_tenant_id: str = "default",
        timeout_s: float = 10.0,
        gateway: _Gateway | None = None,
    ) -> None:
        if not server_url or not user_search_base:
            raise ConfigurationError(
                "AUTH_MODE=ldap 时 LDAP_SERVER_URL 与 LDAP_USER_SEARCH_BASE 必须配置"
            )
        self._gateway: _Gateway = gateway or LDAPGateway(
            server_url=server_url,
            bind_dn=bind_dn,
            bind_password=bind_password,
            user_search_base=user_search_base,
            user_search_filter=user_search_filter,
            group_search_base=group_search_base,
            group_search_filter=group_search_filter,
            group_name_attribute=group_name_attribute,
            timeout_s=timeout_s,
        )
        self._kb_map = {group: frozenset(kb_ids) for group, kb_ids in (group_kb_map or {}).items()}
        self._default_tenant_id = default_tenant_id

    def authenticate(self, request: Request) -> CurrentUser:
        """解析 HTTP Basic 凭证并到目录服务验证，产出带组/知识库权限的 CurrentUser。"""
        username, password = self._extract_basic(request)
        groups = self._gateway.verify_user(username, password)
        return CurrentUser(
            tenant_id=self._default_tenant_id,
            user_id=username,
            allowed_kb_ids=self._map_kb_ids(groups),
            # LDAP 用户的组集合是权威结果：空集合表示仅公开文档可见，不能回落 None
            allowed_groups=groups,
        )

    def _map_kb_ids(self, groups: frozenset[str]) -> frozenset[str] | None:
        """映射表为空 → None 不限制；否则取并集（无匹配组 → 空集合 = 无可访问知识库）。"""
        if not self._kb_map:
            return None
        kb_ids: set[str] = set()
        for group in groups:
            kb_ids.update(self._kb_map.get(group, ()))
        return frozenset(kb_ids)

    @staticmethod
    def _extract_basic(request: Request) -> tuple[str, str]:
        scheme, _, payload = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "basic" or not payload.strip():
            raise AuthenticationError("缺少 Basic 认证凭证")
        try:
            # binascii.Error 与 UnicodeDecodeError 均为 ValueError 子类
            decoded = base64.b64decode(payload.strip(), validate=True).decode("utf-8")
        except ValueError as exc:
            raise AuthenticationError("Basic 认证凭证编码非法") from exc
        username, sep, password = decoded.partition(":")
        if not sep or not username:
            raise AuthenticationError("Basic 认证凭证格式非法")
        return username, password
