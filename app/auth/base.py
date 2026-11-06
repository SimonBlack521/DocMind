"""鉴权抽象：不同认证模式（dev / OIDC / LDAP）统一产出 CurrentUser。

权限单元是知识库：tenant_id 必须匹配，allowed_kb_ids 为 None 表示不限制。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """认证通过后下游所有模块使用的身份视图。

    frozen + slots：身份对象在请求生命周期内只读，杜绝业务代码中途篡改权限。
    两类权限字段的 None / 空集合语义不同，是本文件最重要的约定：
    - None  = 未受限（IdP 没下发该 claim，或 dev 模式），放行全部；
    - 空集合 = 明确"什么都没有"（如 LDAP 用户不属于任何映射组），拒绝全部。
    混淆这两者会把"未配置权限"误当成"无任何权限"，或反之造成越权。
    """

    tenant_id: str
    user_id: str
    # None 表示不限制（dev 模式 / IdP 未下发 kb claim）；否则为可访问的知识库 kb_id 集合
    allowed_kb_ids: frozenset[str] | None = None
    # None 表示不限制（dev 模式 / IdP 未下发组 claim）；否则为用户所属组集合，
    # 用于文档级 ACL：文档 acl_groups 为空（公开）或与用户组有交集才可见
    allowed_groups: frozenset[str] | None = None


class AuthProvider(ABC):
    """认证提供者：从请求中解析并校验身份。"""

    @abstractmethod
    def authenticate(self, request: Request) -> CurrentUser:
        """校验请求身份；凭证缺失/非法时抛 AuthenticationError。"""


class DevAuthProvider(AuthProvider):
    """开发占位身份：所有请求视为同一 dev 用户，不校验任何凭证（禁止用于生产）。"""

    def __init__(self, *, tenant_id: str, user_id: str) -> None:
        self._tenant_id = tenant_id
        self._user_id = user_id

    def authenticate(self, request: Request) -> CurrentUser:
        return CurrentUser(tenant_id=self._tenant_id, user_id=self._user_id)
