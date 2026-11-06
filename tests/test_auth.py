"""鉴权模块单元测试：dev 兼容、OIDC 配置校验、JWT 验签与 claim 映射、LDAP Basic 认证。

JWT 用本地生成的 RSA 密钥对自签，不依赖真实 IdP；
LDAP 用注入的 FakeGateway，不依赖真实目录服务。
"""

from __future__ import annotations

import base64
import time
from types import SimpleNamespace

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.requests import Request

from app.auth import DevAuthProvider, build_auth_provider
from app.auth.ldap import LDAPAuthProvider
from app.auth.oidc import OIDCAuthProvider
from app.core.errors import AuthenticationError, ConfigurationError
from app.core.settings import Settings

ISSUER = "https://idp.example.com"
AUDIENCE = "rag-api"


def _make_request(token: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


def _make_basic_request(username: str | None = None, password: str = "") -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if username is not None:
        payload = base64.b64encode(f"{username}:{password}".encode()).decode()
        headers.append((b"authorization", f"Basic {payload}".encode()))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


def _rsa_keypair() -> tuple[object, object]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


class _FakeJWKS:
    """测试用密钥源：跳过 JWKS 拉取，固定返回预置公钥。"""

    def __init__(self, public_key: object) -> None:
        self._public_key = public_key

    def get_signing_key_from_jwt(self, token: str) -> SimpleNamespace:
        return SimpleNamespace(key=self._public_key)


def _oidc_provider(public_key: object, **overrides: object) -> OIDCAuthProvider:
    kwargs = {
        "issuer": ISSUER,
        "audience": AUDIENCE,
        "jwks_url": "https://idp.example.com/.well-known/jwks.json",
        "jwks_client": _FakeJWKS(public_key),
    }
    kwargs.update(overrides)
    return OIDCAuthProvider(**kwargs)  # type: ignore[arg-type]


def _sign(private_key: object, **claims: object) -> str:
    now = int(time.time())
    payload = {"sub": "user-1", "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 600}
    payload.update(claims)
    return pyjwt.encode(payload, private_key, algorithm="RS256")


# ---------------------------------------------------------------- dev 模式
def test_dev_mode_returns_dev_user() -> None:
    settings = Settings(auth_mode="dev", dev_tenant_id="default", dev_user_id="dev-user")
    provider = build_auth_provider(settings)
    assert isinstance(provider, DevAuthProvider)
    user = provider.authenticate(_make_request())
    assert user.tenant_id == "default"
    assert user.user_id == "dev-user"
    assert user.allowed_kb_ids is None
    # dev 模式不下发组：文档级 ACL 不限制
    assert user.allowed_groups is None


# ---------------------------------------------------------------- 配置校验
def test_oidc_missing_config_raises() -> None:
    settings = Settings(auth_mode="oidc", oidc_issuer="", oidc_audience="", oidc_jwks_url="")
    with pytest.raises(ConfigurationError):
        build_auth_provider(settings)


def test_ldap_missing_config_raises() -> None:
    settings = Settings(auth_mode="ldap", ldap_server_url="", ldap_user_search_base="")
    with pytest.raises(ConfigurationError):
        build_auth_provider(settings)


def test_ldap_invalid_kb_map_json_raises() -> None:
    settings = Settings(
        auth_mode="ldap",
        ldap_server_url="ldap://ldap.example.com",
        ldap_user_search_base="ou=people,dc=example,dc=com",
        ldap_group_kb_map="not-json",
    )
    with pytest.raises(ConfigurationError):
        build_auth_provider(settings)


# ---------------------------------------------------------------- LDAP 认证
class _FakeGateway:
    """测试用目录：预置 用户名 → (密码, 组集合)，验证失败抛 AuthenticationError。"""

    def __init__(self, directory: dict[str, tuple[str, frozenset[str]]]) -> None:
        self._directory = directory
        self.seen: list[tuple[str, str]] = []

    def verify_user(self, username: str, password: str) -> frozenset[str]:
        self.seen.append((username, password))
        entry = self._directory.get(username)
        if entry is None or entry[0] != password:
            raise AuthenticationError("用户名或密码错误")
        return entry[1]


def _ldap_provider(gateway: _FakeGateway, **overrides: object) -> LDAPAuthProvider:
    kwargs = {
        "server_url": "ldap://ldap.example.com",
        "user_search_base": "ou=people,dc=example,dc=com",
        "gateway": gateway,
    }
    kwargs.update(overrides)
    return LDAPAuthProvider(**kwargs)  # type: ignore[arg-type]


def test_ldap_valid_credentials_maps_groups_and_kb() -> None:
    gateway = _FakeGateway({"alice": ("alice123", frozenset({"finance", "hr"}))})
    provider = _ldap_provider(
        gateway, group_kb_map={"finance": ["kb-1"], "hr": ["kb-2"], "ops": ["kb-3"]}
    )

    user = provider.authenticate(_make_basic_request("alice", "alice123"))
    assert user.user_id == "alice"
    assert user.tenant_id == "default"
    assert user.allowed_groups == frozenset({"finance", "hr"})
    # 只取用户所属组的映射并集，ops 不在其中
    assert user.allowed_kb_ids == frozenset({"kb-1", "kb-2"})
    # 凭证原样透传给目录服务
    assert gateway.seen == [("alice", "alice123")]


def test_ldap_empty_kb_map_means_unrestricted() -> None:
    gateway = _FakeGateway({"alice": ("alice123", frozenset({"finance"}))})
    provider = _ldap_provider(gateway)

    user = provider.authenticate(_make_basic_request("alice", "alice123"))
    assert user.allowed_kb_ids is None
    assert user.allowed_groups == frozenset({"finance"})


def test_ldap_user_without_groups_gets_empty_group_set() -> None:
    gateway = _FakeGateway({"carol": ("carol123", frozenset())})
    provider = _ldap_provider(gateway)

    user = provider.authenticate(_make_basic_request("carol", "carol123"))
    # 空集合 ≠ None：仅公开文档可见，不能回落为不限制
    assert user.allowed_groups == frozenset()


def test_ldap_wrong_password_rejected() -> None:
    gateway = _FakeGateway({"alice": ("alice123", frozenset({"finance"}))})
    provider = _ldap_provider(gateway)
    with pytest.raises(AuthenticationError):
        provider.authenticate(_make_basic_request("alice", "wrong"))


def test_ldap_unknown_user_rejected() -> None:
    gateway = _FakeGateway({"alice": ("alice123", frozenset({"finance"}))})
    provider = _ldap_provider(gateway)
    with pytest.raises(AuthenticationError):
        provider.authenticate(_make_basic_request("mallory", "whatever"))


def test_ldap_missing_basic_header_rejected() -> None:
    provider = _ldap_provider(_FakeGateway({}))
    with pytest.raises(AuthenticationError):
        provider.authenticate(_make_basic_request())


def test_ldap_malformed_basic_payload_rejected() -> None:
    provider = _ldap_provider(_FakeGateway({}))
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"authorization", b"Basic !!!not-base64!!!")],
        }
    )
    with pytest.raises(AuthenticationError):
        provider.authenticate(request)


# ---------------------------------------------------------------- JWT 校验
def test_oidc_valid_token_maps_claims() -> None:
    private_key, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key)
    token = _sign(private_key, tenant_id="acme", allowed_kb_ids=["kb-1", "kb-2"])

    user = provider.authenticate(_make_request(token))
    assert user.user_id == "user-1"
    assert user.tenant_id == "acme"
    assert user.allowed_kb_ids == frozenset({"kb-1", "kb-2"})


def test_oidc_kb_claim_accepts_comma_string_and_defaults() -> None:
    private_key, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key)

    # 逗号分隔字符串形式
    token = _sign(private_key, allowed_kb_ids="kb-1, kb-2")
    user = provider.authenticate(_make_request(token))
    assert user.allowed_kb_ids == frozenset({"kb-1", "kb-2"})
    # tenant claim 缺省回落 default_tenant_id
    assert user.tenant_id == "default"

    # 完全不下发 kb claim → 默认 fail-closed：无可访问知识库（frozenset()），
    # 而不是 None（None 在 can_access_kb 里表示不限权）
    token = _sign(private_key, tenant_id="acme")
    user = provider.authenticate(_make_request(token))
    assert user.allowed_kb_ids == frozenset()


def test_oidc_missing_kb_ids_can_be_explicitly_allowed() -> None:
    """显式开启兼容开关后才允许"缺少 kb claim = 不限权"，默认必须 fail-closed。"""
    private_key, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key, allow_missing_kb_ids=True)
    token = _sign(private_key, tenant_id="acme")
    user = provider.authenticate(_make_request(token))
    assert user.allowed_kb_ids is None


def test_oidc_group_claim_maps_to_allowed_groups() -> None:
    private_key, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key)

    # list 形式
    token = _sign(private_key, groups=["finance", "hr"])
    user = provider.authenticate(_make_request(token))
    assert user.allowed_groups == frozenset({"finance", "hr"})

    # 逗号分隔字符串形式
    token = _sign(private_key, groups="finance, hr")
    user = provider.authenticate(_make_request(token))
    assert user.allowed_groups == frozenset({"finance", "hr"})

    # 不下发组 claim → fail-closed：按"无授权组"处理，只能看到公开文档
    token = _sign(private_key, tenant_id="acme")
    user = provider.authenticate(_make_request(token))
    assert user.allowed_groups == frozenset()


def test_oidc_missing_groups_can_be_explicitly_allowed() -> None:
    """显式开启兼容开关后才允许"缺少 groups = 不限权"，默认必须 fail-closed。"""
    private_key, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key, allow_missing_groups=True)
    token = _sign(private_key, tenant_id="acme")
    user = provider.authenticate(_make_request(token))
    assert user.allowed_groups is None


def test_oidc_group_claim_name_configurable() -> None:
    private_key, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key, group_claim="roles")

    token = _sign(private_key, roles=["finance"])
    user = provider.authenticate(_make_request(token))
    assert user.allowed_groups == frozenset({"finance"})


def test_oidc_expired_token_rejected() -> None:
    private_key, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key)
    now = int(time.time())
    token = pyjwt.encode(
        {"sub": "user-1", "iss": ISSUER, "aud": AUDIENCE, "iat": now - 1200, "exp": now - 600},
        private_key,
        algorithm="RS256",
    )
    with pytest.raises(AuthenticationError):
        provider.authenticate(_make_request(token))


def test_oidc_wrong_audience_rejected() -> None:
    private_key, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key)
    token = _sign(private_key, aud="other-api")
    with pytest.raises(AuthenticationError):
        provider.authenticate(_make_request(token))


def test_oidc_wrong_signature_rejected() -> None:
    _, public_key = _rsa_keypair()
    other_private, _ = _rsa_keypair()
    provider = _oidc_provider(public_key)
    token = _sign(other_private)  # 用另一把私钥签名，公钥验不过
    with pytest.raises(AuthenticationError):
        provider.authenticate(_make_request(token))


def test_oidc_missing_bearer_rejected() -> None:
    _, public_key = _rsa_keypair()
    provider = _oidc_provider(public_key)
    with pytest.raises(AuthenticationError):
        provider.authenticate(_make_request())
