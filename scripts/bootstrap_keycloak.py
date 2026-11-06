"""Keycloak 本地联调一键配置（幂等）：realm / client / mapper / 组 / 用户。

用法：
    .venv\\Scripts\\python scripts\\bootstrap_keycloak.py

默认对接 docker-compose.dev.yml 里的 keycloak（localhost:8080，admin/admin）：
- realm：rag
- client：rag-api（public，开启 direct access grants，便于命令行拿 token 联调）
- mapper：groups（组 → access token 的 groups claim，短路径）+ audience（aud=rag-api）
- 组：finance / hr
- 用户：alice/alice123（finance）、bob/bob123（hr）

可用环境变量覆盖：KEYCLOAK_URL / KEYCLOAK_ADMIN / KEYCLOAK_ADMIN_PASSWORD /
KEYCLOAK_REALM / KEYCLOAK_CLIENT_ID
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BASE = os.environ.get("KEYCLOAK_URL", "http://localhost:8080").rstrip("/")
ADMIN = os.environ.get("KEYCLOAK_ADMIN", "admin")
ADMIN_PASSWORD = os.environ.get("KEYCLOAK_ADMIN_PASSWORD", "admin")
REALM = os.environ.get("KEYCLOAK_REALM", "rag")
CLIENT_ID = os.environ.get("KEYCLOAK_CLIENT_ID", "rag-api")

GROUPS = ["finance", "hr"]
USERS = [
    {"username": "alice", "password": "alice123", "groups": ["finance"]},
    {"username": "bob", "password": "bob123", "groups": ["hr"]},
]


def _req(
    method: str,
    path: str,
    *,
    token: str | None = None,
    body: dict[str, Any] | None = None,
    form: dict[str, str] | None = None,
) -> tuple[int, Any]:
    url = path if path.startswith("http") else f"{BASE}{path}"
    data: bytes | None = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif form is not None:
        data = urllib.parse.urlencode(form).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(raw) if raw else None
        except ValueError:
            payload = raw
        return exc.code, payload


def wait_ready(timeout_s: int = 180) -> None:
    # KC 26 起 /health 在管理端口 9000；主端口用 OIDC discovery 探活更通用
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status, _ = _req("GET", "/realms/master/.well-known/openid-configuration")
            if status == 200:
                print(f"[ok] keycloak ready: {BASE}")
                return
        except urllib.error.URLError:
            pass
        time.sleep(3)
    raise SystemExit(f"[fail] 等待 keycloak 就绪超时（{timeout_s}s）：{BASE}")


def admin_token() -> str:
    status, payload = _req(
        "POST",
        "/realms/master/protocol/openid-connect/token",
        form={
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": ADMIN,
            "password": ADMIN_PASSWORD,
        },
    )
    if status != 200 or not isinstance(payload, dict) or "access_token" not in payload:
        raise SystemExit(f"[fail] 获取 admin token 失败：{status} {payload}")
    return str(payload["access_token"])


def ensure_realm(token: str) -> None:
    status, payload = _req(
        "POST", "/admin/realms", token=token, body={"realm": REALM, "enabled": True}
    )
    if status in (201, 409):
        print(f"[ok] realm={REALM}（{'已存在' if status == 409 else '新建'}）")
    else:
        raise SystemExit(f"[fail] 创建 realm 失败：{status} {payload}")
    # 联调便利：access token 寿命放宽到 1 小时（生产 realm 勿用）
    status, realm = _req("GET", f"/admin/realms/{REALM}", token=token)
    if (
        status == 200
        and isinstance(realm, dict)
        and realm.get("accessTokenLifespan") != 3600
    ):
        realm["accessTokenLifespan"] = 3600
        status, payload = _req("PUT", f"/admin/realms/{REALM}", token=token, body=realm)
        if status != 204:
            raise SystemExit(f"[fail] 调整 token 寿命失败：{status} {payload}")
        print("[ok] accessTokenLifespan=3600")
    # KC 26 新 realm 默认启用 VERIFY_PROFILE，资料不全的用户直连授权会被拒
    # （"Account is not fully set up"）；本地联调 realm 直接关掉
    status, action = _req(
        "GET", f"/admin/realms/{REALM}/authentication/required-actions/VERIFY_PROFILE", token=token
    )
    if status == 200 and isinstance(action, dict) and action.get("enabled"):
        action["enabled"] = False
        status, payload = _req(
            "PUT",
            f"/admin/realms/{REALM}/authentication/required-actions/VERIFY_PROFILE",
            token=token,
            body=action,
        )
        if status != 204:
            raise SystemExit(f"[fail] 关闭 VERIFY_PROFILE 失败：{status} {payload}")
        print("[ok] required-action VERIFY_PROFILE 已关闭")


def _client_uuid(token: str) -> str:
    status, payload = _req(
        "GET", f"/admin/realms/{REALM}/clients?clientId={CLIENT_ID}", token=token
    )
    if status == 200 and isinstance(payload, list) and payload:
        return str(payload[0]["id"])
    raise SystemExit(f"[fail] 查询 client 失败：{status} {payload}")


def ensure_client(token: str) -> str:
    status, payload = _req(
        "POST",
        f"/admin/realms/{REALM}/clients",
        token=token,
        body={
            "clientId": CLIENT_ID,
            "enabled": True,
            "publicClient": True,
            "standardFlowEnabled": True,
            "directAccessGrantsEnabled": True,
            "redirectUris": ["*"],
            "webOrigins": ["*"],
        },
    )
    if status not in (201, 409):
        raise SystemExit(f"[fail] 创建 client 失败：{status} {payload}")
    uuid = _client_uuid(token)
    print(f"[ok] client={CLIENT_ID}（{'已存在' if status == 409 else '新建'}）")
    return uuid


def ensure_mappers(token: str, client_uuid: str) -> None:
    status, existing = _req(
        "GET", f"/admin/realms/{REALM}/clients/{client_uuid}/protocol-mappers/models",
        token=token,
    )
    names = (
        {m.get("name") for m in existing} if status == 200 and isinstance(existing, list) else set()
    )
    wanted = [
        {
            "name": "groups",
            "protocol": "openid-connect",
            "protocolMapper": "oidc-group-membership-mapper",
            "config": {
                "claim.name": "groups",
                "full.path": "false",
                "id.token.claim": "false",
                "access.token.claim": "true",
                "userinfo.token.claim": "false",
            },
        },
        {
            "name": f"{CLIENT_ID}-audience",
            "protocol": "openid-connect",
            "protocolMapper": "oidc-audience-mapper",
            "config": {
                "included.client.audience": CLIENT_ID,
                "id.token.claim": "false",
                "access.token.claim": "true",
            },
        },
    ]
    for mapper in wanted:
        if mapper["name"] in names:
            print(f"[ok] mapper={mapper['name']}（已存在）")
            continue
        status, payload = _req(
            "POST",
            f"/admin/realms/{REALM}/clients/{client_uuid}/protocol-mappers/models",
            token=token,
            body=mapper,
        )
        if status not in (201, 409):
            raise SystemExit(f"[fail] 创建 mapper 失败：{status} {payload}")
        print(f"[ok] mapper={mapper['name']}（新建）")


def _group_id(token: str, name: str) -> str:
    status, payload = _req("GET", f"/admin/realms/{REALM}/groups?search={name}", token=token)
    if status == 200 and isinstance(payload, list):
        for group in payload:
            if group.get("name") == name:
                return str(group["id"])
    raise SystemExit(f"[fail] 查询 group 失败：{status} {payload}")


def ensure_groups(token: str) -> dict[str, str]:
    ids: dict[str, str] = {}
    for name in GROUPS:
        status, payload = _req(
            "POST", f"/admin/realms/{REALM}/groups", token=token, body={"name": name}
        )
        if status not in (201, 409):
            raise SystemExit(f"[fail] 创建 group 失败：{status} {payload}")
        ids[name] = _group_id(token, name)
        print(f"[ok] group={name}（{'已存在' if status == 409 else '新建'}）")
    return ids


def ensure_users(token: str, group_ids: dict[str, str]) -> None:
    for spec in USERS:
        username = spec["username"]
        status, payload = _req(
            "POST",
            f"/admin/realms/{REALM}/users",
            token=token,
            body={
                "username": username,
                "enabled": True,
                "emailVerified": True,
                "firstName": username,
                "lastName": "demo",
                "email": f"{username}@example.com",
            },
        )
        if status not in (201, 409):
            raise SystemExit(f"[fail] 创建 user 失败：{status} {payload}")
        status, payload = _req(
            "GET", f"/admin/realms/{REALM}/users?username={username}&exact=true", token=token
        )
        if status != 200 or not isinstance(payload, list) or not payload:
            raise SystemExit(f"[fail] 查询 user 失败：{status} {payload}")
        user_id = str(payload[0]["id"])
        # 已存在的用户（409 分支）可能缺 profile 字段，幂等补齐
        status, payload = _req(
            "PUT",
            f"/admin/realms/{REALM}/users/{user_id}",
            token=token,
            body={
                "firstName": username,
                "lastName": "demo",
                "email": f"{username}@example.com",
                "emailVerified": True,
                "enabled": True,
            },
        )
        if status != 204:
            raise SystemExit(f"[fail] 更新 user 失败：{status} {payload}")
        status, payload = _req(
            "PUT",
            f"/admin/realms/{REALM}/users/{user_id}/reset-password",
            token=token,
            body={"type": "password", "value": spec["password"], "temporary": False},
        )
        if status not in (204,):
            raise SystemExit(f"[fail] 设置密码失败：{status} {payload}")
        for group_name in spec["groups"]:
            status, payload = _req(
                "PUT",
                f"/admin/realms/{REALM}/users/{user_id}/groups/{group_ids[group_name]}",
                token=token,
            )
            if status not in (204, 409):
                raise SystemExit(f"[fail] 加组失败：{status} {payload}")
        print(f"[ok] user={username} 组={spec['groups']}（密码已重置为默认值）")


def main() -> None:
    wait_ready()
    token = admin_token()
    ensure_realm(token)
    client_uuid = ensure_client(token)
    ensure_mappers(token, client_uuid)
    group_ids = ensure_groups(token)
    ensure_users(token, group_ids)
    print()
    print("联调用 .env 配置：")
    print("  AUTH_MODE=oidc")
    print(f"  OIDC_ISSUER={BASE}/realms/{REALM}")
    print(f"  OIDC_AUDIENCE={CLIENT_ID}")
    print(f"  OIDC_JWKS_URL={BASE}/realms/{REALM}/protocol/openid-connect/certs")
    print()
    print("拿 token（alice 在 finance 组）：")
    print(
        "  curl.exe -s -X POST "
        f"{BASE}/realms/{REALM}/protocol/openid-connect/token "
        f'-d "grant_type=password&client_id={CLIENT_ID}&username=alice&password=alice123"'
    )


if __name__ == "__main__":
    sys.exit(main())
