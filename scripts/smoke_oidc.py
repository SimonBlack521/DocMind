"""OIDC 端到端冒烟：Keycloak JWT → API 鉴权 → 文档级 ACL 过滤验证。

前置：
    1. docker compose -f docker-compose.dev.yml up -d keycloak（及 PG/Redis/Milvus/ES）
    2. .venv\\Scripts\\python scripts\\bootstrap_keycloak.py
    3. API 以 oidc 模式启动，例如：
       $env:AUTH_MODE="oidc"; $env:OIDC_ISSUER="http://localhost:8080/realms/rag";
       $env:OIDC_AUDIENCE="rag-api";
       $env:OIDC_JWKS_URL="http://localhost:8080/realms/rag/protocol/openid-connect/certs";
       .venv\\Scripts\\python -m uvicorn app.main:app --port 8000

验证点：
    - 无 token 访问业务接口返回 401
    - alice（finance 组）/ bob（hr 组）的 JWT 都能通过鉴权
    - 上传 hr 组文档与公开文档后：alice 列表/详情不可见 hr 文档，bob 可见
    - 问答引用与文档级 ACL 一致：alice 的引用不含 hr 文档，bob 的引用命中 hr 文档

退出码：0 全部通过；1 存在失败项。
"""

from __future__ import annotations

import os
import sys
import time
import uuid

import httpx

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
KC_BASE = os.environ.get("KEYCLOAK_URL", "http://localhost:8080").rstrip("/")
REALM = os.environ.get("KEYCLOAK_REALM", "rag")
CLIENT_ID = os.environ.get("KEYCLOAK_CLIENT_ID", "rag-api")
KB_KEY = os.environ.get("SMOKE_KB", "finance")

QUESTION = "火星土壤的酸碱度是多少？"
HR_CONTENT = (
    "机密资料：火星土壤的酸碱度 pH 值为 8.8，呈弱碱性。"
    "该数据为内部实验测定，仅限人力资源部相关项目组查阅。"
)
PUBLIC_CONTENT = "公开资料：月球表面的重力加速度约为 1.62 米每二次方秒。"

# 本机直连：绕过系统代理（httpx 默认 trust_env=True 会读 HTTP_PROXY 把 127.0.0.1 也代理走）
_client = httpx.Client(trust_env=False, timeout=300)

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def get_token(username: str, password: str) -> str:
    resp = _client.post(
        f"{KC_BASE}/realms/{REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": CLIENT_ID,
            "username": username,
            "password": password,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return str(resp.json()["access_token"])


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def upload(token: str, filename: str, content: str, acl_groups: str) -> dict:
    resp = _client.post(
        f"{API_BASE}/api/v1/kbs/{KB_KEY}/documents",
        headers=auth(token),
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
        data={"acl_groups": acl_groups},
        timeout=300,
    )
    resp.raise_for_status()
    return dict(resp.json())


def list_titles(token: str) -> list[str]:
    resp = _client.get(
        f"{API_BASE}/api/v1/kbs/{KB_KEY}/documents",
        headers=auth(token),
        params={"limit": 100},
        timeout=15,
    )
    resp.raise_for_status()
    return [item["title"] for item in resp.json()["items"]]


def chat_citation_titles(token: str, question: str) -> list[str]:
    resp = _client.post(
        f"{API_BASE}/api/v1/kbs/{KB_KEY}/chat",
        headers=auth(token),
        json={"question": question},
        timeout=300,
    )
    resp.raise_for_status()
    return [c["title"] for c in resp.json()["citations"]]


def wait_api_ready(timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if _client.get(f"{API_BASE}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise SystemExit(f"[fail] API 未就绪：{API_BASE}")


def main() -> int:
    wait_api_ready()
    suffix = uuid.uuid4().hex[:8]
    hr_title = f"oidc-smoke-hr-{suffix}"
    public_title = f"oidc-smoke-public-{suffix}"

    resp = _client.get(f"{API_BASE}/api/v1/kbs", timeout=15)
    check("无 token 访问业务接口返回 401", resp.status_code == 401, f"got {resp.status_code}")

    alice = os.environ.get("SMOKE_ALICE_TOKEN") or get_token("alice", "alice123")
    bob = os.environ.get("SMOKE_BOB_TOKEN") or get_token("bob", "bob123")

    resp = _client.get(f"{API_BASE}/api/v1/kbs", headers=auth(alice), timeout=15)
    check("alice JWT 通过鉴权（GET /kbs 200）", resp.status_code == 200, f"got {resp.status_code}")

    hr_doc = upload(alice, f"{hr_title}.txt", HR_CONTENT, acl_groups="hr")
    public_doc = upload(alice, f"{public_title}.txt", PUBLIC_CONTENT, acl_groups="")
    print(
        f"[info] 已上传 hr 文档 doc_id={hr_doc['doc_id']} / 公开文档 doc_id={public_doc['doc_id']}"
    )

    alice_titles = list_titles(alice)
    check("alice（finance）列表不含 hr 文档", hr_title not in alice_titles)
    check("alice（finance）列表含公开文档", public_title in alice_titles)

    resp = _client.get(
        f"{API_BASE}/api/v1/kbs/{KB_KEY}/documents/{hr_doc['doc_id']}",
        headers=auth(alice),
        timeout=15,
    )
    check("alice 访问 hr 文档详情返回 404", resp.status_code == 404, f"got {resp.status_code}")

    bob_titles = list_titles(bob)
    check("bob（hr）列表含 hr 文档", hr_title in bob_titles)
    check("bob（hr）列表含公开文档", public_title in bob_titles)

    # 历次冒烟会留下同内容的多份 hr 文档，断言按标题前缀判定，与具体 doc_id 无关
    time.sleep(1)
    alice_cited = chat_citation_titles(alice, QUESTION)
    check(
        "alice 问答引用不含 hr 文档",
        not any(t.startswith("oidc-smoke-hr") for t in alice_cited),
        f"cited={alice_cited}",
    )

    bob_cited = chat_citation_titles(bob, QUESTION)
    check(
        "bob 问答引用命中 hr 文档",
        any(t.startswith("oidc-smoke-hr") for t in bob_cited),
        f"cited={bob_cited}",
    )

    print()
    if _failures:
        print(f"冒烟失败 {len(_failures)} 项：{_failures}")
        return 1
    print("OIDC 冒烟全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
