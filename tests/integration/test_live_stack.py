"""真实依赖集成测试：需要 PG / Redis / Milvus / ES / S3 与 API 实例同时在线。

默认环境（只跑单测）下整文件 skip；上线前在有完整栈的环境执行：
    pytest tests/integration -q

覆盖：入库 → 混合检索引用 → 知识库隔离 → 删除后可检索性消失 → 运维端点。
"""

from __future__ import annotations

import time
import uuid

import pytest

httpx = pytest.importorskip("httpx")

BASE = "http://127.0.0.1:8000"


def _api_ready() -> bool:
    try:
        resp = httpx.get(f"{BASE}/health", timeout=3)
    except Exception:
        return False
    if resp.status_code != 200:
        return False
    return all(v == "up" for v in resp.json().get("services", {}).values())


pytestmark = pytest.mark.skipif(
    not _api_ready(),
    reason="完整依赖栈与 API 未就绪（集成测试需要真实 PG/Redis/Milvus/ES/S3）",
)


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    with httpx.Client(base_url=f"{BASE}/api/v1", timeout=180) as c:
        yield c


def _doc_text(term: str, topic: str, rounds: int = 10) -> str:
    sections = [f"# {topic}管理制度", "", f"本制度适用于所有部门的{topic}相关流程。"]
    for i in range(1, rounds + 1):
        sections.append(
            f"## 第{i}条\n涉及 {term} 的{topic}申请必须在三个工作日内提交，"
            f"审批编号格式为 {term}-{i:03d}，复核通过后归档保存五年。"
        )
    return "\n\n".join(sections)


def _upload(client: httpx.Client, kb: str, name: str, title: str, text: str) -> dict:
    resp = client.post(
        f"/kbs/{kb}/documents",
        files={"file": (name, text.encode("utf-8"), "text/markdown")},
        data={"title": title, "acl_groups": ""},
    )
    resp.raise_for_status()
    return resp.json()


def test_ingest_search_acl_delete_roundtrip(client: httpx.Client) -> None:
    """入库 → 检索命中 → 跨库不串 → 删除后不再命中。"""
    term = f"ITG-{uuid.uuid4().hex[:8]}"
    fin = _upload(client, "finance", f"{term}.md", "集成测试制度", _doc_text(term, "报销"))
    try:
        assert fin["chunk_count"] > 0, "入库未产生任何 chunk"

        answer = client.post(
            "/kbs/finance/chat",
            json={"question": f"{term} 的报销审批流程是什么？", "mode": "document", "top_k": 5},
        )
        answer.raise_for_status()
        body = answer.json()
        cited_docs = {c.get("doc_id") for c in body.get("citations", [])}
        assert fin["doc_id"] in cited_docs, "检索未命中刚入库的文档"

        cross = client.post(
            "/kbs/hr/chat",
            json={"question": f"{term} 的报销审批流程是什么？", "mode": "document", "top_k": 5},
        )
        cross.raise_for_status()
        leaked = {c.get("doc_id") for c in cross.json().get("citations", [])}
        assert fin["doc_id"] not in leaked, "知识库隔离失效：hr 查到了 finance 文档"
    finally:
        client.delete(f"/kbs/finance/documents/{fin['doc_id']}")

    after = client.post(
        "/kbs/finance/chat",
        json={"question": f"{term} 的报销审批流程是什么？", "mode": "document", "top_k": 5},
    )
    after.raise_for_status()
    remaining = {c.get("doc_id") for c in after.json().get("citations", [])}
    assert fin["doc_id"] not in remaining, "删除后仍能检索到文档"


def test_ops_endpoints(client: httpx.Client) -> None:
    """死信列表与队列深度端点可用（不依赖具体数据）。"""
    started = time.perf_counter()
    dead = client.get("/kbs/finance/dead-letters")
    dead.raise_for_status()
    body = dead.json()
    assert {"total", "page", "size", "items"} <= set(body), "死信列表响应字段缺失"
    assert isinstance(body["items"], list)

    depth = httpx.get(f"{BASE}/metrics/queue-depth", timeout=10)
    depth.raise_for_status()
    assert depth.json()["queue"] == "celery"
    assert time.perf_counter() - started < 30
