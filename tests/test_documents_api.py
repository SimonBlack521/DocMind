"""文档列表/详情与审计查询 API 测试：依赖 compose 的 PostgreSQL（标 integration）。

写入路径（上传/删除触发的审计）由 test_tasks.py 的端到端链路覆盖，这里只验证
查询端点的契约：分页结构、状态过滤、404、租户隔离由 repo 层测试覆盖。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = pytest.mark.integration


def test_document_list_and_detail_contract() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/kbs/finance/documents", params={"page": 1, "size": 10})
        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {"total", "page", "size", "items"}
        assert payload["page"] == 1
        assert payload["size"] == 10
        assert isinstance(payload["items"], list)

        # 状态过滤
        filtered = client.get(
            "/api/v1/kbs/finance/documents", params={"status": "ready"}
        )
        assert filtered.status_code == 200
        assert all(item["status"] == "ready" for item in filtered.json()["items"])

        # 非法状态值被 Query 校验拦截
        bad = client.get("/api/v1/kbs/finance/documents", params={"status": "bogus"})
        assert bad.status_code == 422

        # 不存在的文档 404
        missing = client.get(f"/api/v1/kbs/finance/documents/{'0' * 64}")
        assert missing.status_code == 404
        assert missing.json()["code"] == "document_not_found"


def test_audit_logs_contract() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/kbs/finance/audit-logs", params={"page": 1, "size": 10})
        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {"total", "page", "size", "items"}

        by_action = client.get(
            "/api/v1/kbs/finance/audit-logs", params={"action": "chat.query"}
        )
        assert by_action.status_code == 200
        assert all(item["action"] == "chat.query" for item in by_action.json()["items"])

        bad = client.get("/api/v1/kbs/finance/audit-logs", params={"action": "bogus"})
        assert bad.status_code == 422
