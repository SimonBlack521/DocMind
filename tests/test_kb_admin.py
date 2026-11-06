"""知识库管理 API 测试：运行时创建 / 更新 / 删除并热加载 YAML 配置。

用内存假仓库覆盖 MetadataRepo，测试不依赖 PostgreSQL；配置目录用 tmp_path 隔离，
避免污染 configs/kbs 下的真实知识库。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db_repo
from app.core.settings import get_settings
from app.main import create_app

BASE = "/api/v1/kbs"


class _FakeRepo:
    """只实现知识库管理端点用到的两个方法。"""

    def __init__(self) -> None:
        self.total = 0
        self.audits: list[dict[str, Any]] = []

    async def list_documents(self, **kwargs: Any) -> tuple[list[Any], int]:
        return [], self.total

    async def write_audit(self, **kwargs: Any) -> None:
        self.audits.append(kwargs)


@pytest.fixture()
def fake_repo() -> _FakeRepo:
    return _FakeRepo()


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_repo: _FakeRepo
) -> Iterator[TestClient]:
    settings = get_settings()
    monkeypatch.setattr(settings, "kb_config_dir", tmp_path)
    monkeypatch.setattr(settings, "kb_admin_group", "")
    app = create_app()
    app.dependency_overrides[get_db_repo] = lambda: fake_repo
    with TestClient(app) as test_client:
        yield test_client


def _create_payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "kb_code": "legal",
        "name": "法务知识库",
        "description": "合同模板与合规制度",
        "retrieval": {"top_k": 10, "rerank_top_n": 4},
    }
    body.update(overrides)
    return body


def _update_payload(**overrides: Any) -> dict[str, Any]:
    body = _create_payload()
    body.pop("kb_code")
    body.update(overrides)
    return body


def test_create_list_update_delete(
    client: TestClient, tmp_path: Path, fake_repo: _FakeRepo
) -> None:
    created = client.post(BASE, json=_create_payload())
    assert created.status_code == 201
    body = created.json()
    assert body["kb_id"] == "kb-legal"
    assert body["kb_code"] == "legal"
    assert body["collection"] == "rag_legal"
    assert body["es_index"] == "rag_legal_chunks"
    assert body["retrieval"]["top_k"] == 10
    assert (tmp_path / "legal.yaml").is_file()

    assert [item["kb_code"] for item in client.get(BASE).json()] == ["legal"]
    assert client.get(f"{BASE}/kb-legal").json()["name"] == "法务知识库"

    updated = client.put(f"{BASE}/legal", json=_update_payload(name="法务知识库 v2"))
    assert updated.status_code == 200
    assert updated.json()["name"] == "法务知识库 v2"
    assert client.get(f"{BASE}/legal").json()["name"] == "法务知识库 v2"

    deleted = client.delete(f"{BASE}/legal")
    assert deleted.status_code == 200
    assert deleted.json() == {"kb_id": "kb-legal", "kb_code": "legal"}
    assert not (tmp_path / "legal.yaml").exists()
    assert client.get(BASE).json() == []
    assert client.get(f"{BASE}/legal").status_code == 404

    actions = [audit["action"] for audit in fake_repo.audits]
    assert actions == ["kb.create", "kb.update", "kb.delete"]


def test_create_conflicts_and_validation(client: TestClient) -> None:
    assert client.post(BASE, json=_create_payload()).status_code == 201

    duplicate_code = client.post(BASE, json=_create_payload(name="重复库"))
    assert duplicate_code.status_code == 409
    assert duplicate_code.json()["code"] == "kb_conflict"

    duplicate_id = client.post(BASE, json=_create_payload(kb_code="legal2", kb_id="kb-legal"))
    assert duplicate_id.status_code == 409

    invalid_code = client.post(BASE, json=_create_payload(kb_code="Legal"))
    assert invalid_code.status_code == 400
    assert invalid_code.json()["code"] == "kb_validation_error"

    blank_name = client.post(BASE, json=_create_payload(name="   "))
    assert blank_name.status_code == 400


def test_cross_tenant_operations_denied(client: TestClient) -> None:
    created = client.post(BASE, json=_create_payload(tenant_id="other"))
    assert created.status_code == 403
    assert created.json()["code"] == "permission_denied"

    assert client.post(BASE, json=_create_payload()).status_code == 201
    moved = client.put(f"{BASE}/legal", json=_update_payload(tenant_id="other"))
    assert moved.status_code == 403


def test_update_unknown_and_immutable_identity(client: TestClient, tmp_path: Path) -> None:
    assert client.put(f"{BASE}/missing", json=_update_payload()).status_code == 404

    assert client.post(BASE, json=_create_payload()).status_code == 201
    # kb_code / kb_id 不在更新入参内，额外字段被拒绝
    with_identity = client.put(
        f"{BASE}/legal", json={**_update_payload(), "kb_code": "legal-b"}
    )
    assert with_identity.status_code == 422

    renamed = client.put(f"{BASE}/legal", json=_update_payload(name="新名字"))
    assert renamed.status_code == 200
    assert (tmp_path / "legal.yaml").is_file()
    assert client.get(f"{BASE}/legal").json()["kb_id"] == "kb-legal"


def test_delete_rejected_when_documents_exist(
    client: TestClient, tmp_path: Path, fake_repo: _FakeRepo
) -> None:
    assert client.post(BASE, json=_create_payload()).status_code == 201
    fake_repo.total = 3

    refused = client.delete(f"{BASE}/legal")
    assert refused.status_code == 409
    assert refused.json()["code"] == "kb_in_use"
    assert (tmp_path / "legal.yaml").is_file()
    assert [item["kb_code"] for item in client.get(BASE).json()] == ["legal"]
