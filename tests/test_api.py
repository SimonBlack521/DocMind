"""API 冒烟测试：健康检查、知识库列表、未知识别库 404（不访问外部服务）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.settings import Settings
from app.main import create_app


def test_health_and_kb_listing() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["kb_count"] >= 1

        response = client.get("/api/v1/kbs")
        assert response.status_code == 200
        assert {item["kb_code"] for item in response.json()} >= {"finance", "hr"}


def test_unknown_kb_returns_404() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/kbs/not-exists")
        assert response.status_code == 404
        assert response.json()["code"] == "kb_not_found"


def test_production_rejects_dev_auth_and_celery_eager() -> None:
    with pytest.raises(ValueError, match="AUTH_MODE=dev"):
        Settings(app_env="prod", auth_mode="dev")
    with pytest.raises(ValueError, match="CELERY_EAGER"):
        Settings(app_env="prod", auth_mode="oidc", celery_eager=True)


def test_production_forces_strict_rbac_and_requires_admin_group() -> None:
    """生产环境不允许"空授权组 = 人人管理员"的默认放权。"""
    with pytest.raises(ValueError, match="KB_ADMIN_GROUP"):
        Settings(app_env="prod", auth_mode="oidc")

    settings = Settings(
        app_env="prod", auth_mode="oidc", kb_admin_group="rag-admins", rbac_strict=False
    )
    assert settings.rbac_strict is True
