"""RBAC 分层与 Transactional Outbox 测试（v1.2 / v1.4）。

- 角色解析是纯函数，直接用 Settings 覆盖授权组做表驱动断言；
- Outbox 走真实 PG（integration），验证"任务 + 事件同事务"与投递状态流转。
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterator
from typing import Any, TypeVar
from uuid import uuid4

import pytest

from app.api.deps import (
    has_auditor_role,
    has_kb_admin_role,
    has_kb_contributor_role,
    has_platform_admin_role,
)
from app.auth.base import CurrentUser
from app.core.settings import Settings, get_settings
from app.db.models import OUTBOX_STATUS_FAILED, Base
from app.db.repo import MetadataRepo
from app.db.session import build_engine, build_session_factory

_T = TypeVar("_T")


def _user(*, groups: frozenset[str] | None, kb_ids: frozenset[str] | None) -> CurrentUser:
    return CurrentUser(
        tenant_id="default",
        user_id="u-1",
        allowed_kb_ids=kb_ids,
        allowed_groups=groups,
    )


def test_roles_follow_groups_and_restricted_users() -> None:
    settings = Settings(
        kb_admin_group="rag-admins",
        kb_contributor_group="rag-contributors",
        kb_auditor_group="rag-auditors",
        platform_admin_group="rag-platform",
    )
    admin = _user(groups=frozenset({"rag-admins"}), kb_ids=frozenset({"finance"}))
    contributor = _user(groups=frozenset({"rag-contributors"}), kb_ids=frozenset({"finance"}))
    auditor = _user(groups=frozenset({"rag-auditors"}), kb_ids=frozenset({"finance"}))
    reader = _user(groups=frozenset({"rag-readers"}), kb_ids=frozenset({"finance"}))

    assert has_kb_admin_role(admin, settings)
    assert has_kb_contributor_role(admin, settings)
    assert has_auditor_role(admin, settings)

    # 贡献者能写文档，但不能管理知识库、也不能看审计
    assert has_kb_contributor_role(contributor, settings)
    assert not has_kb_admin_role(contributor, settings)
    assert not has_auditor_role(contributor, settings)

    # 审计员只读审计，不具备写权限
    assert has_auditor_role(auditor, settings)
    assert not has_kb_contributor_role(auditor, settings)

    # 普通读者没有任何管理/写入/审计权限（即使只被限制到某个 KB）
    assert not has_kb_admin_role(reader, settings)
    assert not has_kb_contributor_role(reader, settings)
    assert not has_auditor_role(reader, settings)


def test_roles_strict_mode_is_fail_closed_without_groups() -> None:
    """严格模式下未配置授权组即"无人拥有角色"，不允许空配置自动放权。"""
    settings = Settings(
        kb_admin_group="",
        kb_contributor_group="",
        kb_auditor_group="",
        platform_admin_group="",
        rbac_strict=True,
    )
    unrestricted = _user(groups=None, kb_ids=None)

    assert not has_platform_admin_role(unrestricted, settings)
    assert not has_kb_admin_role(unrestricted, settings)
    assert not has_kb_contributor_role(unrestricted, settings)
    assert not has_auditor_role(unrestricted, settings)


def test_roles_legacy_mode_keeps_unrestricted_users_working() -> None:
    """非严格模式（默认）保留单租户部署的旧行为，避免升级即锁死。"""
    settings = Settings(rbac_strict=False)
    unrestricted = _user(groups=None, kb_ids=None)
    restricted = _user(groups=None, kb_ids=frozenset({"finance"}))

    assert has_kb_admin_role(unrestricted, settings)
    assert has_kb_contributor_role(unrestricted, settings)
    assert not has_kb_admin_role(restricted, settings)
    assert not has_kb_contributor_role(restricted, settings)


# ---------------------------------------------------------------- Outbox
class _RepoClient:
    def __init__(self) -> None:
        self.engine = build_engine(get_settings().database_url)
        self.repo = MetadataRepo(build_session_factory(self.engine))
        self.loop = asyncio.new_event_loop()

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        return self.loop.run_until_complete(coro)

    def close(self) -> None:
        self.run(self.engine.dispose())
        self.loop.close()


@pytest.fixture()
def repo_client() -> Iterator[_RepoClient]:
    client = _RepoClient()
    client.run(_create_tables(client))
    yield client
    client.close()


async def _create_tables(client: _RepoClient) -> None:
    async with client.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@pytest.mark.integration
def test_create_task_with_outbox_is_atomic_and_dispatchable(repo_client: _RepoClient) -> None:
    task_id = uuid4().hex
    repo_client.run(
        repo_client.repo.create_task_with_outbox(
            task_id,
            kb_id="kb-finance-001",
            tenant_id="default",
            status="pending",
            payload={"kb_key": "finance", "filename": "a.txt"},
        )
    )

    info = repo_client.run(repo_client.repo.get_task(task_id))
    assert info is not None and info.status == "pending"

    events = repo_client.run(repo_client.repo.list_pending_outbox(limit=10))
    event = next(item for item in events if item.event_id == task_id)
    # task_id 由 repo 自动补进 payload，dispatcher 据此对齐 Celery task id
    assert event.payload["task_id"] == task_id
    assert event.payload["kb_key"] == "finance"

    repo_client.run(repo_client.repo.mark_outbox_dispatched(task_id))
    pending = repo_client.run(repo_client.repo.list_pending_outbox(limit=10))
    assert all(item.event_id != task_id for item in pending)


@pytest.mark.integration
def test_outbox_failure_counters_retire_event(repo_client: _RepoClient) -> None:
    task_id = uuid4().hex
    repo_client.run(
        repo_client.repo.create_task_with_outbox(
            task_id,
            kb_id="kb-finance-001",
            tenant_id="default",
            status="pending",
            payload={"kb_key": "finance"},
        )
    )

    for attempt in range(1, 6):
        repo_client.run(repo_client.repo.mark_outbox_failed(task_id, error=f"boom-{attempt}"))

    events = repo_client.run(repo_client.repo.list_pending_outbox(limit=100))
    assert all(item.event_id != task_id for item in events)

    async def _load_status() -> str | None:
        from app.db.models import OutboxEventRow

        async with repo_client.repo._sessions() as session:
            row = await session.get(OutboxEventRow, task_id)
            return None if row is None else row.status

    assert repo_client.run(_load_status()) == OUTBOX_STATUS_FAILED
