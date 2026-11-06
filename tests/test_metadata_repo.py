"""MetadataRepo 集成测试：依赖 compose 的 PostgreSQL（标 integration，CI 可按 mark 隔离）。

表结构由 alembic 管理（见 b1-8），这里用 create_all 幂等兜底，保证单独跑测试也能通过；
teardown 只删测试自建行，不动 alembic_version 与其他数据。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import delete as sql_delete

from app.core.settings import get_settings
from app.db.models import (
    AUDIT_ACTION_CHAT_QUERY,
    AUDIT_ACTION_DOCUMENT_UPLOAD,
    DOC_STATUS_FAILED,
    DOC_STATUS_INGESTING,
    DOC_STATUS_READY,
    TASK_STATUS_PENDING,
    TASK_STATUS_RUNNING,
    TASK_STATUS_SUCCESS,
    AuditLogRow,
    Base,
    DocumentRow,
    IngestTaskRow,
)
from app.db.repo import MetadataRepo, acl_visible
from app.db.session import build_engine, build_session_factory

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture()
async def repo() -> AsyncIterator[MetadataRepo]:
    engine = build_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield MetadataRepo(build_session_factory(engine))
    await engine.dispose()


async def _get_document(engine_repo: MetadataRepo, doc_id: str) -> DocumentRow | None:
    session_factory = engine_repo._sessions
    async with session_factory() as session:
        return await session.get(DocumentRow, doc_id)


async def _get_task_row(engine_repo: MetadataRepo, task_id: str) -> IngestTaskRow | None:
    session_factory = engine_repo._sessions
    async with session_factory() as session:
        return await session.get(IngestTaskRow, task_id)


async def test_document_status_flow(repo: MetadataRepo) -> None:
    doc_id = uuid4().hex
    await repo.upsert_document_ingesting(
        doc_id=doc_id,
        kb_id="kb-pytest",
        tenant_id="default",
        source_uri=f"kb://pytest/{doc_id}.txt",
        title="pytest 文档",
    )
    row = await _get_document(repo, doc_id)
    assert row is not None
    assert row.status == DOC_STATUS_INGESTING
    assert row.chunk_count == 0

    await repo.mark_document_ready(
        doc_id, chunk_count=7, collection="rag_pytest", es_index="rag_pytest_chunks",
        raw_object_key="tenants/default/kbs/kb-pytest/docs/x/v1/a.txt",
    )
    row = await _get_document(repo, doc_id)
    assert row is not None
    assert row.status == DOC_STATUS_READY
    assert row.chunk_count == 7
    assert row.collection == "rag_pytest"

    # 覆盖式重建：upsert 同 doc_id 回到 ingesting
    await repo.upsert_document_ingesting(
        doc_id=doc_id,
        kb_id="kb-pytest",
        tenant_id="default",
        source_uri=f"kb://pytest/{doc_id}.txt",
        title="pytest 文档 v2",
        version=2,
    )
    row = await _get_document(repo, doc_id)
    assert row is not None
    assert row.status == DOC_STATUS_INGESTING
    assert row.title == "pytest 文档 v2"

    await repo.delete_document(doc_id)
    assert await _get_document(repo, doc_id) is None


async def test_document_mark_failed(repo: MetadataRepo) -> None:
    doc_id = uuid4().hex
    await repo.upsert_document_ingesting(
        doc_id=doc_id,
        kb_id="kb-pytest",
        tenant_id="default",
        source_uri=f"kb://pytest/{doc_id}.txt",
        title="pytest 失败文档",
    )
    await repo.mark_document_failed(doc_id)
    row = await _get_document(repo, doc_id)
    assert row is not None
    assert row.status == DOC_STATUS_FAILED
    await repo.delete_document(doc_id)


async def test_task_status_flow(repo: MetadataRepo) -> None:
    task_id = uuid4().hex
    await repo.create_task(
        task_id, kb_id="kb-pytest", tenant_id="default", status=TASK_STATUS_PENDING
    )
    info = await repo.get_task(task_id)
    assert info is not None
    assert info.status == TASK_STATUS_PENDING
    assert info.tenant_id == "default"
    assert info.doc_id is None

    await repo.mark_task(task_id, status=TASK_STATUS_RUNNING)
    info = await repo.get_task(task_id)
    assert info is not None
    assert info.status == TASK_STATUS_RUNNING

    await repo.mark_task(task_id, status=TASK_STATUS_SUCCESS, doc_id="doc-1")
    info = await repo.get_task(task_id)
    assert info is not None
    assert info.status == TASK_STATUS_SUCCESS
    assert info.doc_id == "doc-1"
    row = await _get_task_row(repo, task_id)
    assert row is not None
    assert row.finished_at is not None

    assert await repo.get_task(uuid4().hex) is None


async def test_list_and_get_documents(repo: MetadataRepo) -> None:
    kb_id = f"kb-list-{uuid4().hex[:8]}"
    doc_ids = [uuid4().hex for _ in range(3)]
    for index, doc_id in enumerate(doc_ids):
        await repo.upsert_document_ingesting(
            doc_id=doc_id,
            kb_id=kb_id,
            tenant_id="default",
            source_uri=f"kb://pytest/{doc_id}.txt",
            title=f"文档{index}",
        )
    await repo.mark_document_ready(
        doc_ids[0], chunk_count=3, collection="c", es_index="i", raw_object_key=None
    )

    items, total = await repo.list_documents(kb_id=kb_id, tenant_id="default")
    assert total == 3
    assert {item.doc_id for item in items} == set(doc_ids)

    ready_items, ready_total = await repo.list_documents(
        kb_id=kb_id, tenant_id="default", status=DOC_STATUS_READY
    )
    assert ready_total == 1
    assert ready_items[0].doc_id == doc_ids[0]
    assert ready_items[0].chunk_count == 3

    # 分页：每页 2 条，第二页剩 1 条
    page2, _ = await repo.list_documents(
        kb_id=kb_id, tenant_id="default", offset=2, limit=2
    )
    assert len(page2) == 1

    # 其他租户/知识库不可见
    _, other_total = await repo.list_documents(kb_id=kb_id, tenant_id="other-tenant")
    assert other_total == 0

    info = await repo.get_document(doc_ids[0])
    assert info is not None
    assert info.status == DOC_STATUS_READY
    assert await repo.get_document(uuid4().hex) is None

    for doc_id in doc_ids:
        await repo.delete_document(doc_id)


async def test_list_documents_acl_filter(repo: MetadataRepo) -> None:
    kb_id = f"kb-acl-{uuid4().hex[:8]}"
    public_doc = uuid4().hex
    finance_doc = uuid4().hex
    hr_doc = uuid4().hex
    await repo.upsert_document_ingesting(
        doc_id=public_doc, kb_id=kb_id, tenant_id="default",
        source_uri=f"kb://pytest/{public_doc}.txt", title="公开文档",
    )
    await repo.upsert_document_ingesting(
        doc_id=finance_doc, kb_id=kb_id, tenant_id="default",
        source_uri=f"kb://pytest/{finance_doc}.txt", title="财务文档",
        acl_groups=["finance"],
    )
    await repo.upsert_document_ingesting(
        doc_id=hr_doc, kb_id=kb_id, tenant_id="default",
        source_uri=f"kb://pytest/{hr_doc}.txt", title="人事文档",
        acl_groups=["hr", "recruiter"],
    )
    try:
        # 不限制：全部可见
        _, total = await repo.list_documents(kb_id=kb_id, tenant_id="default")
        assert total == 3

        # finance 组：公开 + 财务，看不到人事
        items, total = await repo.list_documents(
            kb_id=kb_id, tenant_id="default", allowed_groups=frozenset({"finance"})
        )
        assert total == 2
        assert {item.doc_id for item in items} == {public_doc, finance_doc}

        # recruiter 组：公开 + 人事（交集命中）
        items, total = await repo.list_documents(
            kb_id=kb_id, tenant_id="default", allowed_groups=frozenset({"recruiter"})
        )
        assert total == 2
        assert {item.doc_id for item in items} == {public_doc, hr_doc}

        # 无关组：只有公开文档
        items, total = await repo.list_documents(
            kb_id=kb_id, tenant_id="default", allowed_groups=frozenset({"legal"})
        )
        assert total == 1
        assert items[0].doc_id == public_doc
        assert items[0].acl_groups == []

        # 覆盖式重建：ACL 同步重置
        await repo.upsert_document_ingesting(
            doc_id=finance_doc, kb_id=kb_id, tenant_id="default",
            source_uri=f"kb://pytest/{finance_doc}.txt", title="财务文档 v2",
        )
        info = await repo.get_document(finance_doc)
        assert info is not None and info.acl_groups == []
        _, total = await repo.list_documents(
            kb_id=kb_id, tenant_id="default", allowed_groups=frozenset({"legal"})
        )
        assert total == 2

        # acl_visible 单测语义
        assert acl_visible([], frozenset({"x"})) is True
        assert acl_visible(["finance"], frozenset({"finance"})) is True
        assert acl_visible(["finance"], frozenset({"hr"})) is False
        assert acl_visible(["finance"], None) is True
    finally:
        for doc_id in (public_doc, finance_doc, hr_doc):
            await repo.delete_document(doc_id)


async def test_owner_and_retention_queries(repo: MetadataRepo) -> None:
    """合规查询：按责任人枚举文档；按采集时间筛过期文档（缺采集时间不自动清理）。"""
    kb_id = f"kb-retention-{uuid4().hex[:8]}"
    now = datetime.now(UTC)
    expired_doc = uuid4().hex
    fresh_doc = uuid4().hex
    legacy_doc = uuid4().hex

    await repo.upsert_document_ingesting(
        doc_id=expired_doc, kb_id=kb_id, tenant_id="default",
        source_uri=f"kb://pytest/{expired_doc}.txt", title="超期文档", owner="alice",
    )
    await repo.mark_document_ready(
        expired_doc, chunk_count=2, collection="c", es_index="i", raw_object_key=None,
        collected_at=now - timedelta(days=400),
    )
    await repo.upsert_document_ingesting(
        doc_id=fresh_doc, kb_id=kb_id, tenant_id="default",
        source_uri=f"kb://pytest/{fresh_doc}.txt", title="新文档", owner="bob",
    )
    await repo.mark_document_ready(
        fresh_doc, chunk_count=1, collection="c", es_index="i", raw_object_key=None,
        collected_at=now,
    )
    # 采集时间默认回填（collected_at 列 server_default=now()，NOT NULL），
    # 该行等价于"近期采集的文档"：不会被 365 天 cutoff 命中
    await repo.upsert_document_ingesting(
        doc_id=legacy_doc, kb_id=kb_id, tenant_id="default",
        source_uri=f"kb://pytest/{legacy_doc}.txt", title="近期采集文档", owner="alice",
    )

    try:
        assert set(await repo.list_document_ids_by_owner(
            kb_id=kb_id, tenant_id="default", owner="alice"
        )) == {expired_doc, legacy_doc}
        assert await repo.list_document_ids_by_owner(
            kb_id=kb_id, tenant_id="other", owner="alice"
        ) == []

        expired = await repo.list_expired_document_ids(
            kb_id=kb_id, tenant_id="default", cutoff=now - timedelta(days=365)
        )
        assert expired == [expired_doc]
        # limit 生效，便于分批清理；同 cutoff 下只有超期文档命中，近期采集文档不进入结果
        limited = await repo.list_expired_document_ids(
            kb_id=kb_id, tenant_id="default", cutoff=now - timedelta(days=365), limit=1
        )
        assert limited == [expired_doc]
    finally:
        for doc_id in (expired_doc, fresh_doc, legacy_doc):
            await repo.delete_document(doc_id)


async def test_audit_log_flow(repo: MetadataRepo) -> None:
    kb_id = f"kb-audit-{uuid4().hex[:8]}"
    await repo.write_audit(
        tenant_id="default",
        kb_id=kb_id,
        user_id="user-1",
        action=AUDIT_ACTION_DOCUMENT_UPLOAD,
        doc_id="doc-x",
        detail={"filename": "a.txt", "chunk_count": 5},
    )
    await repo.write_audit(
        tenant_id="default",
        kb_id=kb_id,
        user_id="user-2",
        action=AUDIT_ACTION_CHAT_QUERY,
        detail={"question": "报销流程？"},
    )
    # 其他租户的数据不参与本库查询
    await repo.write_audit(
        tenant_id="other", kb_id=kb_id, user_id="user-9", action=AUDIT_ACTION_CHAT_QUERY
    )

    items, total = await repo.list_audit_logs(kb_id=kb_id, tenant_id="default")
    assert total == 2
    by_action = {item.action: item for item in items}
    assert by_action[AUDIT_ACTION_CHAT_QUERY].user_id == "user-2"
    assert by_action[AUDIT_ACTION_CHAT_QUERY].detail == {"question": "报销流程？"}
    assert by_action[AUDIT_ACTION_DOCUMENT_UPLOAD].doc_id == "doc-x"
    assert by_action[AUDIT_ACTION_DOCUMENT_UPLOAD].detail == {
        "filename": "a.txt",
        "chunk_count": 5,
    }

    filtered, filtered_total = await repo.list_audit_logs(
        kb_id=kb_id, tenant_id="default", action=AUDIT_ACTION_CHAT_QUERY
    )
    assert filtered_total == 1
    assert filtered[0].doc_id is None

    # 清理测试数据
    session_factory = repo._sessions
    async with session_factory() as session, session.begin():
        await session.execute(sql_delete(AuditLogRow).where(AuditLogRow.kb_id == kb_id))


async def test_supersede_previous_versions_switches_active_version(
    repo: MetadataRepo,
) -> None:
    """Active Version 切换：新版本 ready 后，同 source_id 的旧文档退出检索。"""
    kb_id = f"kb-version-{uuid4().hex[:8]}"
    old_doc, new_doc, other_doc = uuid4().hex, uuid4().hex, uuid4().hex
    try:
        for doc_id, source_id, title in (
            (old_doc, "SRC-1", "旧版"),
            (new_doc, "SRC-1", "新版"),
            (other_doc, "SRC-2", "无关文档"),
        ):
            await repo.upsert_document_ingesting(
                doc_id=doc_id,
                kb_id=kb_id,
                tenant_id="default",
                source_uri=f"kb://pytest/{doc_id}.txt",
                title=title,
                source_id=source_id,
            )
            await repo.mark_document_ready(
                doc_id,
                chunk_count=1,
                collection="rag_pytest",
                es_index="rag_pytest_chunks",
                raw_object_key=None,
            )

        # 新版本完成双写并 ready 后，才切换 Active Version
        superseded = await repo.supersede_previous_versions(
            doc_id=new_doc, kb_id=kb_id, tenant_id="default", source_id="SRC-1"
        )
        assert superseded == 1

        active_excluded = await repo.list_superseded_doc_ids(
            kb_id=kb_id, tenant_id="default"
        )
        assert active_excluded == frozenset({old_doc})

        # 重复调用幂等：已替代的旧版不再重复计入
        assert await repo.supersede_previous_versions(
            doc_id=new_doc, kb_id=kb_id, tenant_id="default", source_id="SRC-1"
        ) == 0
        # source_id 为空时不做替代判定，避免误伤无关文档
        assert await repo.supersede_previous_versions(
            doc_id=new_doc, kb_id=kb_id, tenant_id="default", source_id=None
        ) == 0
    finally:
        for doc_id in (old_doc, new_doc, other_doc):
            await repo.delete_document(doc_id)
