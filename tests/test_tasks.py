"""异步入库任务集成测试：真实 PG / Milvus / ES / S3 / Ollama（标 integration）。

通过 task.apply() 在当前进程内同步执行任务（与 eager 等效），
验证 ingest_tasks / documents 两表的状态流转与检索侧真实入库结果。
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterator
from typing import Any, TypeVar
from uuid import uuid4

import pytest

from app.core.errors import KBNotFoundError
from app.core.settings import get_settings
from app.db.models import (
    DOC_STATUS_DUPLICATE,
    DOC_STATUS_READY,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_SUCCESS,
    Base,
    DocumentRow,
)
from app.db.repo import MetadataRepo
from app.db.session import build_engine, build_session_factory
from app.domain.kb_config import KBRegistry
from app.storage.vector_base import scope_expression
from app.tasks.ingest import _get_resources, ingest_document_task

pytestmark = pytest.mark.integration

KB_CODE = "finance"
_T = TypeVar("_T")


class _RepoClient:
    """同步测试里操作异步 repo：复用单一事件循环，避免跨循环使用连接池。"""

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


def _task_kwargs(source_uri: str, content: str | None = None) -> dict[str, Any]:
    payload = content or (
        "企业差旅报销标准：住宿每晚不超过 500 元，高铁二等座，"
        "出差补助每天 100 元，需提供发票。" * 3
    )
    return {
        "kb_key": KB_CODE,
        "payload": payload.encode("utf-8"),
        "staging_key": None,
        "filename": source_uri.rsplit("/", 1)[-1],
        "title": "pytest 异步入库文档",
        "source_uri": source_uri,
    }


def _make_task_kwargs(source_uri: str) -> dict[str, Any]:
    return _task_kwargs(source_uri)


def _new_task(repo_client: _RepoClient, kb: Any) -> str:
    task_id = uuid4().hex
    repo_client.run(
        repo_client.repo.create_task(
            task_id, kb_id=kb.kb_id, tenant_id=kb.tenant_id, status=TASK_STATUS_PENDING
        )
    )
    return task_id


def test_ingest_document_task_success(repo_client: _RepoClient) -> None:
    task_id = uuid4().hex
    source_uri = f"kb://{KB_CODE}/pytest-{task_id}.txt"
    kb = KBRegistry.from_directory(get_settings().kb_config_dir).resolve(KB_CODE)
    repo_client.run(
        repo_client.repo.create_task(
            task_id, kb_id=kb.kb_id, tenant_id=kb.tenant_id, status=TASK_STATUS_PENDING
        )
    )

    doc_id: str | None = None
    raw_key: str | None = None
    try:
        result = ingest_document_task.apply(
            kwargs=_make_task_kwargs(source_uri), task_id=task_id
        )
        assert result.successful()
        doc_id = result.result["doc_id"]
        assert result.result["status"] == TASK_STATUS_SUCCESS
        assert result.result["chunk_count"] > 0

        info = repo_client.run(repo_client.repo.get_task(task_id))
        assert info is not None
        assert info.status == TASK_STATUS_SUCCESS
        assert info.doc_id == doc_id

        async def _load_document() -> DocumentRow | None:
            async with repo_client.repo._sessions() as session:
                return await session.get(DocumentRow, doc_id)

        row = repo_client.run(_load_document())
        assert row is not None
        assert row.status == DOC_STATUS_READY
        assert row.chunk_count == result.result["chunk_count"]
        assert row.collection == kb.milvus_collection
        raw_key = row.raw_object_key

        # 治理字段随 chunk 一起落索引：检索侧（向量库 metadata）能直接读到，
        # 这是权威度/时效加权的数据来源；结构定位信息同样必须可回查
        resources = _get_resources()
        runtime = resources.runtime_registry.runtime(kb)
        # 检索范围显式限定到本次入库的 doc_id：开发库可能已有同主题文档，
        # 只按 KB 过滤时 top_k 名额会被历史数据挤占，导致断言随环境数据漂移
        hits = runtime.vector_store.search(
            runtime.embedding.embed_query("差旅报销标准"),
            top_k=10,
            expr=f"({scope_expression(kb.tenant_id, kb.kb_id)}) and doc_id == '{doc_id}'",
        )
        matched = [hit for hit in hits if hit.doc_id == doc_id]
        assert matched, "入库后应能在该知识库范围内检索到新文档的 chunk"
        metadata = matched[0].metadata
        assert metadata["source_type"] == "file"
        assert metadata["authority_level"] == 0
        assert metadata["collected_at"]
        assert metadata["char_start"] == 0
        assert "content_hash" in metadata and "section_path" in metadata
    finally:
        if doc_id is not None:
            # 清理检索侧与元数据，避免污染开发环境数据
            resources = _get_resources()
            resources.pipeline.delete_document(kb, doc_id)
            repo_client.run(repo_client.repo.delete_document(doc_id))
            if raw_key:
                resources.object_store.delete(raw_key)


def test_ingest_document_task_governs_reuse_version_and_duplicates(
    repo_client: _RepoClient,
) -> None:
    """治理裁决：同内容复用 → 内容变化升版本 → 跨来源重复只登记血缘。"""
    kb = KBRegistry.from_directory(get_settings().kb_config_dir).resolve(KB_CODE)
    resources = _get_resources()
    base = f"kb://{KB_CODE}/pytest-gov-{uuid4().hex}"
    uri_a = f"{base}-a.txt"
    uri_dup = f"{base}-dup.txt"
    content_a = "知识治理测试正文 A：差旅报销需在 15 日内提交。" * 4
    content_b = "知识治理测试正文 B：住宿标准调整为每晚 600 元。" * 4

    doc_ids: list[str] = []
    raw_keys: list[str] = []
    try:
        # 1) 首次入库：内容指纹、解析器身份、质量报告落库，版本为 1
        first_task = _new_task(repo_client, kb)
        first = ingest_document_task.apply(
            kwargs=_task_kwargs(uri_a, content_a), task_id=first_task
        )
        assert first.successful()
        doc_a = first.result["doc_id"]
        doc_ids.append(doc_a)
        assert first.result["version"] == 1

        row = repo_client.run(repo_client.repo.get_document(doc_a))
        assert row is not None
        assert row.status == DOC_STATUS_READY
        assert row.content_hash is not None
        assert row.parser_name == "builtin-text"
        assert row.parse_config_hash is not None
        assert row.parse_report["passed"] is True
        assert row.collected_at is not None
        assert row.source_type == "file"
        if row.raw_object_key:
            raw_keys.append(row.raw_object_key)

        # 2) 同内容重传：命中复用，不重新索引，版本不变
        reuse_task = _new_task(repo_client, kb)
        reuse = ingest_document_task.apply(
            kwargs=_task_kwargs(uri_a, content_a), task_id=reuse_task
        )
        assert reuse.successful()
        assert reuse.result["reused"] is True
        assert reuse.result["version"] == 1

        # 3) 内容变化：自动升版本并覆盖重建
        change_task = _new_task(repo_client, kb)
        changed = ingest_document_task.apply(
            kwargs=_task_kwargs(uri_a, content_b), task_id=change_task
        )
        assert changed.successful()
        assert changed.result["version"] == 2
        row = repo_client.run(repo_client.repo.get_document(doc_a))
        assert row is not None
        assert row.version == 2
        assert row.chunk_count == changed.result["chunk_count"]

        # 4) 不同来源但正文相同：登记 duplicate_of，不重复建索引
        dup_task = _new_task(repo_client, kb)
        duplicate = ingest_document_task.apply(
            kwargs=_task_kwargs(uri_dup, content_b), task_id=dup_task
        )
        assert duplicate.successful()
        assert duplicate.result["duplicate_of"] == doc_a
        doc_dup = duplicate.result["doc_id"]
        doc_ids.append(doc_dup)
        dup_row = repo_client.run(repo_client.repo.get_document(doc_dup))
        assert dup_row is not None
        assert dup_row.status == DOC_STATUS_DUPLICATE
        assert dup_row.duplicate_of == doc_a
    finally:
        for doc_id in doc_ids:
            resources.pipeline.delete_document(kb, doc_id)
            repo_client.run(repo_client.repo.delete_document(doc_id))
        for raw_key in raw_keys:
            resources.object_store.delete(raw_key)


def test_ingest_document_task_unknown_kb_marks_failed(repo_client: _RepoClient) -> None:
    task_id = uuid4().hex
    repo_client.run(
        repo_client.repo.create_task(
            task_id, kb_id="kb-not-exists", tenant_id="default", status=TASK_STATUS_PENDING
        )
    )
    kwargs = _make_task_kwargs(f"kb://not-exists/pytest-{task_id}.txt")
    kwargs["kb_key"] = "not-exists"

    with pytest.raises(KBNotFoundError):
        ingest_document_task.apply(kwargs=kwargs, task_id=task_id)

    info = repo_client.run(repo_client.repo.get_task(task_id))
    assert info is not None
    assert info.status == TASK_STATUS_FAILED
