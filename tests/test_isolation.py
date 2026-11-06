"""知识库隔离测试：命名约定、过滤表达式、注册表解析（不依赖真实服务）。"""

from __future__ import annotations

import pytest

from app.core.errors import KBConfigError, KBNotFoundError
from app.domain.kb_config import KBConfig, KBRegistry
from app.storage.vector_base import normalize_acl_groups, scope_expression


def _kb(code: str, tenant: str = "default") -> KBConfig:
    return KBConfig(kb_id=f"kb-{code}-001", kb_code=code, name=f"{code} 知识库", tenant_id=tenant)


def test_each_kb_has_isolated_collection_and_index() -> None:
    finance = _kb("finance")
    hr = _kb("hr")

    assert finance.milvus_collection == "rag_finance"
    assert finance.es_index == "rag_finance_chunks"
    assert finance.es_alias == "rag_finance_search"
    assert finance.milvus_collection != hr.milvus_collection
    assert finance.es_index != hr.es_index
    assert finance.es_alias != hr.es_alias


def test_scope_expression_filters_tenant_and_kb() -> None:
    expr = scope_expression("default", "kb-finance-001")

    assert 'tenant_id == "default"' in expr
    assert 'kb_id == "kb-finance-001"' in expr
    assert "json_contains_any" not in expr


def test_scope_expression_with_allowed_groups() -> None:
    expr = scope_expression("default", "kb-finance-001", frozenset({"finance", "hr"}))

    assert 'json_contains_any(metadata["acl_groups"]' in expr
    # 公开哨兵恒在查询组列表里
    assert '"*"' in expr
    assert '"finance"' in expr and '"hr"' in expr


def test_scope_expression_escapes_group_names() -> None:
    expr = scope_expression("default", "kb-1", frozenset({'bad"group'}))

    # 组名中的引号被 JSON 转义，不能注出数组字面量
    assert 'bad\\"group' in expr


def test_normalize_acl_groups() -> None:
    # 空 / None → 公开哨兵
    assert normalize_acl_groups(None) == ["*"]
    assert normalize_acl_groups([]) == ["*"]
    assert normalize_acl_groups(["  ", ""]) == ["*"]
    # 去重、去空白、剔除伪造哨兵，排序稳定
    assert normalize_acl_groups([" hr ", "finance", "hr", "*"]) == ["finance", "hr"]


def test_registry_resolves_by_id_and_code() -> None:
    registry = KBRegistry([_kb("finance"), _kb("hr")])

    assert registry.resolve("finance").kb_id == "kb-finance-001"
    assert registry.resolve("kb-hr-001").kb_code == "hr"
    assert {kb.kb_code for kb in registry.list_enabled()} == {"finance", "hr"}


def test_registry_rejects_unknown_and_duplicate_kb() -> None:
    registry = KBRegistry([_kb("finance")])
    with pytest.raises(KBNotFoundError):
        registry.resolve("unknown")

    with pytest.raises(KBConfigError):
        KBRegistry([_kb("finance"), _kb("finance")])
