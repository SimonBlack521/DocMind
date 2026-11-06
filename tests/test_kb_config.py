"""知识库配置测试：加载、命名约定、注册表唯一性。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import KBConfigError, KBNotFoundError
from app.domain.kb_config import (
    KBRegistry,
    delete_kb_config,
    discover_kb_configs,
    find_kb_config_path,
    load_kb_config,
    save_kb_config,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "kbs"


def test_load_finance_config() -> None:
    config = load_kb_config(CONFIG_DIR / "finance.yaml")
    assert config.kb_code == "finance"
    assert config.milvus_collection == "rag_finance"
    assert config.es_index == "rag_finance_chunks"
    assert config.es_alias == "rag_finance_search"
    assert config.doc_acl_enabled is False


def test_discover_configs_sorted_by_code() -> None:
    # 目录由运维维护，可能随部署新增知识库（如压测库）；只校验内置库存在且整体有序
    codes = [config.kb_code for config in discover_kb_configs(CONFIG_DIR)]
    assert codes[:2] == ["finance", "hr"]
    assert codes == sorted(codes)


def test_registry_resolves_by_id_and_code() -> None:
    registry = KBRegistry.from_directory(CONFIG_DIR)
    assert registry.resolve("finance").kb_id == "kb-finance-001"
    assert registry.resolve("kb-hr-001").kb_code == "hr"
    assert len(registry) >= 2


def test_registry_unknown_key_raises() -> None:
    registry = KBRegistry.from_directory(CONFIG_DIR)
    with pytest.raises(KBNotFoundError):
        registry.resolve("unknown")


def test_invalid_kb_code_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "kb_id: kb-bad\nkb_code: Finance\nname: 非法知识库\n", encoding="utf-8"
    )
    with pytest.raises(KBConfigError):
        load_kb_config(path)


def test_duplicate_kb_code_rejected() -> None:
    finance = load_kb_config(CONFIG_DIR / "finance.yaml")
    duplicate = finance.model_copy(update={"kb_id": "kb-finance-002"})
    with pytest.raises(KBConfigError):
        KBRegistry([finance, duplicate])


def test_save_find_delete_roundtrip(tmp_path: Path) -> None:
    config = load_kb_config(CONFIG_DIR / "finance.yaml").model_copy(
        update={"kb_id": "kb-copy-001", "kb_code": "finance_copy", "name": "副本知识库"}
    )
    path = save_kb_config(tmp_path, config)
    assert path == tmp_path / "finance_copy.yaml"
    assert load_kb_config(path) == config
    assert find_kb_config_path(tmp_path, "finance_copy") == path
    assert find_kb_config_path(tmp_path, "missing") is None

    # 更新回写同一文件，人工命名保持不变
    save_kb_config(tmp_path, config.model_copy(update={"name": "改名后"}), path=path)
    assert load_kb_config(path).name == "改名后"

    delete_kb_config(path)
    assert find_kb_config_path(tmp_path, "finance_copy") is None
    # 幂等：重复删除不报错
    delete_kb_config(path)
