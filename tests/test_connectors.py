"""Connector Framework 单元测试（v2.0）。

只覆盖"发现与读取"层：顺序稳定、隐藏文件/后缀/大小过滤、since 增量、
未知类型报错、KBConfig 解析连接器段落。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.connectors.base import ConnectorError
from app.connectors.local_directory import LocalDirectoryConnector
from app.connectors.registry import build_connector, supported_types
from app.domain.kb_config import ConnectorConfig, KBConfig


def _connector(root: Path, **kwargs: object) -> LocalDirectoryConnector:
    return LocalDirectoryConnector(
        connector_id="docs",
        base_path=root,
        include_suffixes=[".md", ".txt"],
        **kwargs,  # type: ignore[arg-type]
    )


def test_fetch_orders_by_relative_path_and_filters(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "skip.exe").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "c.md").write_text("c", encoding="utf-8")
    (nested / ".secret.md").write_text("secret", encoding="utf-8")

    documents = list(_connector(tmp_path).fetch())

    # v2.1（清单 P1-05）：source_uri 携带连接器身份 file://<connector_id>/<相对路径>，
    # 避免多个连接器同名相对路径时来源冲突；断言随行为变更同步更新
    assert [doc.source_uri for doc in documents] == [
        "file://docs/a.txt",
        "file://docs/b.md",
        "file://docs/sub/c.md",
    ]
    assert documents[1].filename == "b.md"
    # mtime 作为外部版本标识，治理层据此判断是否需要升版本
    assert documents[0].source_version is not None


def test_fetch_respects_since_and_max_bytes(tmp_path: Path) -> None:
    old = tmp_path / "old.md"
    old.write_text("old", encoding="utf-8")
    new = tmp_path / "new.md"
    new.write_text("new", encoding="utf-8")
    big = tmp_path / "big.md"
    big.write_text("x" * 100, encoding="utf-8")

    connector = LocalDirectoryConnector(
        connector_id="docs",
        base_path=tmp_path,
        include_suffixes=[".md"],
        max_bytes=20,
    )
    cutoff = datetime.now(UTC) - timedelta(seconds=1)
    since_docs = list(connector.fetch(since=cutoff))
    # 刚写入的文件 mtime 一定晚于 cutoff；大文件被 max_bytes 过滤
    assert {doc.filename for doc in since_docs} == {"old.md", "new.md"}
    assert "big.md" not in {doc.filename for doc in since_docs}

    # 全量模式同样过滤大文件
    assert {doc.filename for doc in connector.fetch()} == {"old.md", "new.md"}


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ConnectorError):
        list(_connector(tmp_path / "not-exists").fetch())


def test_registry_rejects_unknown_type() -> None:
    config = ConnectorConfig(connector_id="x", type="sql", path="/tmp")
    with pytest.raises(ConnectorError):
        build_connector(config)
    assert "local_directory" in supported_types()


def test_kb_config_parses_connectors_block() -> None:
    kb = KBConfig.model_validate(
        {
            "kb_id": "kb-finance-001",
            "kb_code": "finance",
            "name": "财务知识库",
            "connectors": [
                {
                    "connector_id": "finance-share",
                    "type": "local_directory",
                    "path": "D:/data/finance",
                    "owner": "finance-team",
                    "authority_level": 2,
                }
            ],
        }
    )
    assert kb.connectors[0].connector_id == "finance-share"
    assert kb.connectors[0].authority_level == 2

    built = build_connector(kb.connectors[0])
    assert built.connector_id == "finance-share"


def test_connector_requires_path() -> None:
    with pytest.raises(ConnectorError):
        build_connector(ConnectorConfig(connector_id="empty", type="local_directory"))
