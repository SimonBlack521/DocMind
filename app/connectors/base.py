"""v2.0 Connector Framework：外部知识源统一拉取入口。

设计边界：
- 连接器只负责"取数"（列出文档、读取字节、给出 source_uri/治理元数据），
  不直接写索引、不绕过 governance / 质量门禁 / 增量 Diff；
- 所有连接器输出统一的 ConnectorDocument，入库由现有异步入库任务复用；
- 新增数据源（SQL/API/对象存储）只需实现 Connector protocol 并注册到 registry。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class ConnectorError(Exception):
    """连接器配置或取数失败；由调用方决定是告警还是终止同步。"""


@dataclass(slots=True, frozen=True)
class ConnectorDocument:
    """一条待入库的外部文档。

    source_uri 是幂等键：同一连接器同一路径重复同步会命中同一 doc_id，
    从而走 governance 的"内容未变则复用"分支，不会产生重复文档。
    """

    source_uri: str
    filename: str
    content: bytes
    title: str | None = None
    # 连接器可从外部系统读到的版本标识（mtime / etag / revision），用于治理比较
    source_version: str | None = None


class Connector(Protocol):
    """连接器协议：实现方只需提供 id 与增量/全量拉取。"""

    connector_id: str

    def fetch(self, *, since: datetime | None = None) -> Iterator[ConnectorDocument]:
        """产出文档；since 为 None 表示全量扫描。"""
        ...
