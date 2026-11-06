"""连接器注册表：配置类型 → 构造函数。

新增数据源时只改这里：实现 Connector protocol 并在 _BUILDERS 登记一条。
"""

from __future__ import annotations

from collections.abc import Callable

from app.connectors.base import Connector, ConnectorError
from app.connectors.local_directory import build_local_directory
from app.domain.kb_config import ConnectorConfig

_BUILDERS: dict[str, Callable[[ConnectorConfig], Connector]] = {
    "local_directory": build_local_directory,
}


def build_connector(config: ConnectorConfig) -> Connector:
    builder = _BUILDERS.get(config.type)
    if builder is None:
        raise ConnectorError(
            f"未知连接器类型 {config.type}（支持：{', '.join(sorted(_BUILDERS))}）"
        )
    return builder(config)


def supported_types() -> list[str]:
    return sorted(_BUILDERS)
