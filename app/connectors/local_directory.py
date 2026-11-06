"""本地目录连接器：把挂载目录（NAS / 共享盘 / 导出的文档包）作为知识源。

行为约定：
- 按相对路径字典序产出，保证同一目录多次同步的顺序稳定、便于复现与测试；
- 跳过隐藏文件/目录（. 开头）、超过 max_bytes 的文件、后缀不在白名单的文件；
- source_uri 使用 file://<connector_id>/<相对路径>：携带连接器身份，避免多个
  连接器挂载目录里出现同名相对路径时来源混淆/误去重；重复同步天然幂等
  （内容不变时 governance 判定复用）。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from app.connectors.base import ConnectorDocument, ConnectorError
from app.domain.kb_config import ConnectorConfig

DEFAULT_MAX_BYTES = 32 * 1024 * 1024


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


class LocalDirectoryConnector:
    """遍历本地目录并按白名单采集文件。"""

    connector_id: str

    def __init__(
        self,
        *,
        connector_id: str,
        base_path: str | Path,
        recursive: bool = True,
        include_suffixes: list[str] | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.connector_id = connector_id
        self._base = Path(base_path).expanduser()
        self._recursive = recursive
        self._suffixes = {suffix.lower() for suffix in (include_suffixes or [])}
        self._max_bytes = max_bytes

    def _iter_files(self) -> Iterator[Path]:
        root = self._base
        if not root.exists() or not root.is_dir():
            raise ConnectorError(f"连接器目录不存在或不是目录：{root}")
        candidates = root.rglob("*") if self._recursive else root.glob("*")
        paths = [
            path
            for path in candidates
            if path.is_file()
            and not _is_hidden(path.relative_to(root))
            and (not self._suffixes or path.suffix.lower() in self._suffixes)
        ]
        # 固定顺序：跨平台文件系统遍历顺序不稳定，排序后行为可预期
        paths.sort(key=lambda item: item.relative_to(root).as_posix())
        yield from paths

    def fetch(self, *, since: datetime | None = None) -> Iterator[ConnectorDocument]:
        for path in self._iter_files():
            stat = path.stat()
            if stat.st_size > self._max_bytes:
                continue
            relative = path.relative_to(self._base).as_posix()
            if since is not None and datetime.fromtimestamp(stat.st_mtime, UTC) <= since:
                continue
            yield ConnectorDocument(
                # source_uri 携带连接器身份：不同连接器的同名相对路径不再冲突
                source_uri=f"file://{self.connector_id}/{relative}",
                filename=path.name,
                title=path.stem,
                content=path.read_bytes(),
                # mtime 作为外部版本标识：治理层可与上一版 source_version 比较
                source_version=str(int(stat.st_mtime)),
            )


def build_local_directory(config: ConnectorConfig) -> LocalDirectoryConnector:
    if not config.path:
        raise ConnectorError(
            f"连接器 {config.connector_id} 缺少 path 配置（local_directory 必填）"
        )
    return LocalDirectoryConnector(
        connector_id=config.connector_id,
        base_path=config.path,
        recursive=config.recursive,
        include_suffixes=config.include_suffixes,
    )
