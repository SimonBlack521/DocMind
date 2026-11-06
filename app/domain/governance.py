"""知识治理的纯函数：内容指纹、来源分类与版本裁决。

这些规则被 API 层与 Celery 任务共用，抽成无 IO 的纯函数便于单测，
也避免两个入口的治理逻辑出现分叉。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from urllib.parse import urlsplit

# 允许的 source_type：业务侧知识源分类，而不是文件后缀
SOURCE_TYPES = frozenset(
    {"file", "wiki", "ticket", "api", "table", "code", "presentation", "config"}
)

# 后缀 → 业务来源类型（仅作默认推断，调用方可显式覆盖）
_SOURCE_TYPE_BY_SUFFIX = {
    ".pdf": "file",
    ".docx": "file",
    ".txt": "file",
    ".pptx": "presentation",
    ".xlsx": "table",
    ".xlsm": "table",
    ".csv": "table",
    ".json": "api",
    ".html": "wiki",
    ".htm": "wiki",
    ".markdown": "wiki",
    ".md": "wiki",
}

_WHITESPACE = re.compile(r"\s+")


def normalize_content(content: str) -> str:
    """归一化正文：统一空白，保证同一内容不同换行/缩进得到同一指纹。"""
    return _WHITESPACE.sub(" ", content).strip()


def compute_content_hash(content: str) -> str:
    """内容指纹：归一化正文的 sha256。

    - 用于跨来源去重（同一份材料从 Wiki 与 PDF 导出时正文一致即视为重复）；
    - 用于检测内容漂移（同 doc_id 重新上传但哈希变化时必须升版本）。
    """
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def infer_source_type(source_uri: str, suffix: str | None = None) -> str:
    """推断业务来源类型；未知后缀回退为 file，调用方可通过表单显式指定。"""
    resolved = (suffix or PurePosixPath(urlsplit(source_uri).path).suffix).lower()
    return _SOURCE_TYPE_BY_SUFFIX.get(resolved, "file")


def normalize_source_type(value: str | None, *, source_uri: str, suffix: str | None) -> str:
    """校验并归一化调用方传入的 source_type；空值走后缀推断。"""
    if not value:
        return infer_source_type(source_uri, suffix)
    normalized = value.strip().lower()
    if normalized not in SOURCE_TYPES:
        raise ValueError(f"非法 source_type：{value}，可选 {sorted(SOURCE_TYPES)}")
    return normalized


def resolve_version(
    existing_hash: str | None, existing_version: int, new_hash: str
) -> int:
    """版本裁决：内容变化自动升版本，内容不变沿用原版本。

    为什么必须升版本：chunk_id 由 doc_id + version + index 生成，若版本不变则
    新旧 chunk_id 完全相同，覆盖写入时无法区分"重建"与"新增"，也破坏了历史引用。
    """
    if existing_hash is None:
        return max(existing_version, 1)
    if existing_hash != new_hash:
        return existing_version + 1
    return existing_version


def compute_config_hash(values: Mapping[str, object]) -> str:
    """把切块/解析参数序列化成稳定指纹，随文档留存以支持"解析可复现"。

    同一份文件在参数变更后重跑会得到不同 chunk，只有把当时的参数快照落库，
    事后才能判断两次结果差异究竟来自内容变化还是参数变化。
    """
    payload = json.dumps(dict(values), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
