"""统一证据结构（v2.1）：文档、SQL、API 三种来源用同一种方式描述"证据从哪来"。

为什么需要统一：文档 RAG 能给出文档名/片段/版本，但 SQL 结果过去只是把文本
交给 LLM，缺少可追溯的来源证明。审计与前端引用需要知道：
- 这条结论来自文档、数据库还是实时接口；
- 具体定位（chunk / 表+SQL / endpoint）；
- 观测时间与版本，便于判断时效性与新旧口径。

字段语义：
- source_type：document / database / api；
- source_id：文档 doc_id / 数据库标识 / endpoint_id；
- locator：chunk_id、可执行 SQL、API 路径；
- snippet：进入答案的最小证据片段（可为摘要）；
- provenance：补充原始信息（source_uri、SQL、请求摘要等）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

EvidenceSource = Literal["document", "database", "api"]


@dataclass(frozen=True, slots=True)
class Evidence:
    """一条可追溯证据：来源类型 + 定位 + 片段 + 观测时间与版本。"""

    source_type: EvidenceSource
    source_id: str
    locator: str
    snippet: str
    observed_at: datetime
    version: str = ""
    provenance: dict[str, object] = field(default_factory=dict)
