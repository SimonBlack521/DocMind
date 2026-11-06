"""稀疏检索（BM25）抽象：默认 Elasticsearch 实现，后续可换 OpenSearch。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class KeywordRecord:
    chunk_id: str
    doc_id: str
    kb_id: str
    tenant_id: str
    content: str
    source_uri: str
    version: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KeywordHit:
    chunk_id: str
    score: float
    content: str
    doc_id: str
    source_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


class KeywordStore(ABC):
    @abstractmethod
    def ensure_index(self) -> None:
        """索引/别名不存在则创建，已存在时绝不重建。"""

    @abstractmethod
    def upsert(self, records: Sequence[KeywordRecord]) -> int:
        """按 chunk_id 覆盖写入，返回写入条数。"""

    @abstractmethod
    def delete_by_doc(self, doc_id: str) -> int:
        """删除某文档的全部片段。"""

    @abstractmethod
    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int:
        """按 chunk_id 精确删除：增量重建只清理失效 chunk（幂等，重复删除不报错）。"""

    @abstractmethod
    def list_chunk_ids(self, *, batch_size: int = 1000) -> Iterator[list[str]]:
        """分批遍历索引内全部 chunk_id（对账用，只带 id 不带正文）。"""

    @abstractmethod
    def get_by_ids(self, chunk_ids: Sequence[str]) -> list[KeywordHit]:
        """按 chunk_id 批量回取片段（父子切片回取父块、引用定位都会用到）。"""

    def get_records_by_doc(self, doc_id: str) -> list[KeywordRecord] | None:
        """按文档取回完整记录：重复内容血缘的 ACL 并集调和时整行刷新用。

        默认返回 None 表示该实现不支持整行回取；支持方应保证记录可原样回写 upsert。
        """
        return None

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int,
        kb_id: str | None = None,
        allowed_groups: Collection[str] | None = None,
        excluded_doc_ids: Collection[str] | None = None,
    ) -> list[KeywordHit]:
        """BM25 检索，kb_id 用于库内过滤；allowed_groups 非 None 时叠加文档级 ACL 过滤。

        excluded_doc_ids 用于排除已被替代的历史版本文档（与向量路同一语义）。
        """

    @abstractmethod
    def close(self) -> None:
        """释放底层连接。"""
