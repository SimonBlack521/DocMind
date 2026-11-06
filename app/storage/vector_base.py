"""向量存储抽象：业务侧只依赖 VectorStore，Milvus 可替换为 PgVector / OpenSearch。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class VectorRecord:
    chunk_id: str
    vector: list[float]
    content: str
    doc_id: str
    kb_id: str
    tenant_id: str
    source_uri: str
    version: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    score: float
    content: str
    doc_id: str
    source_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


# 文档级 ACL 约定：chunk metadata["acl_groups"] 列出可访问的组；公开文档（业务语义为
# 空列表）在存储侧统一写成哨兵 ["*"]，因为 Milvus 2.4 的 JSON 表达式无法判断空数组，
# 查询时把 "*" 并入用户组列表即可同时命中公开文档。ES 侧沿用同一约定（terms 查询）。
ACL_PUBLIC_GROUP = "*"


def normalize_acl_groups(acl_groups: Sequence[str] | None) -> list[str]:
    """入库前归一化：去空白/去重/剔除哨兵（防伪造公开），空列表落到公开哨兵。"""
    cleaned = sorted(
        {
            group.strip()
            for group in acl_groups or []
            if group.strip() and group.strip() != ACL_PUBLIC_GROUP
        }
    )
    return cleaned or [ACL_PUBLIC_GROUP]


def merge_lineage_acl(groups_list: Sequence[Sequence[str]]) -> list[str]:
    """重复内容血缘的并集 ACL（存储侧哨兵形式）：任一来源公开则内容公开，否则取并集。

    语义（清单第二优先级第 9 项）：同一物理内容被多个逻辑文档共享时，
    内容可见性 = "任一来源可见"；各来源在 PG 中仍保留独立 ACL 与引用关系。
    入参是 PG 侧业务语义（空列表 = 公开），返回存储侧形式（公开 = 哨兵 ["*"]）。
    """
    merged: set[str] = set()
    for groups in groups_list:
        cleaned = {
            group.strip()
            for group in groups or []
            if group.strip() and group.strip() != ACL_PUBLIC_GROUP
        }
        if not cleaned:
            # 任一来源公开（空 ACL），内容整体公开
            return [ACL_PUBLIC_GROUP]
        merged |= cleaned
    return sorted(merged) or [ACL_PUBLIC_GROUP]


def scope_expression(
    tenant_id: str,
    kb_id: str,
    allowed_groups: Collection[str] | None = None,
    excluded_doc_ids: Collection[str] | None = None,
) -> str:
    """向量库过滤表达式，双重兜底：每个 KB 独立 collection，同时带 tenant/kb 条件。

    allowed_groups 为 None 表示不限制（dev / IdP 未下发组）；否则要求 chunk 的
    metadata.acl_groups 与用户组有交集（公开哨兵 "*" 恒在查询组列表里）。

    excluded_doc_ids 是"已被新版本替代"的文档集合：过滤同样下推到向量库内部，
    保证 top_k 名额不会被历史版本片段占用（而不是召回后再丢弃）。
    """
    expr = f'tenant_id == "{tenant_id}" and kb_id == "{kb_id}"'
    if allowed_groups is not None:
        # json.dumps 负责字符串转义，避免组名中的引号注入过滤表达式
        visible_groups = {group.strip() for group in allowed_groups if group.strip()}
        groups = json.dumps(sorted(visible_groups | {ACL_PUBLIC_GROUP}))
        expr = f'{expr} and json_contains_any(metadata["acl_groups"], {groups})'
    if excluded_doc_ids:
        ids = json.dumps(sorted({doc_id for doc_id in excluded_doc_ids if doc_id}))
        expr = f"{expr} and doc_id not in {ids}"
    return expr


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, dim: int) -> None:
        """集合不存在则创建；存在则校验维度，维度不一致直接报错。"""

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord], *, dim: int) -> int:
        """按 chunk_id 覆盖写入，返回写入条数。"""

    @abstractmethod
    def delete_by_doc(self, doc_id: str) -> int:
        """删除某文档的全部片段（重建/删除文档时使用）。"""

    @abstractmethod
    def get_vectors_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        """按 chunk_id 批量取向量：增量入库时未变 chunk 直接复用旧向量，省 Embedding 算力。

        返回 {chunk_id: vector}；索引里查不到的 id 静默缺席（调用方视为"需重算"）。
        """

    @abstractmethod
    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int:
        """按 chunk_id 精确删除：增量重建只清理失效 chunk，不误删同文档的新版本片段。

        幂等：重复删除不存在的 id 返回 0，不报错。
        """

    def get_records_by_doc(self, doc_id: str) -> list[VectorRecord] | None:
        """按文档取回完整记录（含向量）：重复内容血缘的 ACL 并集调和时整行刷新用。

        默认返回 None 表示该实现不支持整行回取；支持方应保证记录可原样回写 upsert。
        """
        return None

    @abstractmethod
    def list_chunk_ids(self, *, batch_size: int = 1000) -> Iterator[list[str]]:
        """分批遍历索引内全部 chunk_id：对账任务（reconciler）发现孤儿向量用。

        只产出 id，不带向量，百万级数据量下内存可控。
        """

    def reconcile_index_params(self) -> bool:
        """维护窗口的索引参数对账（允许原地重建索引），返回是否发生变更。

        入库热路径只做漂移检测与告警；重建集中在此入口由 rag.reconcile_index
        定时任务触发。默认空实现（返回 False），支持原地重建的实现应覆盖。
        """
        return False

    @abstractmethod
    def search(
        self, vector: Sequence[float], *, top_k: int, expr: str | None = None
    ) -> list[VectorHit]:
        """相似度检索，expr 用于租户/知识库过滤。"""

    @abstractmethod
    def close(self) -> None:
        """释放底层连接。"""
