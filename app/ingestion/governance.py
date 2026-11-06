"""治理裁决：入库前的版本化决策与复用判断。

同步 API 与异步 Celery 共享同一份逻辑，避免两边各自维护漂移：
- 内容未变、ACL/治理未变、上次已 ready → 幂等复用，不重复索引；
- 同库已有相同指纹文档（不同来源）→ 只登记 duplicate_of 血缘；
- 内容变化 / ACL 变化 / 治理变化 → 自动升版本，整篇覆盖重建。

所有入参必须是**标准化后**的值（空字符串已转 None、source_type 已 normalize_source_type），
避免"空字符串 vs 'file'"这种比较误差。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from app.db.models import DOC_STATUS_READY

Decision = Literal["reuse", "duplicate", "ingest_new_version"]


@dataclass(frozen=True)
class GovernanceDecision:
    decision: Decision


def compare_governance(
    *,
    source_id: str | None,
    source_type: str,
    owner: str | None,
    source_version: str | None,
    authority_level: int,
    existing_source_id: str | None,
    existing_source_type: str | None,
    existing_owner: str | None,
    existing_source_version: str | None,
    existing_authority_level: int,
) -> bool:
    """返回 True 表示治理字段与现有文档有变化——必须重建，不能复用。

    所有入参必须是**标准化后**的值，避免比较误差。
    """
    return (
        (source_id or None) != (existing_source_id or None)
        or source_type != (existing_source_type or "file")
        or (owner or None) != (existing_owner or None)
        or (source_version or None) != (existing_source_version or None)
        or authority_level != existing_authority_level
    )


def acl_changed(
    new_acl: list[str] | None, existing_acl: list[str] | None
) -> bool:
    """ACL 排序比较：None/[] 都表示公开；三态语义保持一致。"""
    return sorted(new_acl or []) != sorted(existing_acl or [])


def _normalize_datetime(value: datetime | None) -> str:
    """统一为 UTC 秒级 ISO 串：DB 读回值与请求入参的时区表示可能不同。"""
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds")


def validity_changed(
    *,
    new_valid_from: datetime | None,
    new_valid_to: datetime | None,
    existing_valid_from: datetime | None,
    existing_valid_to: datetime | None,
) -> bool:
    """有效期变化必须重建：已过期文档不能被当作有效缓存继续复用。"""
    return (
        _normalize_datetime(new_valid_from) != _normalize_datetime(existing_valid_from)
        or _normalize_datetime(new_valid_to) != _normalize_datetime(existing_valid_to)
    )


def parse_config_changed(
    new_config_hash: str | None, existing_config_hash: str | None
) -> bool:
    """切块配置 / Embedding 版本指纹变化必须重建。

    指纹把 chunking（块大小、重叠、父子切片）与 embedding_version 一起哈希；
    调整切块参数或更换同维度模型后，旧 chunk_id/旧向量不再语义等价的直接复用。
    历史文档没有记录指纹时按"已变化"处理（多做一次重建换取正确性）。
    """
    return (new_config_hash or "") != (existing_config_hash or "")


def decide(
    *,
    new_content_hash: str,
    new_acl: list[str] | None,
    new_source_id: str | None,
    new_source_type: str,
    new_owner: str | None,
    new_source_version: str | None,
    new_authority_level: int,
    existing_status: str | None,
    existing_content_hash: str | None,
    existing_acl: list[str] | None,
    existing_source_id: str | None,
    existing_source_type: str | None,
    existing_owner: str | None,
    existing_source_version: str | None,
    existing_authority_level: int,
    new_valid_from: datetime | None = None,
    new_valid_to: datetime | None = None,
    existing_valid_from: datetime | None = None,
    existing_valid_to: datetime | None = None,
    new_config_hash: str | None = None,
    existing_config_hash: str | None = None,
) -> GovernanceDecision:
    """治理裁决公共入口。

    调用方先查出 existing 文档，再把**标准化后**的新旧值一起传进来，本函数
    只做纯判断、不访问存储，sync/async 两条链路都复用这一份逻辑。

    完整复用条件（缺一不可）：内容指纹一致、ACL 一致、治理字段一致、
    有效期一致、切块配置与 Embedding 版本指纹一致、上次已 ready。
    """
    if (
        existing_status is not None
        and existing_status == DOC_STATUS_READY
        and existing_content_hash == new_content_hash
        and not acl_changed(new_acl, existing_acl)
        and not compare_governance(
            source_id=new_source_id,
            source_type=new_source_type,
            owner=new_owner,
            source_version=new_source_version,
            authority_level=new_authority_level,
            existing_source_id=existing_source_id,
            existing_source_type=existing_source_type,
            existing_owner=existing_owner,
            existing_source_version=existing_source_version,
            existing_authority_level=existing_authority_level,
        )
        and not validity_changed(
            new_valid_from=new_valid_from,
            new_valid_to=new_valid_to,
            existing_valid_from=existing_valid_from,
            existing_valid_to=existing_valid_to,
        )
        and not parse_config_changed(new_config_hash, existing_config_hash)
    ):
        return GovernanceDecision(decision="reuse")
    return GovernanceDecision(decision="ingest_new_version")
