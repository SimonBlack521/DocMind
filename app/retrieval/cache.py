"""检索结果热点缓存：只缓存召回 chunks（不缓存答案），重复问题免走 Embedding/Milvus/ES。

设计约束：
- key 由 租户 + 知识库 + 问题 + 检索参数 + 用户组 + 检索配置哈希 共同决定，配置变更自动失效；
- allowed_groups 必须进 key（安全硬约束，不是性能优化）：不同用户组能看到的文档
  集合不同，若共用缓存，高权限组的检索结果会被低权限组直接命中，造成越权读取；
- 入库 / 删除文档后按知识库前缀 SCAN 失效，避免旧片段命中；
- Redis 任何异常都降级为"缓存未命中 / 不写"，绝不阻断检索主链路（硬约束）。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import asdict

import redis

from app.core.logging import get_logger
from app.core.metrics import metrics
from app.domain.dto import RetrievedChunk
from app.domain.kb_config import RetrievalConfig

logger = get_logger(__name__)

KEY_PREFIX = "rag:q"
VERSION_KEY_PREFIX = "rag:qv"


class RetrievalCache:
    """Redis 检索缓存；ttl_s <= 0 时整体禁用（所有操作退化为空操作）。"""

    def __init__(
        self,
        redis_url: str,
        *,
        ttl_s: int = 300,
        client: redis.Redis | None = None,
    ) -> None:
        self._ttl_s = ttl_s
        # client 参数供测试注入 fakeredis；生产按 redis_url 懒建连
        self._client: redis.Redis | None = client
        if self._client is None and ttl_s > 0:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        if ttl_s <= 0:
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def build_key(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        question: str,
        top_k: int,
        use_rewrite: bool,
        use_rerank: bool,
        config: RetrievalConfig,
        allowed_groups: frozenset[str] | None = None,
        excluded_doc_ids: frozenset[str] | None = None,
        embedding_version: str = "",
        cache_version: int = 0,
    ) -> str:
        """检索配置与用户组都参与哈希：检索参数或 ACL 组变化后旧 key 自然失效。

        allowed_groups 三种语义必须显式区分（安全硬约束），以规范 JSON 编码：
        - None        → null   不限制（dev 环境、显式全组可访问）
        - frozenset() → []     显式空组（严格受限，只有公开文档可见）
        - 有组        → ["g1","g2"] 排序后的字符串数组
        JSON 的类型信息天然区分以上三态，也避免 {"finance","hr"} 与 {"finance,hr"}
        因逗号拼接而碰撞；组顺序不影响结果。

        excluded_doc_ids 同样进 key：文档被标记为"已被替代"后过滤集合变化，
        若沿用旧 key 会把旧版本的召回结果继续返回。

        embedding_version 进 key：换 embedding 模型并升版本后，旧向量语义下
        的召回结果不会被误命中（无需手动清缓存）。

        cache_version 进 key：入库/删除/配置变更会 bump 知识库版本号，即使
        invalidate_kb 的 SCAN+DELETE 失败，旧版本 key 也不会再被命中（双保险）。
        """
        config_hash = hashlib.sha256(
            json.dumps(config.model_dump(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        # ACL 必须用"带类型信息"的规范 JSON，不能用分隔符拼接：
        # 拼接会把 {"finance","hr"} 与 {"finance,hr"}、None 与 {"*"} 压成同一个
        # 字符串，不同权限用户因此命中彼此的缓存（越权读取）。
        # JSON 的 null / [] / ["g1","g2"] 天然区分"不限权 / 显式空组 / 具体组集合"。
        acl_key = None if allowed_groups is None else sorted(allowed_groups)
        payload = {
            "question": question.strip(),
            "top_k": top_k,
            "use_rewrite": use_rewrite,
            "use_rerank": use_rerank,
            "acl_groups": acl_key,
            "excluded_doc_ids": sorted(excluded_doc_ids) if excluded_doc_ids else [],
            "config_hash": config_hash,
            "embedding_version": embedding_version,
            "cache_version": cache_version,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return f"{KEY_PREFIX}:{tenant_id}:{kb_id}:{digest}"

    def get(self, key: str) -> list[RetrievedChunk] | None:
        """读取缓存：Redis 异常、内容损坏都按未命中处理；命中/未命中都计指标。"""
        if self._client is None:
            return None
        try:
            payload = self._client.get(key)
        except redis.RedisError as exc:
            logger.warning("检索缓存读取失败，按未命中处理：%s", exc)
            return None
        if payload is None:
            metrics.inc("retrieval_cache_misses_total")
            return None
        try:
            raw = json.loads(payload)
            chunks = [RetrievedChunk(**item) for item in raw]
        except (TypeError, ValueError, KeyError) as exc:
            # 缓存内容损坏（如版本升级后结构变化）：删掉脏 key，按未命中走
            logger.warning("检索缓存反序列化失败，按未命中处理：%s", exc)
            self._safe_delete(key)
            metrics.inc("retrieval_cache_misses_total")
            return None
        metrics.inc("retrieval_cache_hits_total")
        return chunks

    def set(self, key: str, chunks: list[RetrievedChunk]) -> None:
        """写入缓存（带 TTL）：序列化或网络失败只告警，不影响本次检索结果返回。"""
        if self._client is None:
            return
        try:
            payload = json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False)
            self._client.setex(key, self._ttl_s, payload)
        except (redis.RedisError, TypeError, ValueError) as exc:
            logger.warning("检索缓存写入失败（不影响检索结果）：%s", exc)

    def invalidate_kb(self, tenant_id: str, kb_id: str) -> int:
        """按知识库前缀批量失效；返回删除条数（失败返回 0）。"""
        if self._client is None:
            return 0
        pattern = f"{KEY_PREFIX}:{tenant_id}:{kb_id}:*"
        removed = 0
        try:
            # 用 scan_iter 而非 KEYS：KEYS 会一次性阻塞 Redis，SCAN 游标式分批遍历不影响线上
            batch: list[str] = []
            for key in self._client.scan_iter(match=pattern, count=200):
                batch.append(key)
                if len(batch) >= 200:
                    removed += self._client.delete(*batch)
                    batch = []
            if batch:
                removed += self._client.delete(*batch)
        except redis.RedisError as exc:
            logger.warning("检索缓存失效失败 kb=%s（TTL 兜底自动过期）：%s", kb_id, exc)
            return 0
        if removed:
            logger.info("检索缓存已失效 kb=%s 条数=%s", kb_id, removed)
        return removed

    def bump_version(self, tenant_id: str, kb_id: str) -> int:
        """版本号自增：即使 invalidate_kb 里的 SCAN+DELETE 失败，版本号变化也能让旧 key 自然失效。

        任何 Redis 异常静默吞掉并返回 0：退化为只依赖 SCAN+DELETE，不阻塞主链路。
        """
        if self._client is None:
            return 0
        key = f"{VERSION_KEY_PREFIX}:{tenant_id}:{kb_id}"
        try:
            return int(self._client.incr(key) or 1)
        except redis.RedisError as exc:
            logger.warning("缓存版本号 bump 失败（退化为 SCAN+DELETE）：%s", exc)
            return 0

    def get_version(self, tenant_id: str, kb_id: str) -> int:
        """读取当前版本号；Redis 不可用时退化返回 0。"""
        if self._client is None:
            return 0
        key = f"{VERSION_KEY_PREFIX}:{tenant_id}:{kb_id}"
        try:
            raw = self._client.get(key)
            return int(raw) if raw is not None else 0
        except redis.RedisError:
            return 0

    def _safe_delete(self, key: str) -> None:
        """尽力删除：Redis 异常静默吞掉（调用方本就按缓存未命中继续）。"""
        if self._client is None:
            return
        with contextlib.suppress(redis.RedisError):
            self._client.delete(key)

    def close(self) -> None:
        """关闭连接：进程退出时调用；异常静默，不阻塞退出流程。"""
        if self._client is not None:
            with contextlib.suppress(redis.RedisError):
                self._client.close()
