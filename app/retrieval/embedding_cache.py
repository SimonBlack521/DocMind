"""Query Embedding 缓存（对应 9.7 的第一层缓存）。

同一个问题在 TTL 内重复出现时，不必再调一次 Embedding 模型：
- 进程内 LRU 承接高频重复问题，零网络开销；
- Redis 承接多进程 / 多实例共享，重启后仍可命中；
- key 必须绑定 租户 + 知识库 + 模型 + embedding_version + 文本：
  换模型/升版本后旧向量语义不可复用，否则会拿旧语义向量去检索新空间；
- 任何 Redis 异常都降级为"本地 LRU 未命中"，绝不阻断检索主链路。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from collections import OrderedDict

import redis

from app.core.logging import get_logger

logger = get_logger(__name__)

KEY_PREFIX = "rag:e"


class QueryEmbeddingCache:
    """Redis + 进程内 LRU 两级缓存；ttl_s<=0 时整体禁用。"""

    def __init__(
        self,
        redis_url: str = "",
        *,
        ttl_s: int = 3600,
        max_entries: int = 2048,
        client: redis.Redis | None = None,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max(1, max_entries)
        self._local: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._client: redis.Redis | None = None
        if ttl_s > 0:
            self._client = client
            if self._client is None and redis_url:
                self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    @property
    def enabled(self) -> bool:
        return self._ttl_s > 0

    @staticmethod
    def build_key(
        *,
        tenant_id: str,
        kb_id: str,
        model: str,
        embedding_version: str,
        text: str,
    ) -> str:
        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        return f"{KEY_PREFIX}:{tenant_id}:{kb_id}:{model}:{embedding_version}:{digest}"

    def get(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        model: str,
        embedding_version: str,
        text: str,
    ) -> list[float] | None:
        """先查进程内 LRU，再查 Redis；命中后回填 LRU。"""
        if not self.enabled:
            return None
        key = self.build_key(
            tenant_id=tenant_id,
            kb_id=kb_id,
            model=model,
            embedding_version=embedding_version,
            text=text,
        )
        with self._lock:
            local = self._local.get(key)
            if local is not None:
                self._local.move_to_end(key)
                return list(local)
        if self._client is None:
            return None
        try:
            payload = self._client.get(key)
        except redis.RedisError as exc:
            logger.warning("Query Embedding 缓存读取失败，按未命中处理：%s", exc)
            return None
        if payload is None:
            return None
        try:
            vector = [float(item) for item in json.loads(payload)]
        except (TypeError, ValueError):
            logger.warning("Query Embedding 缓存内容损坏，按未命中处理 key=%s", key)
            return None
        self._set_local(key, vector)
        return vector

    def set(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        model: str,
        embedding_version: str,
        text: str,
        vector: list[float],
    ) -> None:
        """写两级缓存；失败只告警，不影响本次检索。"""
        if not self.enabled:
            return
        key = self.build_key(
            tenant_id=tenant_id,
            kb_id=kb_id,
            model=model,
            embedding_version=embedding_version,
            text=text,
        )
        self._set_local(key, vector)
        if self._client is None:
            return
        try:
            self._client.setex(key, self._ttl_s, json.dumps(vector))
        except (redis.RedisError, TypeError, ValueError) as exc:
            logger.warning("Query Embedding 缓存写入失败（不影响检索）：%s", exc)

    def _set_local(self, key: str, vector: list[float]) -> None:
        with self._lock:
            self._local[key] = list(vector)
            self._local.move_to_end(key)
            while len(self._local) > self._max_entries:
                self._local.popitem(last=False)

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(redis.RedisError):
                self._client.close()
