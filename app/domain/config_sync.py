"""知识库配置失效通知（v1.4）：多副本部署下让 YAML 配置变更快速生效。

背景：知识库配置以 YAML 文件为准，各副本启动时各自扫目录进内存。单副本写入
新配置后，其他副本只有"请求的 kb 解析不到"时才会重扫目录；已有库的配置修改
（如检索参数、启用状态）在旧副本上可能长期不可见。

方案（保持文件为权威，Redis 只做通知，不存配置本体）：
    - 写操作成功后 INCR 一个全局版本号并 PUBLISH 一个失效消息；
    - 每个副本每 N 秒最多 GET 一次版本号，发现变化就重扫目录；
    - Redis 不可用时静默降级为"只在解析不到 kb 时重扫"，不阻断任何请求。
"""

from __future__ import annotations

import redis

from app.core.logging import get_logger

logger = get_logger(__name__)

CONFIG_VERSION_KEY = "rag:kb:config:version"
CONFIG_INVALIDATION_CHANNEL = "rag:kb:config:invalidate"


class ConfigSync:
    """配置版本号与失效消息的发布/读取；所有 Redis 异常都降级为无操作。"""

    def __init__(
        self,
        redis_url: str,
        *,
        enabled: bool = True,
        client: redis.Redis | None = None,
    ) -> None:
        # client 参数供测试注入 fakeredis；生产按 redis_url 懒建连
        self._client: redis.Redis | None = client
        if self._client is None and enabled and redis_url.strip():
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def publish(self, kb_code: str) -> None:
        """写入新版本号并广播失效消息；失败只告警，不影响已完成的配置写入。"""
        if self._client is None:
            return
        try:
            self._client.incr(CONFIG_VERSION_KEY)
            self._client.publish(CONFIG_INVALIDATION_CHANNEL, kb_code)
        except redis.RedisError as exc:
            logger.warning("配置失效通知发送失败 kb=%s：%s", kb_code, exc)

    def version(self) -> int | None:
        """读取当前配置版本号；读取失败或无版本时返回 None（调用方保持现状）。"""
        if self._client is None:
            return None
        try:
            value = self._client.get(CONFIG_VERSION_KEY)
        except redis.RedisError as exc:
            logger.warning("配置版本号读取失败：%s", exc)
            return None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning("配置版本号格式异常：%r", value)
            return None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
