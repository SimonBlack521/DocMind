"""配置失效通知测试（v1.4）：版本号发布、读取与 Redis 故障降级。"""

from __future__ import annotations

import fakeredis
import redis

from app.domain.config_sync import CONFIG_INVALIDATION_CHANNEL, ConfigSync


def test_publish_increments_version_and_messages_channel() -> None:
    client = fakeredis.FakeRedis(decode_responses=True)
    sync = ConfigSync("redis://unused", client=client)

    assert sync.version() is None
    sync.publish("finance")
    assert sync.version() == 1

    pubsub = client.pubsub()
    pubsub.subscribe(CONFIG_INVALIDATION_CHANNEL)
    sync.publish("finance")  # 重新发布一次，验证消息内容
    message = None
    for _ in range(10):
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.05)
        if message is not None:
            break
    assert message is not None
    assert message["data"] == "finance"


def test_disabled_sync_is_noop() -> None:
    sync = ConfigSync("redis://unused", enabled=False)

    assert sync.enabled is False
    sync.publish("finance")
    assert sync.version() is None


class _BrokenRedis:
    """模拟 Redis 不可用：所有读写都抛连接错误。"""

    def incr(self, key: str) -> int:
        raise redis.RedisError("down")

    def publish(self, channel: str, message: str) -> int:
        raise redis.RedisError("down")

    def get(self, key: str) -> None:
        raise redis.RedisError("down")

    def close(self) -> None:
        pass


def test_redis_failure_degrades_without_raising() -> None:
    sync = ConfigSync("redis://unused", client=_BrokenRedis())  # type: ignore[arg-type]

    sync.publish("finance")  # 不抛异常
    assert sync.version() is None
    sync.close()
