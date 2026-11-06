"""固定窗口频率限制器：Redis 共享状态（多副本部署），降级到进程内 dict。

固定窗口算法原理：
    以 (action, user_id) 为粒度维护一个带 TTL 的计数器。窗口内每次请求计数 +1，
    计数超过 limit 即拒绝，TTL 到期后计数器自然消失、下一请求开启新窗口。
    优点是实现极简、每次请求 O(1)；已知边界问题是窗口切换瞬间允许约 2×limit
    的突发（窗口尾部打满 + 新窗口开头再打满）。对分钟级的 chat/upload 配额
    这种突发无害，因此没上滑动窗口/令牌桶（实现与运维成本高一个量级）。

Redis 不可用的降级权衡：
    限流是"保护性"功能而非"正确性"功能——Redis 抖动时如果直接 503，等于
    把缓存层的故障放大成全站不可用，违背限流的初衷。因此选择两级降级：
    1) 构建期连不上 Redis → 整体换成 InMemoryRateLimiter（各副本独立计数，
       总上限变为 limit×副本数，限流变松但服务可用；Redis 恢复后重建即回归精准）；
    2) 运行期单次 INCR 失败 → 该次请求放行并告警（fail-open）。
    如果业务对配额有强一致要求（如计费），这套 fail-open 策略需要反过来设计。

调用模式：
    limiter = build_rate_limiter(settings)
    result = limiter.check("chat", user_id, limit=30)  # 每分钟 30 次
    if not result.allowed:
        raise HTTPException(
            429, detail="rate_limit_exceeded",
            headers={"Retry-After": str(result.reset_after_s)},
        )
"""

from __future__ import annotations

import contextlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import redis

from app.core.logging import get_logger

logger = get_logger(__name__)

_KEY_PREFIX = "rag:rl"


@dataclass
class RateResult:
    allowed: bool
    """请求是否通过。"""
    remaining: int
    """当前窗口剩余可发次数（-1 表示未被限流）。"""
    reset_after_s: float
    """距离窗口重置还剩多少秒（仅被拒绝时有意义）。"""


class RateLimiter(ABC):
    """频率限制器抽象基类；Redis 实现与进程内实现共享接口。"""

    @abstractmethod
    def check(
        self, action: str, user_id: str, *, limit: int, window_s: int = 60
    ) -> RateResult:
        """检查一次请求是否通过。

        :param action: 业务标识，例如 "chat" / "upload"
        :param user_id: 限粒度，按用户隔离
        :param limit: 窗口内最多允许次数
        :param window_s: 窗口长度（秒），默认 60 秒
        """

    @abstractmethod
    def close(self) -> None:
        """释放资源（Redis 连接池等）。"""


class RedisRateLimiter(RateLimiter):
    """基于 Redis 固定窗口：INCR 原子 + 首次 EXPIRE。

    固定窗口在窗口切换时允许 2×limit 的瞬时突发，对 Chat/Upload 这种分钟级配额完全够用，
    比 sliding window / token bucket 实现成本低一个数量级。
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._r = redis_client

    def check(
        self, action: str, user_id: str, *, limit: int, window_s: int = 60
    ) -> RateResult:
        """INCR 原子计数；新窗口首个请求补 EXPIRE 作为窗口 TTL。

        竞态说明：INCR 与 EXPIRE 不是一次 Lua 事务，理论上"INCR 后 EXPIRE 前
        进程崩溃"会留下无 TTL 的 key。但 key 内容只是计数，影响限于该用户
        被永久按旧计数限流——实践中可接受；且只有 count==1 这一个请求会补
        EXPIRE，代价远低于每次请求都跑 Lua。
        """
        key = f"{_KEY_PREFIX}:{action}:{user_id}"
        try:
            count = self._r.incr(key)
        except redis.RedisError as exc:
            # Redis 挂了 → 直接放行并报警
            logger.warning("Redis Rate Limiter 异常，降级放行：%s", exc)
            return RateResult(allowed=True, remaining=-1, reset_after_s=0.0)

        # 第一次 INCR 时设置窗口 TTL
        if count == 1:
            self._r.expire(key, window_s)
        elif count > limit:
            ttl = self._r.ttl(key)
            return RateResult(
                allowed=False, remaining=0, reset_after_s=max(1.0, float(ttl))
            )

        return RateResult(
            allowed=True, remaining=limit - int(count), reset_after_s=0.0
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._r.close()


@dataclass
class _Window:
    count: int
    expires_at: float


class InMemoryRateLimiter(RateLimiter):
    """进程内固定窗口 dict；Redis 不可用时的兜底。

    多副本部署下各实例独立计数，总速率 = limit × 实例数；这在 Redis 恢复后自动回归精准限流。
    """

    def __init__(self) -> None:
        self._windows: dict[tuple[str, str], _Window] = {}
        self._lock = threading.Lock()

    def check(
        self, action: str, user_id: str, *, limit: int, window_s: int = 60
    ) -> RateResult:
        """单机版固定窗口；用 monotonic 时钟避免系统时间回拨导致窗口异常延长。"""
        now = time.monotonic()
        key = (action, user_id)

        with self._lock:
            # 定期清理过期窗口，避免长期运行 dict 无限增长
            if len(self._windows) > 4096:
                stale = [k for k, w in self._windows.items() if w.expires_at <= now]
                for k in stale:
                    del self._windows[k]

            win = self._windows.get(key)
            if win is None or win.expires_at <= now:
                win = _Window(count=1, expires_at=now + window_s)
                self._windows[key] = win
                return RateResult(allowed=True, remaining=limit - 1, reset_after_s=0.0)

            win.count += 1
            if win.count > limit:
                return RateResult(
                    allowed=False,
                    remaining=0,
                    reset_after_s=max(0.1, win.expires_at - now),
                )
            return RateResult(
                allowed=True, remaining=limit - win.count, reset_after_s=0.0
            )

    def close(self) -> None:
        with self._lock:
            self._windows.clear()


class DisabledRateLimiter(RateLimiter):
    """rate_limit_enabled=false 时使用，check() 永远放行。"""

    def check(
        self, action: str, user_id: str, *, limit: int, window_s: int = 60
    ) -> RateResult:
        return RateResult(allowed=True, remaining=-1, reset_after_s=0.0)

    def close(self) -> None:
        pass


def build_rate_limiter(
    *,
    enabled: bool,
    redis_url: str,
    redis_client: redis.Redis | None = None,
) -> RateLimiter:
    """按配置构造限流器；enabled=false 返回 DisabledRateLimiter。

    :param enabled: 总开关（对应 Settings.rate_limit_enabled）；
    :param redis_url: Redis 连接串，仅在未注入 redis_client 时使用；
    :param redis_client: 测试可注入 fakeredis 实例，跳过真实网络连接。

    构造期会先 ping 一次探活：连不上则告警并返回进程内实现（见模块 docstring
    的降级权衡）。注意 InMemoryRateLimiter 是构建期的一次性决策，运行期 Redis
    再恢复不会自动切回；如需切回，重启进程或重新调用本函数替换引用。
    """
    if not enabled:
        return DisabledRateLimiter()

    client = redis_client
    if client is None:
        try:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
        except (redis.RedisError, Exception) as exc:
            logger.warning("Redis 不可用（%s），Rate Limiter 降级为进程内实现", exc)
            return InMemoryRateLimiter()

    return RedisRateLimiter(client)
