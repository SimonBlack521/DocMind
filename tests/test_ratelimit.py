"""Rate Limiter 三种实现的单元测试：Redis / 进程内 / 禁用。

Redis 实现用 fakeredis 模拟，无需真实 Redis 服务。
"""

from __future__ import annotations

import time

import fakeredis
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import rate_limit_dep
from app.auth.base import CurrentUser
from app.core.ratelimit import (
    DisabledRateLimiter,
    InMemoryRateLimiter,
    RedisRateLimiter,
    build_rate_limiter,
)

# ---------------------------------------------------------------- Disabled


def test_disabled_always_allows() -> None:
    limiter = DisabledRateLimiter()
    for _ in range(200):
        res = limiter.check("chat", "u1", limit=5)
        assert res.allowed
        assert res.remaining == -1
    limiter.close()


# ---------------------------------------------------------------- InMemory


def test_in_memory_allows_within_limit() -> None:
    limiter = InMemoryRateLimiter()
    for i in range(5):
        res = limiter.check("chat", "u1", limit=5)
        assert res.allowed, f"第 {i + 1} 次应通过"
        assert res.remaining == 4 - i


def test_in_memory_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter()
    for _ in range(5):
        assert limiter.check("chat", "u1", limit=5).allowed
    blocked = limiter.check("chat", "u1", limit=5)
    assert not blocked.allowed
    assert blocked.reset_after_s > 0
    assert blocked.reset_after_s <= 60


def test_in_memory_per_user_isolation() -> None:
    limiter = InMemoryRateLimiter()
    for _ in range(5):
        limiter.check("chat", "u1", limit=5)
    # u2 应该还能发
    assert limiter.check("chat", "u2", limit=5).allowed
    # u3 不同 action 也隔离
    assert limiter.check("upload", "u1", limit=5).allowed


def test_in_memory_window_expires() -> None:
    limiter = InMemoryRateLimiter()
    for _ in range(5):
        limiter.check("chat", "u1", limit=5, window_s=1)
    assert not limiter.check("chat", "u1", limit=5, window_s=1).allowed
    time.sleep(1.1)
    # 窗口过期后应该重新放行
    assert limiter.check("chat", "u1", limit=5, window_s=1).allowed


def test_in_memory_close_clears() -> None:
    limiter = InMemoryRateLimiter()
    for _ in range(5):
        limiter.check("chat", "u1", limit=5)
    limiter.close()
    # close 后窗口清空，应该又能发
    assert limiter.check("chat", "u1", limit=5).allowed


# ---------------------------------------------------------------- Redis (fakeredis)


def test_redis_allows_within_limit() -> None:
    r = fakeredis.FakeRedis(decode_responses=True)
    limiter = RedisRateLimiter(r)
    for i in range(5):
        res = limiter.check("chat", "u1", limit=5)
        assert res.allowed, f"第 {i + 1} 次应通过"
        assert res.remaining == 4 - i
    limiter.close()


def test_redis_blocks_after_limit() -> None:
    r = fakeredis.FakeRedis(decode_responses=True)
    limiter = RedisRateLimiter(r)
    for _ in range(5):
        assert limiter.check("chat", "u1", limit=5).allowed
    blocked = limiter.check("chat", "u1", limit=5)
    assert not blocked.allowed
    assert blocked.reset_after_s > 0
    limiter.close()


def test_redis_key_format() -> None:
    r = fakeredis.FakeRedis(decode_responses=True)
    limiter = RedisRateLimiter(r)
    limiter.check("chat", "user-42", limit=10)
    keys = list(r.keys("rag:rl:*"))
    assert keys == ["rag:rl:chat:user-42"]
    limiter.close()


def test_redis_per_user_isolation() -> None:
    r = fakeredis.FakeRedis(decode_responses=True)
    limiter = RedisRateLimiter(r)
    for _ in range(5):
        limiter.check("chat", "u1", limit=5)
    assert limiter.check("chat", "u2", limit=5).allowed
    assert limiter.check("upload", "u1", limit=5).allowed
    limiter.close()


# ---------------------------------------------------------------- build_rate_limiter


def test_build_disabled_when_flag_false() -> None:
    limiter = build_rate_limiter(enabled=False, redis_url="redis://localhost:6379/0")
    assert isinstance(limiter, DisabledRateLimiter)


def test_build_with_injected_redis_client() -> None:
    r = fakeredis.FakeRedis(decode_responses=True)
    limiter = build_rate_limiter(enabled=True, redis_url="", redis_client=r)
    assert isinstance(limiter, RedisRateLimiter)


def test_build_falls_back_to_in_memory_when_redis_down(monkeypatch) -> None:
    import redis as real_redis

    class _BadClient:
        def ping(self):
            raise real_redis.ConnectionError("nope")

    monkeypatch.setattr(
        real_redis.Redis, "from_url", staticmethod(lambda *a, **kw: _BadClient())
    )
    limiter = build_rate_limiter(enabled=True, redis_url="redis://nope:6379/0")
    assert isinstance(limiter, InMemoryRateLimiter)


# ---------------------------------------------------------------- FastAPI dependency 集成


def _fake_user_dep() -> CurrentUser:
    return CurrentUser(
        user_id="u1",
        tenant_id="default",
        allowed_groups=frozenset(),
        allowed_kb_ids=None,
    )


def test_dependency_returns_429_when_exceeded() -> None:
    """走 FastAPI TestClient：连续请求超上限 → 429 + Retry-After header。"""
    app = FastAPI()
    _chat_limit = rate_limit_dep("chat", limit=3)

    @app.get("/test", dependencies=[Depends(_fake_user_dep), _chat_limit])
    async def _endpoint():
        return {"ok": True}

    fake_r = fakeredis.FakeRedis(decode_responses=True)
    limiter = build_rate_limiter(enabled=True, redis_url="", redis_client=fake_r)
    app.state.rate_limiter = limiter

    client = TestClient(app)
    try:
        for i in range(3):
            resp = client.get("/test")
            assert resp.status_code == 200, f"第 {i + 1} 次应 200"
        resp = client.get("/test")
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") is not None
        assert resp.headers.get("X-RateLimit-Limit") == "3"
    finally:
        limiter.close()


def test_disabled_rate_limit_never_blocks() -> None:
    """rate_limit_enabled=false 时 100 次请求全过（DisabledRateLimiter）。"""
    app = FastAPI()
    _chat_limit = rate_limit_dep("chat", limit=3)

    @app.get("/test", dependencies=[Depends(_fake_user_dep), _chat_limit])
    async def _endpoint():
        return {"ok": True}

    # 不挂 limiter → get_rate_limiter 即时构建 DisabledRateLimiter
    client = TestClient(app)
    for _ in range(100):
        resp = client.get("/test")
        assert resp.status_code == 200
