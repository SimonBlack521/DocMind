"""异步数据库会话：engine 随应用/worker 生命周期创建与关闭。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def build_engine(database_url: str) -> AsyncEngine:
    """创建异步 engine；pool_pre_ping 保证拿到的连接是活的（数据库重启后自动重连）。"""
    return create_async_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False：commit 后对象属性仍可读取，避免懒加载触发异步异常
    return async_sessionmaker(engine, expire_on_commit=False)
