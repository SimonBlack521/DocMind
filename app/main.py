"""FastAPI 应用装配：启动加载知识库注册表，统一异常输出，挂载业务路由。

本文件是整个 RAG 后端的"装配车间"，在请求链路中的位置：
    客户端 → uvicorn → create_app() 装配出的 FastAPI 实例 → 各 router（app/api/routers/）
启动期（lifespan）完成所有重量级单例的构建并挂到 app.state 上（知识库注册表、
运行时注册表、S3、PG engine/仓库、认证提供者、检索缓存、RAG 管线、限流器），
请求期各端点通过 app/api/deps.py 的依赖函数从 app.state 取用这些单例。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routers import audit, chat, connectors, documents, health, kbs, tasks
from app.api.routers import metrics as metrics_router
from app.auth import build_auth_provider
from app.core.errors import AppError
from app.core.logging import get_logger, setup_logging
from app.core.metrics import metrics
from app.core.ratelimit import build_rate_limiter
from app.core.settings import get_settings
from app.db.repo import MetadataRepo
from app.db.session import build_engine, build_session_factory
from app.domain.config_sync import ConfigSync
from app.domain.kb_config import KBRegistry
from app.orchestration.pipeline import RAGPipeline
from app.orchestration.registry import RuntimeRegistry
from app.retrieval.cache import RetrievalCache
from app.storage.object_store import S3ObjectStore

logger = get_logger(__name__)

API_PREFIX = "/api/v1"  # 业务 API 统一前缀；health/metrics 属系统端点，不带前缀


def create_app() -> FastAPI:
    """应用工厂：完成日志初始化、lifespan（单例构建/回收）、异常处理、中间件与路由挂载。

    之所以做成工厂函数而不是模块级直接 new FastAPI，是便于测试/工具按需创建实例；
    注意模块末尾仍会以 `app = create_app()` 产出 uvicorn 入口用的单例。
    """
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """启动时构建全部进程级单例并挂到 app.state；退出时按依赖顺序释放资源。"""
        # 知识库配置以 YAML 文件为准，启动即扫目录加载为内存注册表
        kb_registry = KBRegistry.from_directory(settings.kb_config_dir)
        runtime_registry = RuntimeRegistry(settings)
        object_store = S3ObjectStore(
            endpoint_url=settings.s3_endpoint_url,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            bucket=settings.s3_bucket,
            region=settings.s3_region,
            use_ssl=settings.s3_use_ssl,
        )
        # 元数据库 engine 懒建连，启动时不依赖 PG 可用；建表走 alembic upgrade head
        db_engine = build_engine(settings.database_url)
        app.state.settings = settings
        app.state.kb_registry = kb_registry
        app.state.runtime_registry = runtime_registry
        app.state.object_store = object_store
        app.state.db_engine = db_engine
        app.state.db_repo = MetadataRepo(build_session_factory(db_engine))
        app.state.auth_provider = build_auth_provider(settings)
        retrieval_cache = RetrievalCache(
            settings.redis_url, ttl_s=settings.retrieval_cache_ttl_s
        )
        app.state.retrieval_cache = retrieval_cache
        # 配置失效通知：Redis 只传版本号/消息，YAML 仍是配置权威
        app.state.config_sync = ConfigSync(
            settings.redis_url,
            enabled=settings.kb_config_sync_interval_s > 0,
        )
        app.state.config_sync_state = None
        app.state.pipeline = RAGPipeline(
            settings=settings,
            registry=runtime_registry,
            object_store=object_store,
            cache=retrieval_cache,
            # 注入元数据仓库：启用重复内容血缘归属（citation 回溯到实际授权来源）
            repo=app.state.db_repo,
        )
        app.state.rate_limiter = build_rate_limiter(
            enabled=settings.rate_limit_enabled,
            redis_url=settings.redis_url,
        )
        logger.info("RAG 服务启动，已加载知识库 %s 个", len(kb_registry))
        try:
            yield
        finally:
            # 关停顺序：先停业务侧客户端（运行时/S3/缓存/限流），最后释放 PG 连接池
            runtime_registry.close()
            object_store.close()
            retrieval_cache.close()
            app.state.config_sync.close()
            app.state.rate_limiter.close()
            await db_engine.dispose()
            logger.info("RAG 服务已停止")

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    async def handle_app_error(request: Request, exc: Exception) -> JSONResponse:
        """统一异常出口：业务异常 AppError 按自带 status_code/payload 返回；
        未预期异常兜底为 500，避免把内部堆栈细节泄漏给调用方。"""
        if isinstance(exc, AppError):
            return JSONResponse(status_code=exc.status_code, content=exc.to_payload())
        logger.error("未预期异常：%s", exc, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "服务内部错误"},
        )

    # 只注册 AppError 类型；AppError 的子类（KBNotFoundError 等）都会命中此 handler
    app.add_exception_handler(AppError, handle_app_error)

    @app.middleware("http")
    async def metrics_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """HTTP 指标中间件：统计每个路由模板的请求量与耗时，供 /metrics 端点导出。"""
        # 统计每个路由模板的请求量与耗时；未匹配的路由统一记为 unmatched，避免标签爆炸
        started = time.perf_counter()
        response: Response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        metrics.inc(
            "http_requests_total",
            method=request.method,
            route=path,
            status=str(response.status_code),
        )
        metrics.observe("http_request_duration_seconds", time.perf_counter() - started, route=path)
        return response

    # CORS 仅在显式配置白名单时启用；未配置则完全不加中间件，默认拒绝跨域
    origins = [item.strip() for item in settings.cors_allow_origins.split(",") if item.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 挂载顺序即注册顺序：系统端点（health/metrics）无前缀，业务端点统一 /api/v1
    app.include_router(health.router)
    app.include_router(metrics_router.router)
    app.include_router(kbs.router, prefix=API_PREFIX)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(chat.router, prefix=API_PREFIX)
    app.include_router(tasks.router, prefix=API_PREFIX)
    app.include_router(audit.router, prefix=API_PREFIX)
    app.include_router(connectors.router, prefix=API_PREFIX)
    # ---- Web 前端（静态 SPA，最后挂载兜底 "/"，不影响已注册的 API 路由）----
    # app/web 目录由前端三件套（index.html / app.js / style.css）组成；
    # html=True 使 "/" 直接返回 index.html。目录缺失（如裁剪部署）时跳过挂载。
    web_dir = Path(__file__).resolve().parent / "web"
    if web_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=web_dir, html=True), name="web"
        )

    return app


app = create_app()  # uvicorn 入口单例（uvicorn app.main:app）
