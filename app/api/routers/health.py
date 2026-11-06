"""系统健康检查：进程 liveness + 基础设施 readiness。

- GET /health        —— 完整快照（含各依赖状态）；给人看或排障
- GET /healthz       —— 进程存活（200 即活着，不探依赖）；K8s liveness probe
- GET /readyz        —— 依赖就绪（任一必需依赖 down → 503）；K8s readiness probe

基础设施可选依赖（失败不拖垮主链路）：
- Milvus / Elasticsearch / PostgreSQL —— 必需
- Redis —— 检索缓存关闭时 skipped
- S3/MinIO —— 异步入库开启时必需，同步入库只用磁盘缓存
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.api.schemas import HealthResponse
from app.core.logging import get_logger
from app.domain.kb_config import KBConfig
from app.storage.object_store import ObjectStore

logger = get_logger(__name__)

router = APIRouter(tags=["system"])

_UP = "up"
_DOWN = "down"
_SKIPPED = "skipped"
_UNKNOWN = "unknown"


async def _probe_postgres(engine: Any) -> str:
    """探活 PostgreSQL：执行 SELECT 1，失败仅记 warning 并返回 down（不抛出拖垮健康检查）。"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return _UP
    except Exception as exc:
        logger.warning("健康检查 PostgreSQL 失败：%s", exc)
        return _DOWN


def _probe_redis(cache: Any) -> str:
    """探活 Redis：缓存未启用/未建连时返回 skipped（不参与总体状态计算）。"""
    client = getattr(cache, "_client", None)
    if client is None or not getattr(cache, "enabled", True):
        return _SKIPPED
    try:
        client.ping()
        return _UP
    except Exception as exc:
        logger.warning("健康检查 Redis 失败：%s", exc)
        return _DOWN


def _probe_milvus(kb: KBConfig, runtime_registry: Any) -> str:
    """探活 Milvus：优先用客户端 ping()，旧版客户端退化为 has_collection 判断。"""
    try:
        runtime = runtime_registry.runtime(kb)
        runtime.vector_store._get_client()  # 触发懒建连并校验可用
        # 真正发一个轻量请求确认网络 OK
        client = runtime.vector_store._get_client()
        # MilvusClient 在 2.4+ 有 ping()，返回 bool
        if hasattr(client, "ping"):
            return _UP if client.ping() else _DOWN
        # 退而求其次：确认 collection 存在即认为 OK（建集合时已验证过连接）
        return _UP if client.has_collection(runtime.vector_store._collection) else _DOWN
    except Exception as exc:
        logger.warning("健康检查 Milvus 失败（kb=%s）：%s", kb.kb_code, exc)
        return _DOWN


def _probe_elasticsearch(kb: KBConfig, runtime_registry: Any) -> str:
    """探活 Elasticsearch：官方客户端 ping() 返回 bool。"""
    try:
        runtime = runtime_registry.runtime(kb)
        client = runtime.keyword_store._client
        return _UP if client.ping() else _DOWN
    except Exception as exc:
        logger.warning("健康检查 ES 失败（kb=%s）：%s", kb.kb_code, exc)
        return _DOWN


def _probe_s3(object_store: ObjectStore | None) -> str:
    """探活 S3/MinIO：用 head_bucket/create_bucket 这类最便宜的元数据请求。"""
    if object_store is None:
        return _SKIPPED
    try:
        # S3ObjectStore._ensure_bucket 会 head_bucket 或 create_bucket，是最便宜的探活
        object_store._ensure_bucket()  # type: ignore[attr-defined]
        return _UP
    except Exception as exc:
        logger.warning("健康检查 S3/MinIO 失败：%s", exc)
        return _DOWN


async def _build_response(request: Request) -> HealthResponse:
    """汇总各依赖探活结果并计算整体状态：
    全部 up（或无可探依赖）→ ok；全部 down → down；其余 → degraded。
    Milvus/ES 按第一个启用的知识库探测（运行时客户端按库缓存，探一个即可代表连通性）。
    """
    settings = request.app.state.settings
    registry = request.app.state.kb_registry
    runtime_registry = request.app.state.runtime_registry
    db_engine = request.app.state.db_engine
    cache = request.app.state.retrieval_cache
    object_store = getattr(request.app.state, "object_store", None)

    services: dict[str, str] = {
        "postgres": await _probe_postgres(db_engine),
        "redis": _probe_redis(cache),
        "s3": _probe_s3(object_store),
    }

    first_kb = next(iter(registry.list_enabled()), None)
    if first_kb is not None:
        services["milvus"] = _probe_milvus(first_kb, runtime_registry)
        services["elasticsearch"] = _probe_elasticsearch(first_kb, runtime_registry)
    else:
        services["milvus"] = _SKIPPED
        services["elasticsearch"] = _SKIPPED

    up = sum(1 for v in services.values() if v == _UP)
    total = sum(1 for v in services.values() if v != _SKIPPED)
    if total == 0 or up == total:
        status = "ok"
    elif up == 0:
        status = "down"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        app_env=settings.app_env,
        auth_mode=settings.auth_mode,
        kb_count=len(registry),
        services=services,
    )


@router.get("/health", summary="完整健康检查（含依赖细粒度状态）")
async def health(request: Request) -> HealthResponse:
    return await _build_response(request)


@router.get("/healthz", summary="K8s liveness probe（只探进程）")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="K8s readiness probe（探所有依赖）")
async def readyz(request: Request) -> dict[str, Any]:
    response = await _build_response(request)
    # liveness 返回的 service 字段供上层观测；K8s 只看 HTTP status
    if response.status == "down":
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="依赖不可用")
    return response.model_dump()
