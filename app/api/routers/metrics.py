"""进程内运行指标查询：JSON 快照 + Prometheus 文本格式。

指标数据由 main.py 的 metrics_middleware 埋点写入进程内注册表（app/core/metrics.py），
本路由只负责导出：/metrics 给人工查看，/metrics/prometheus 给 Prometheus 抓取。
注意：指标是进程内的，多副本部署时每个副本各自计数，聚合由 Prometheus 完成。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.api.deps import SettingsDep
from app.core.metrics import metrics

router = APIRouter(tags=["system"])


@router.get("/metrics", summary="进程内运行指标快照（JSON）")
def get_metrics() -> dict[str, Any]:
    return metrics.snapshot()


@router.get(
    "/metrics/prometheus",
    response_class=PlainTextResponse,
    summary="进程内运行指标（Prometheus exposition format）",
)
def get_metrics_prometheus() -> PlainTextResponse:
    return PlainTextResponse(
        content=metrics.prometheus_exposition(), media_type="text/plain; version=0.0.4"
    )


@router.get(
    "/metrics/queue-depth",
    summary="Celery 队列深度（broker 上待消费的消息数，供堆积告警）",
)
def get_queue_depth(settings: SettingsDep) -> dict[str, Any]:
    """读取 Redis broker 上 celery 默认队列的 LLEN。

    broker 不可用时返回 error 字段而不是 500：队列深度是观测面，
    不能因为 Redis 抖动让监控系统把 API 实例误判为宕机。
    """
    try:
        import redis as redis_lib

        client = redis_lib.Redis.from_url(settings.redis_url, socket_timeout=2)
        try:
            depth = int(client.llen("celery"))
        finally:
            client.close()
        return {"queue": "celery", "depth": depth, "error": None}
    except Exception as exc:
        return {"queue": "celery", "depth": None, "error": str(exc)}
