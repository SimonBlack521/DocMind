"""Celery 应用：Redis 同时作为 broker 与结果后端。

task_serializer=json：消息里只放业务字段与对象存储 key（原始文件先落 S3），
避免 pickle 反序列化带来的任意代码执行面，也便于跨语言消费者接入。
CELERY_EAGER=true 时任务在调用方进程内同步执行，仅限测试，禁止用于生产。
"""

from __future__ import annotations

from celery import Celery

from app.core.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.tasks.ingest",
        "app.tasks.maintenance",
        "app.tasks.outbox",
        "app.tasks.connectors",
    ],
)
celery_app.conf.update(
    # eager 模式：任务在调用方进程内同步执行、异常直接上抛（eager_propagates）。
    # 测试用它摆脱对真实 Redis/worker 的依赖（CELERY_EAGER=true），
    # 断言时无需等待异步完成；生产必须为 false，否则 API 进程会被入库任务阻塞。
    task_always_eager=settings.celery_eager,
    task_eager_propagates=True,
    task_track_started=True,
    # 任务时限：软时限先到（抛 SoftTimeLimitExceeded，任务可捕获做清理），
    # 硬时限兜底强杀，防止单个任务卡死 worker 进程
    task_soft_time_limit=settings.task_soft_time_limit_seconds,
    task_time_limit=settings.task_time_limit_seconds,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=3600,
    broker_connection_retry_on_startup=True,
)

# Beat 周期调度：只在配置非 0 时才注册，避免开发环境空转。
# 运维需要单独起 celery beat 进程：celery -A app.tasks.celery_app beat
beat_schedule: dict[str, dict] = {}
if settings.outbox_dispatch_interval_seconds > 0:
    beat_schedule["dispatch_outbox"] = {
        "task": "rag.dispatch_outbox",
        "schedule": float(settings.outbox_dispatch_interval_seconds),
        "kwargs": {"limit": 100},
    }
if settings.reconcile_interval_minutes > 0:
    beat_schedule["reconcile_index"] = {
        "task": "rag.reconcile_index",
        "schedule": float(settings.reconcile_interval_minutes) * 60.0,
        "kwargs": {"repair": True},
    }
if settings.deletion_retry_interval_seconds > 0:
    beat_schedule["resume_pending_deletions"] = {
        "task": "rag.resume_pending_deletions",
        "schedule": float(settings.deletion_retry_interval_seconds),
        "kwargs": {"limit": 100},
    }
if settings.document_retention_days > 0:
    beat_schedule["purge_expired_documents"] = {
        "task": "rag.purge_expired_documents",
        "schedule": 3600.0 * 24.0,  # 每天跑一次，分批删除
        "kwargs": {
            "retention_days": settings.document_retention_days,
            "limit": settings.retention_purge_batch,
        },
    }
if settings.superseded_retention_days > 0:
    # Active Version 延迟清理：被替代旧版超过观察窗后统一删除
    beat_schedule["purge_superseded_documents"] = {
        "task": "rag.purge_superseded_documents",
        "schedule": 3600.0 * 24.0,
        "kwargs": {"retention_days": settings.superseded_retention_days, "limit": 200},
    }
if settings.conversation_retention_days > 0:
    # 会话过期清理：长期不活跃的会话（含问答内容）到期删除
    beat_schedule["purge_expired_conversations"] = {
        "task": "rag.purge_expired_conversations",
        "schedule": 3600.0 * 24.0,
        "kwargs": {"retention_days": settings.conversation_retention_days, "limit": 500},
    }
if settings.connector_sync_scan_seconds > 0:
    # 连接器定时同步：按各连接器的 sync_interval_minutes 到期派发全量幂等同步
    beat_schedule["sync_due_connectors"] = {
        "task": "rag.sync_due_connectors",
        "schedule": float(settings.connector_sync_scan_seconds),
    }
if beat_schedule:
    celery_app.conf.beat_schedule = beat_schedule
