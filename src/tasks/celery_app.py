# ============================================================
# Celery 应用（生产模式下由独立 worker 消费）
#
# 启动：
#   celery -A src.tasks.celery_app.celery_app worker -l info -Q celery
#
# broker/backend 默认复用 Redis（REDIS_URL），可分别用
# CELERY_BROKER_URL / CELERY_RESULT_BACKEND 覆盖。
# ============================================================

from celery import Celery

from config.settings import settings

_broker = settings.celery_broker_url or settings.redis_url
_backend = settings.celery_result_backend or settings.redis_url

celery_app = Celery(
    "medical_rag",
    broker=_broker,
    backend=_backend,
    include=["src.tasks.definitions"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_track_started=True,
    # broker 重启后 worker 自动重连，不在启动期直接崩掉
    broker_connection_retry_on_startup=True,
    # 灌库是重任务：晚确认 + worker 丢失重投，避免半途丢任务
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # 每个子进程处理 50 个任务后回收，防止长期持有模型/连接句柄
    worker_max_tasks_per_child=50,
    result_expires=3600,
)
