# ============================================================
# Celery 任务定义（薄壳：只做参数搬运，逻辑在 runner.run_task）
#
# 任务名与 runner.TASK_NAMES 一一对应，改名必须两边同时改。
# ============================================================

from src.tasks.celery_app import celery_app
from src.tasks.runner import run_task


@celery_app.task(name="kb.upload")
def upload_task(task_id: str, payload: dict) -> None:
    run_task("upload", task_id, payload)


@celery_app.task(name="kb.rebuild_doc")
def rebuild_doc_task(task_id: str, payload: dict) -> None:
    run_task("rebuild_doc", task_id, payload)


@celery_app.task(name="kb.rebuild_collection")
def rebuild_collection_task(task_id: str, payload: dict) -> None:
    run_task("rebuild_collection", task_id, payload)
