# ============================================================
# 异步任务执行器
#
# 一份实现，两种投递：
#   - TASK_MODE=inline：BackgroundTasks 进程内执行（本地/测试/无 Redis）
#   - TASK_MODE=celery：投递到 broker，由 worker 执行（生产 compose）
#
# 两条路径最终都调用 run_task()，确保"本地能跑通"与"线上跑得动"
# 不是两套逻辑。任务状态一律写 kb_task（跨进程可见），进程内不留状态。
#
# 之所以在函数内部 import src.api.documents：documents 在模块层 import
# 了 submit()，模块级互相 import 会成环；执行期再取即可。
# ============================================================

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from loguru import logger

from config.settings import settings
from src.storage import repository

# 任务名 → Celery 注册名（worker 与 API 必须一致）
TASK_NAMES = {
    "upload": "kb.upload",
    "rebuild_doc": "kb.rebuild_doc",
    "rebuild_collection": "kb.rebuild_collection",
}


def submit(background, kind: str, task_id: str, payload: Dict) -> str:
    """投递任务，返回执行器标识（inline / celery）。"""
    if kind not in TASK_NAMES:
        raise ValueError(f"未知任务类型: {kind!r}")

    if (settings.task_mode or "inline").lower() == "celery":
        from src.tasks.celery_app import celery_app

        celery_app.send_task(TASK_NAMES[kind], args=[task_id, payload])
        logger.info(f"[任务] 已投递 Celery: {TASK_NAMES[kind]} task={task_id}")
        return "celery"

    background.add_task(run_task, kind, task_id, payload)
    return "inline"


def _resolve_paths(kind: str, payload: Dict) -> List[Path]:
    from src.api.documents import _iter_collection_files, _locate_document

    collection = payload.get("collection", "local_kb")

    if kind == "upload":
        return [Path(p) for p in payload.get("files", [])]

    if kind == "rebuild_doc":
        path = _locate_document(
            collection,
            payload.get("doc_id", ""),
            payload.get("department", "通用"),
        )
        if path is None:
            raise FileNotFoundError(
                f"文档不存在: {collection}/{payload.get('department')}/"
                f"{payload.get('doc_id')}"
            )
        return [path]

    if kind == "rebuild_collection":
        return _iter_collection_files(collection)

    raise ValueError(f"未知任务类型: {kind!r}")


def run_task(kind: str, task_id: str, payload: Dict) -> None:
    """任务体：逐文件入库（替换语义），并把进度回写 kb_task。"""
    from src.api.documents import _ingest_one

    collection = payload.get("collection", "local_kb")
    chunks_total = 0
    failed = 0
    errors: List[str] = []

    try:
        paths = _resolve_paths(kind, payload)
        repository.update_task(
            task_id, status="processing", files=len(paths)
        )
        logger.info(f"[任务] 开始 {kind} task={task_id} files={len(paths)}")

        for path in paths:
            try:
                chunks = _ingest_one(path, collection)
                chunks_total += chunks
                repository.upsert_document(
                    collection=collection,
                    doc_id=path.stem,
                    department=path.parent.name,
                    path=str(path),
                    size_bytes=path.stat().st_size,
                    chunk_count=chunks,
                )
            except Exception as e:  # 单文件失败不影响同批其他文件
                failed += 1
                errors.append(f"{path.name}: {e}")
                logger.error(f"入库失败 {path.name}: {e}")

        repository.update_task(
            task_id,
            status="done",
            chunks=chunks_total,
            failed=failed,
            errors=errors,
            finished_at=datetime.now(),
        )
        logger.info(
            f"[任务] 完成 {kind} task={task_id} "
            f"chunks={chunks_total} failed={failed}"
        )
    except Exception as e:  # 任务级失败（目录不存在 / 文件定位失败等）
        logger.error(f"[任务] 异常 {kind} task={task_id}: {e}")
        repository.update_task(
            task_id,
            status="error",
            chunks=chunks_total,
            failed=failed,
            errors=[*errors, str(e)],
            finished_at=datetime.now(),
        )
