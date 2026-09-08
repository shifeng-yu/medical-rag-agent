# ============================================================
# 知识库文档管理路由（上传 → 分块 → 向量化 → 入库 / 删除 / 替换）
#
# 使用方式（网页 / curl 皆可）：
#   POST   /api/v1/documents/upload   multipart 上传 .md/.txt/.json
#       字段: files(可多个) + department(科室标签) + collection(仅 local_kb)
#       返回 task_id；分块与向量化在后台任务执行
#   GET    /api/v1/documents/tasks/{task_id}   轮询灌库状态
#   GET    /api/v1/documents                   列出知识库目录文件与近期任务
#   DELETE /api/v1/documents/{doc_id}          删除文档（文件 + 向量一起清）
#
# 设计约束（见 docs/adr/0005-upload-delete-api.md）：
#   - 网页上传只开放 local_kb 集合；pubmed 文献库只经 ingest_pubmed.py 离线管线
#   - 灌库走 src/core/retrieval.get_retriever()（Docker Milvus / Lite 双后端）
#     与 scripts/ingest_kb.py 同一链路，不重复实现向量化/建表逻辑
#   - 同名文档 = 替换语义：同 department 下重传同名文件，先删旧 doc_id 再落盘重灌
#   - .json 若是 Q&A 问答结构则按条切分（每问答对独立成块），否则当文本处理
#   - 向量化需要真实 BGE-M3 权重（生产环境必备）；测试注入假编码器
#   - 模块级互斥锁防并发写同一 Lite 库（milvus-lite 单文件后端）
# ============================================================

import json
import re
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from loguru import logger

from config.settings import settings
from src.chunking.medical_qa_chunker import MedicalQAChunker
from src.api.schemas import (
    UploadDocumentsResponse,
    DocumentTask,
    DocumentListResponse,
    DeleteDocumentResponse,
    StoredFileInfo,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

ALLOWED_SUFFIX = {".md", ".txt", ".json"}
# 网页上传只开放本地医疗库（ADR-0005 决策 1/2：命名收敛为 local_kb）
ALLOWED_UPLOAD_COLLECTIONS = {"local_kb"}
# 查看接口仍允许 local_kb / pubmed（pubmed 目录由离线脚本维护，可只读列出）
ALLOWED_VIEW_COLLECTIONS = {"local_kb", "pubmed"}
# 单文件大小上限 50MB、单批最多 20 个（ADR-0005 决策 6）
_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_FILES_PER_BATCH = 20
# department 只允许中文/字母/数字/下划线/横线，防路径穿越
_SAFE_TAG = re.compile(r"^[\w\u4e00-\u9fff-]{1,32}$")
# 列表接口最多返回的文件数（目录可能很大，如 pubmed 离线库）
_LIST_FILE_LIMIT = 200

_chunker = MedicalQAChunker()

# 后台灌库任务状态表（进程内存态） + 互斥锁
_tasks: Dict[str, dict] = {}
_ingest_lock = threading.Lock()


def _collection_name(collection: str) -> str:
    """API 集合名 → Milvus 物理集合名（local_kb → medical_knowledge_base…）"""
    if collection == "pubmed":
        return settings.milvus_collection_pubmed
    return settings.milvus_collection_kb


def _store_dir(collection: str) -> Path:
    """上传文件落盘目录：local_kb → medical_kb_dir；pubmed → pubmed_offline_dir"""
    base = (
        settings.pubmed_offline_dir
        if collection == "pubmed"
        else settings.medical_kb_dir
    )
    return Path(base)


def _task_model(task: dict) -> DocumentTask:
    return DocumentTask(
        task_id=task["task_id"],
        status=task["status"],
        collection=task["collection"],
        department=task["department"],
        files=task.get("files", 0),
        chunks=task.get("chunks", 0),
        failed=task.get("failed", 0),
        errors=task.get("errors", []),
        created_at=task.get("created_at"),
        finished_at=task.get("finished_at"),
    )


# ---- json Q&A 结构解析 ----

def _is_qa_list(data) -> bool:
    """判断是否为问答对列表：[{"question":..,"answer":..}, ...]"""
    return (
        isinstance(data, list)
        and len(data) > 0
        and all(
            isinstance(item, dict) and item.get("question") and item.get("answer")
            for item in data
        )
    )


def _chunks_from_qa_json(path: Path, base_meta: Dict, qas: List[Dict]) -> List[Dict]:
    """Q&A json → 每个问答对独立成一块（问题+答案同块，不做二次切分）。

    文档 doc_id 保持 = 文件名 stem（整个 json 文件为一个删除/替换单元）；
    块文本形如 "问：…\n答：…"，检索召回时自带上下文。
    """
    out: List[Dict] = []
    for i, item in enumerate(qas):
        text = f"问：{item['question']}\n答：{item['answer']}"
        tokens = _chunker.count_tokens(text)
        if tokens > _chunker.max_tokens:
            # 超长问答对：语义模块二级拆分（复用 chunker 的拆分逻辑）
            for c in _chunker._semantic_split(text, source_prefix=str(item["question"])[:80]):
                out.append({
                    "text": c.text,
                    "tokens": c.tokens,
                    "metadata": {**base_meta, "doc_id": path.stem},
                })
        else:
            out.append({
                "text": text,
                "tokens": tokens,
                "metadata": {**base_meta, "doc_id": path.stem},
            })
    return out


def _load_documents(path: Path, collection: str) -> List[Dict]:
    """读单文件 → 分块前的 documents 列表（[{content, metadata}]）。

    - .json 若为 Q&A 列表：按条切（_chunks_from_qa_json 直接产出块级文本）；
    - 其余/整份 json：按普通文本交给 MedicalQAChunker 处理。
    """
    meta = {
        "title": path.stem,
        "department": path.parent.name,
        "source_type": "medical_kb" if collection == "local_kb" else "pubmed",
        "publish_time": datetime.now().strftime("%Y-%m"),
        "doc_id": path.stem,
    }
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"JSON 解析失败: {path.name}: {e}")
        if _is_qa_list(data):
            return [{
                "content": f"问：{it['question']}\n答：{it['answer']}",
                "metadata": {**meta, "doc_id": path.stem},
            } for it in data]
        # 非 Q&A json：整份序列化后当文本分块
        return [{"content": json.dumps(data, ensure_ascii=False, indent=2), "metadata": meta}]
    return [{"content": path.read_text(encoding="utf-8"), "metadata": meta}]


def _ingest_one(path: Path, collection: str) -> int:
    """读单个文件 → 分块 → 向量化入库，返回分块数。

    替换语义：入库前先按 (doc_id, department) 删旧向量（同目录同名文件覆盖重传）。
    """
    from src.core.retrieval import get_retriever

    col_name = _collection_name(collection)
    retriever = get_retriever()

    documents = _load_documents(path, collection)
    with _ingest_lock:
        retriever.connect()
        # 同名替换：清掉旧 doc_id 向量（不存在则 0，无害）
        try:
            retriever.delete_by_doc_id(col_name, path.stem, department=path.parent.name)
        except Exception as e:
            logger.warning(f"替换前清理旧向量失败（继续插入）: {e}")

        chunks = _chunker.batch_chunk(documents)
        if not chunks:
            logger.warning(f"分块为空，跳过: {path}")
            return 0
        retriever.insert(col_name, chunks, batch_size=64)
    logger.info(f"入库完成: {path.name} → {len(chunks)} 块 → {col_name}")
    return len(chunks)


def _run_task(task_id: str, files: List[Path], collection: str) -> None:
    """后台任务体：逐文件入库并回写任务状态。"""
    task = _tasks[task_id]
    chunks_total = 0
    failed = 0
    try:
        for path in files:
            try:
                chunks_total += _ingest_one(path, collection)
            except Exception as e:
                failed += 1
                task["errors"].append(f"{path.name}: {e}")
                logger.error(f"入库失败 {path.name}: {e}")
        task.update(status="done", chunks=chunks_total, failed=failed)
    except Exception as e:  # 兜底：任务级失败
        task.update(status="error", errors=[*task["errors"], str(e)])
        logger.error(f"灌库任务异常 [{task_id}]: {e}")
    finally:
        task["finished_at"] = datetime.now().isoformat(timespec="seconds")


@router.post("/upload", response_model=UploadDocumentsResponse)
async def upload_documents(
    background: BackgroundTasks,
    files: List[UploadFile] = File(...),
    department: str = Form("通用"),
    collection: str = Form("local_kb"),
):
    """上传知识库文档（.md/.txt/.json），异步分块向量化后写入 local_kb 集合。

    - collection: 仅 `local_kb`（网页上传只开本地库；pubmed 走 ingest_pubmed.py）
    - department: 科室/主题标签，会作为文档元数据并决定落盘子目录
    - 同名文件重传 = 替换：先删旧 doc_id 向量再落盘重灌，不累积重复
    - 返回 task_id，用 GET /documents/tasks/{task_id} 轮询入库结果
    """
    if collection not in ALLOWED_UPLOAD_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection 仅支持 {' / '.join(sorted(ALLOWED_UPLOAD_COLLECTIONS))}（pubmed 文献库请走离线灌库管线）",
        )
    if not _SAFE_TAG.match(department):
        raise HTTPException(
            status_code=400,
            detail="department 仅允许中文/字母/数字/下划线/横线（≤32 字符）",
        )
    if len(files) > _MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"单批最多上传 {_MAX_FILES_PER_BATCH} 个文件",
        )

    store_dir = _store_dir(collection)
    dest_dir = store_dir / department
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for f in files:
        filename = Path(f.filename or "").name  # 去路径，防穿越
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIX:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {filename}（仅 .md/.txt/.json）",
            )
        if not _SAFE_TAG.match(Path(filename).stem):
            raise HTTPException(
                status_code=400,
                detail=f"文件名不合法: {filename}（仅允许中文/字母/数字/下划线/横线）",
            )
        content = await f.read()
        if len(content) > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"文件过大: {filename}（单文件上限 {_MAX_FILE_BYTES // (1024 * 1024)}MB）",
            )
        # 同名覆盖（替换语义在 _ingest_one 里先删旧向量再重灌）
        dest = dest_dir / filename
        dest.write_bytes(content)
        saved.append(dest)
        logger.info(f"文件已保存: {dest} ({len(content)}B)")

    if not saved:
        raise HTTPException(status_code=400, detail="未收到有效文件")

    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "collection": collection,
        "department": department,
        "files": len(saved),
        "chunks": 0,
        "failed": 0,
        "errors": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
    }
    background.add_task(_run_task, task_id, saved, collection)

    return UploadDocumentsResponse(
        task_id=task_id,
        status="processing",
        collection=collection,
        department=department,
        accepted_files=len(saved),
    )


@router.get("/tasks/{task_id}", response_model=DocumentTask)
async def get_task(task_id: str):
    """查询灌库任务状态（processing / done / error，含入库分块数）。"""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return _task_model(task)


@router.get("/tasks", response_model=List[DocumentTask])
async def list_tasks(limit: int = 20):
    """列出最近的知识库灌库任务（按创建时间倒序）。"""
    ordered = sorted(
        _tasks.values(),
        key=lambda t: t.get("created_at") or "",
        reverse=True,
    )
    return [_task_model(t) for t in ordered[: max(1, min(limit, 100))]]


@router.get("", response_model=DocumentListResponse)
async def list_documents(collection: str = "local_kb"):
    """列出指定集合的知识库目录文件与近期任务。

    用于在管理界面里确认"哪些文档已上传、落在哪个科室目录"。
    """
    if collection not in ALLOWED_VIEW_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection 仅支持 {' / '.join(sorted(ALLOWED_VIEW_COLLECTIONS))}",
        )

    base = _store_dir(collection)
    files: List[StoredFileInfo] = []
    if base.exists():
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in ALLOWED_SUFFIX:
                continue
            try:
                rel = p.relative_to(settings.medical_kb_dir
                                    if collection == "local_kb"
                                    else settings.pubmed_offline_dir)
                files.append(StoredFileInfo(
                    path=str(rel).replace("\\", "/"),
                    size=p.stat().st_size,
                    modified=datetime.fromtimestamp(p.stat().st_mtime)
                    .isoformat(timespec="seconds"),
                ))
            except ValueError:
                continue
            if len(files) >= _LIST_FILE_LIMIT:
                break

    ordered = sorted(
        _tasks.values(),
        key=lambda t: t.get("created_at") or "",
        reverse=True,
    )
    tasks = [_task_model(t) for t in ordered[:20]]
    return DocumentListResponse(
        collection=collection,
        files=files,
        tasks=tasks,
    )


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    doc_id: str,
    department: str = "通用",
    collection: str = "local_kb",
):
    """删除一篇文档：先从集合删该 (doc_id, department) 的全部向量，再删落盘文件。

    - 文档名去后缀即 doc_id（与上传/替换同一规则）
    - 向量删除成功但文件不存在 / 文件删除失败都会如实报告（不静默吞错）
    """
    if collection not in ALLOWED_UPLOAD_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection 仅支持 {' / '.join(sorted(ALLOWED_UPLOAD_COLLECTIONS))}",
        )
    if not _SAFE_TAG.match(department):
        raise HTTPException(status_code=400, detail="department 不合法")

    from src.core.retrieval import get_retriever

    col_name = _collection_name(collection)
    retriever = get_retriever()
    retriever.connect()
    try:
        deleted_vectors = retriever.delete_by_doc_id(
            col_name, doc_id, department=department
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量删除失败: {e}")

    # 定位并删除落盘文件（md/txt/json 三选一）
    # 逐文件容错：单文件删除失败（权限/占用/系统回收站不可用）如实记录到 errors，
    # 不整体 500——向量已删、文件仍留在磁盘时响应里能看得到（ADR-0005 决策 5）。
    store_dir = _store_dir(collection) / department
    file_deleted = 0
    delete_errors: List[str] = []
    for suffix in ALLOWED_SUFFIX:
        p = store_dir / f"{doc_id}{suffix}"
        if p.exists() and p.is_file():
            try:
                p.unlink()
                file_deleted += 1
                logger.info(f"文件已删除: {p}")
            except OSError as e:
                delete_errors.append(f"{p.name}: {e}")
                logger.warning(f"文件删除失败（向量已删）: {p}: {e}")

    return DeleteDocumentResponse(
        doc_id=doc_id,
        department=department,
        collection=collection,
        deleted_vectors=deleted_vectors,
        deleted_files=file_deleted,
        errors=delete_errors,
    )
