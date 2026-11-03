# ============================================================
# 知识库文档管理路由（上传 → 分块 → 向量化 → 入库 / 删除 / 替换 / 重建）
#
#   POST   /api/v1/documents/upload              multipart 上传 → 异步灌库
#   POST   /api/v1/documents/preview             切分预览（dry-run，不落盘不入库）
#   POST   /api/v1/documents/{doc_id}/rebuild    单文档重建（删旧向量 → 重切重嵌）
#   POST   /api/v1/documents/rebuild             集合级全量重建
#   GET    /api/v1/documents/tasks/{task_id}     轮询任务状态
#   GET    /api/v1/documents                     目录文件 + 近期任务
#   DELETE /api/v1/documents/{doc_id}            删除文档（文件 + 向量 + 元数据）
#
# 设计约束（见 docs/adr/0005-upload-delete-api.md、0008-storage-layering.md）：
#   - 网页上传只开放 local_kb 集合；pubmed 文献库只经 ingest_pubmed.py 离线管线
#   - 灌库走 src/core/retrieval.get_retriever()（Docker Milvus / Lite 双后端）
#     与 scripts/ingest_kb.py 同一链路，不重复实现向量化/建表逻辑
#   - 同名文档 = 替换语义：同 department 下重传同名文件，先删旧 doc_id 再落盘重灌
#   - 任务状态落 MySQL（kb_task）：API 与 Celery worker 不共享内存，必须外置
#   - TASK_MODE=inline 用 BackgroundTasks 进程内跑，celery 则投递到 worker；
#     两条路径共用 src/tasks/runner.run_task，不存在第二份实现
#   - 切分预览写到临时目录，复用真实分块器后立即删除，绝不污染知识库目录
# ============================================================

import json
import re
import shutil
import tempfile
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from loguru import logger

from config.settings import settings
from src.chunking.medical_qa_chunker import MedicalQAChunker
from src.api.schemas import (
    ChunkPreviewResponse,
    UploadDocumentsResponse,
    DocumentTask,
    DocumentListResponse,
    DeleteDocumentResponse,
    PreviewChunk,
    PreviewFile,
    StoredFileInfo,
)
from src.storage import repository
from src.tasks.runner import submit

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

ALLOWED_SUFFIX = {".md", ".txt", ".json"}
# 网页上传只开放本地医疗库（ADR-0005 决策 1/2：命名收敛为 local_kb）
ALLOWED_UPLOAD_COLLECTIONS = {"local_kb"}
# 查看/重建接口仍允许 local_kb / pubmed（pubmed 目录由离线脚本维护）
ALLOWED_VIEW_COLLECTIONS = {"local_kb", "pubmed"}
# 单文件大小上限 50MB、单批最多 20 个（ADR-0005 决策 6）
_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_FILES_PER_BATCH = 20
# department 只允许中文/字母/数字/下划线/横线，防路径穿越
_SAFE_TAG = re.compile(r"^[\w\u4e00-\u9fff-]{1,32}$")
# 列表接口最多返回的文件数（目录可能很大，如 pubmed 离线库）
_LIST_FILE_LIMIT = 200
# 预览单文件最多回传的块数（前端只展示前若干块，余量给统计）
_PREVIEW_CHUNK_LIMIT = 200

_chunker = MedicalQAChunker()

# 模块级互斥锁防并发写同一 Lite 库（milvus-lite 单文件后端）
_ingest_lock = threading.Lock()


# ---------------- 路径与命名 ----------------


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


def _iter_collection_files(collection: str) -> List[Path]:
    """集合目录下的全部可入库文件（集合级重建用）。"""
    base = _store_dir(collection)
    if not base.exists():
        return []
    return [
        p for p in sorted(base.rglob("*"))
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIX
    ]


def _locate_document(
    collection: str, doc_id: str, department: str
) -> Optional[Path]:
    """按 (集合, 科室, doc_id) 定位落盘文件；三个后缀依次尝试。"""
    folder = _store_dir(collection) / department
    for suffix in ALLOWED_SUFFIX:
        candidate = folder / f"{doc_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _task_model(task: dict) -> DocumentTask:
    return DocumentTask(
        task_id=task["task_id"],
        status=task["status"],
        collection=task["collection"],
        kind=task.get("kind", "upload"),
        doc_id=task.get("doc_id", ""),
        executor=task.get("executor", "inline"),
        department=task.get("department", ""),
        files=task.get("files", 0),
        chunks=task.get("chunks", 0),
        failed=task.get("failed", 0),
        errors=task.get("errors", []),
        created_at=task.get("created_at"),
        finished_at=task.get("finished_at"),
    )


def _validate_department(department: str) -> None:
    if not _SAFE_TAG.match(department):
        raise HTTPException(
            status_code=400,
            detail="department 仅允许中文/字母/数字/下划线/横线（≤32 字符）",
        )


def _validate_upload_file(filename: str) -> str:
    """校验文件名与扩展名，返回净化后的文件名。"""
    name = Path(filename or "").name  # 去路径，防穿越
    if Path(name).suffix.lower() not in ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {name}（仅 .md/.txt/.json）",
        )
    if not _SAFE_TAG.match(Path(name).stem):
        raise HTTPException(
            status_code=400,
            detail=f"文件名不合法: {name}（仅允许中文/字母/数字/下划线/横线）",
        )
    return name


# ---------------- json Q&A 结构解析 ----------------


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
    for item in qas:
        text = f"问：{item['question']}\n答：{item['answer']}"
        tokens = _chunker.count_tokens(text)
        if tokens > _chunker.max_tokens:
            # 超长问答对：语义模块二级拆分（复用 chunker 的拆分逻辑）
            for c in _chunker._semantic_split(
                text, source_prefix=str(item["question"])[:80]
            ):
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
        return [{
            "content": json.dumps(data, ensure_ascii=False, indent=2),
            "metadata": meta,
        }]
    return [{"content": path.read_text(encoding="utf-8"), "metadata": meta}]


def _ingest_one(path: Path, collection: str) -> int:
    """读单个文件 → 分块 → 向量化入库，返回分块数。

    替换/重建语义：入库前先按 (doc_id, department) 删旧向量。
    """
    from src.core.retrieval import get_retriever

    col_name = _collection_name(collection)
    retriever = get_retriever()

    documents = _load_documents(path, collection)
    with _ingest_lock:
        retriever.connect()
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


# ---------------- 任务 ----------------


def _new_task(
    background: BackgroundTasks,
    *,
    kind: str,
    collection: str,
    department: str,
    doc_id: str = "",
    files: int = 0,
    payload: Dict,
) -> str:
    """建任务行 → 投递执行 → 回写执行器。返回 task_id。"""
    task_id = uuid.uuid4().hex[:12]
    repository.create_task(
        task_id=task_id,
        kind=kind,
        collection=collection,
        department=department,
        doc_id=doc_id,
        files=files,
    )
    executor = submit(background, kind, task_id, payload)
    repository.update_task(task_id, executor=executor)
    return task_id


# ---------------- 路由 ----------------


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
    _validate_department(department)
    if len(files) > _MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"单批最多上传 {_MAX_FILES_PER_BATCH} 个文件",
        )

    dest_dir = _store_dir(collection) / department

    # 两阶段：先全部校验完再落盘。否则第 N 个文件非法时，前 N-1 个已经写进
    # 知识库目录却没有任务去灌库，留下"磁盘有文件、库里没向量"的孤儿。
    prepared: List[Tuple[str, bytes]] = []
    for f in files:
        filename = _validate_upload_file(f.filename or "")
        content = await f.read()
        if len(content) > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"文件过大: {filename}（单文件上限 {_MAX_FILE_BYTES // (1024 * 1024)}MB）",
            )
        prepared.append((filename, content))

    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for filename, content in prepared:
        # 同名覆盖（替换语义在 _ingest_one 里先删旧向量再重灌）
        dest = dest_dir / filename
        dest.write_bytes(content)
        saved.append(dest)
        logger.info(f"文件已保存: {dest} ({len(content)}B)")

    if not saved:
        raise HTTPException(status_code=400, detail="未收到有效文件")

    task_id = _new_task(
        background,
        kind="upload",
        collection=collection,
        department=department,
        files=len(saved),
        payload={"files": [str(p) for p in saved], "collection": collection},
    )

    return UploadDocumentsResponse(
        task_id=task_id,
        status="processing",
        collection=collection,
        department=department,
        accepted_files=len(saved),
    )


@router.post("/preview", response_model=ChunkPreviewResponse)
async def preview_chunks(
    files: List[UploadFile] = File(...),
    department: str = Form("通用"),
    collection: str = Form("local_kb"),
    max_chunks: int = Form(50),
):
    """切分预览（dry-run）：用真实分块器切一遍，返回块数与正文，不落库。

    分块参数是摄入侧全局参数（max 800 / min 100 / overlap 50 token），
    先预览再入库，才能判断参数是否适合这批文档。文件只写临时目录，
    请求结束即删除，绝不进知识库目录。
    """
    if collection not in ALLOWED_UPLOAD_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection 仅支持 {' / '.join(sorted(ALLOWED_UPLOAD_COLLECTIONS))}",
        )
    _validate_department(department)
    if not files:
        raise HTTPException(status_code=400, detail="未收到有效文件")
    if len(files) > _MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"单批最多预览 {_MAX_FILES_PER_BATCH} 个文件",
        )

    limit = max(1, min(max_chunks, _PREVIEW_CHUNK_LIMIT))
    tmp_root = Path(tempfile.mkdtemp(prefix="medrag_preview_"))
    dest_dir = tmp_root / department
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        preview_files: List[PreviewFile] = []
        total_chunks = 0
        for f in files:
            filename = _validate_upload_file(f.filename or "")
            content = await f.read()
            if len(content) > _MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件过大: {filename}",
                )
            path = dest_dir / filename
            path.write_bytes(content)

            documents = _load_documents(path, collection)
            chunks = _chunker.batch_chunk(documents)
            total_chunks += len(chunks)
            preview_files.append(PreviewFile(
                name=filename,
                doc_id=path.stem,
                department=department,
                chars=len(content.decode("utf-8", errors="ignore")),
                chunk_count=len(chunks),
                returned=min(len(chunks), limit),
                truncated=len(chunks) > limit,
                # batch_chunk 返回的是 dict（text/tokens/metadata），不是 dataclass
                chunks=[
                    PreviewChunk(index=i, tokens=int(c["tokens"]), text=c["text"])
                    for i, c in enumerate(chunks[:limit])
                ],
            ))

        return ChunkPreviewResponse(
            collection=collection,
            department=department,
            total_chunks=total_chunks,
            files=preview_files,
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


@router.post("/rebuild", response_model=DocumentTask)
async def rebuild_collection(
    background: BackgroundTasks,
    collection: str = "local_kb",
):
    """集合级全量重建：遍历目录重灌（换模型 / 改分块参数后用）。

    重建 = 逐文件"删旧向量 → 重切重嵌"，按 doc_id 幂等，可安全重跑；
    任务粒度是集合，失败文件数如实上报。
    """
    if collection not in ALLOWED_VIEW_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection 仅支持 {' / '.join(sorted(ALLOWED_VIEW_COLLECTIONS))}",
        )
    files = _iter_collection_files(collection)
    if not files:
        raise HTTPException(
            status_code=404, detail=f"集合目录为空: {_store_dir(collection)}"
        )

    task_id = _new_task(
        background,
        kind="rebuild_collection",
        collection=collection,
        department="",
        files=len(files),
        payload={"collection": collection},
    )
    logger.info(f"[重建] 集合 {collection}: {len(files)} 个文件, task={task_id}")
    return _task_model(repository.get_task(task_id))


@router.post("/{doc_id}/rebuild", response_model=DocumentTask)
async def rebuild_document(
    background: BackgroundTasks,
    doc_id: str,
    department: str = "通用",
    collection: str = "local_kb",
):
    """单文档重建：删旧向量 → 重新分块向量化（改分块参数后的日常入口）。"""
    if collection not in ALLOWED_VIEW_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection 仅支持 {' / '.join(sorted(ALLOWED_VIEW_COLLECTIONS))}",
        )
    _validate_department(department)
    path = _locate_document(collection, doc_id, department)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"文档不存在: {collection}/{department}/{doc_id}",
        )

    task_id = _new_task(
        background,
        kind="rebuild_doc",
        collection=collection,
        department=department,
        doc_id=doc_id,
        files=1,
        payload={
            "collection": collection,
            "doc_id": doc_id,
            "department": department,
        },
    )
    logger.info(f"[重建] 文档 {doc_id}: task={task_id}")
    return _task_model(repository.get_task(task_id))


@router.get("/tasks/{task_id}", response_model=DocumentTask)
async def get_task(task_id: str):
    """查询灌库/重建任务状态（processing / done / error）。"""
    task = repository.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return _task_model(task)


@router.get("/tasks", response_model=List[DocumentTask])
async def list_tasks(limit: int = 20, collection: str = ""):
    """列出最近的知识库任务（按创建时间倒序，可按集合过滤）。"""
    rows = repository.list_tasks(limit=limit, collection=collection)
    return [_task_model(t) for t in rows]


@router.get("", response_model=DocumentListResponse)
async def list_documents(collection: str = "local_kb"):
    """列出指定集合的知识库目录文件与近期任务。

    文件清单以磁盘为源真（上传/删除都写同一棵树）；任务与分块数来自
    kb_task / kb_document，进程重启后仍可见。
    """
    if collection not in ALLOWED_VIEW_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection 仅支持 {' / '.join(sorted(ALLOWED_VIEW_COLLECTIONS))}",
        )

    base = _store_dir(collection)
    # (doc_id, department) → kb_document 行：磁盘清单只给"有哪些文件"，
    # "入库多少块"必须来自入库记录，缺这条前端就只能显示一个空表
    indexed_docs = {
        (d["doc_id"], d["department"]): d
        for d in repository.list_documents(collection)
    }

    files: List[StoredFileInfo] = []
    if base.exists():
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in ALLOWED_SUFFIX:
                continue
            try:
                rel = p.relative_to(base)
            except ValueError:
                continue
            meta = indexed_docs.get((p.stem, p.parent.name))
            files.append(StoredFileInfo(
                path=str(rel).replace("\\", "/"),
                doc_id=p.stem,
                department=p.parent.name,
                size=p.stat().st_size,
                modified=datetime.fromtimestamp(p.stat().st_mtime)
                .isoformat(timespec="seconds"),
                chunks=int(meta["chunk_count"]) if meta else 0,
                indexed=meta is not None,
            ))
            if len(files) >= _LIST_FILE_LIMIT:
                break

    tasks = [_task_model(t) for t in repository.list_tasks(limit=20, collection=collection)]
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
    """删除一篇文档：先删向量，再删落盘文件与元数据。

    - 文档名去后缀即 doc_id（与上传/替换同一规则）
    - 向量删除成功但文件不存在 / 文件删除失败都会如实报告（不静默吞错）
    """
    if collection not in ALLOWED_UPLOAD_COLLECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"collection 仅支持 {' / '.join(sorted(ALLOWED_UPLOAD_COLLECTIONS))}",
        )
    _validate_department(department)

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

    # 逐文件容错：单文件删除失败（权限/占用）如实记录到 errors，不整体 500
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

    repository.delete_document_meta(collection, doc_id, department)

    return DeleteDocumentResponse(
        doc_id=doc_id,
        department=department,
        collection=collection,
        deleted_vectors=deleted_vectors,
        deleted_files=file_deleted,
        errors=delete_errors,
    )
