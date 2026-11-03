# ============================================================
# 全科医疗问诊RAG智能助手 - FastAPI 应用入口
# ============================================================

import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

from config.settings import settings
from src.api.routes import router
from src.api.documents import router as documents_router
from src.api.sessions import router as sessions_router
from src.api.feedback import router as feedback_router

# ---- 应用初始化 ----

app = FastAPI(
    title="全科医疗问诊RAG智能助手",
    description=(
        "基于 RAGFlow + Qwen-14B + BGE-M3 + Milvus 的私有化医疗问诊Agent。\n"
        "功能: 双源检索增强、Q&A语义分块、LLM-Judge幻觉防控、动态上下文压缩。\n"
        "特性: 全链路本地化部署，数据不出域。\n\n"
        "【免责声明】本产品仅提供医学科普参考，不构成任何诊疗建议。"
        "身体不适请及时前往正规医疗机构就诊。"
    ),
    version="1.0.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 内网环境可放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)
app.include_router(documents_router)
app.include_router(sessions_router)
app.include_router(feedback_router)

# ---- 静态资源与网页管理台 ----
# 新前端：frontend/（Vue3 + Vite），构建产物 frontend/dist 由本服务直接托管，
# 不依赖 CDN，断网可用。未构建时回落到旧的单页管理台，保证"clone 下来就能看"。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"
_LEGACY_INDEX = Path(__file__).resolve().parent / "static" / "index.html"

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

if (_FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_FRONTEND_DIST / "assets")),
        name="assets",
    )


@app.get("/", include_in_schema=False)
async def index():
    """管理台入口：优先 Vue 构建产物，其次旧单页。

    hash 路由（/#/admin）不需要服务端回退规则，因此这里只处理根路径。
    """
    dist_index = _FRONTEND_DIST / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)
    return FileResponse(_LEGACY_INDEX)


# ---- Compliance middleware ----
from fastapi import Request

@app.middleware("http")
async def add_compliance_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Medical-Disclaimer"] = (
        "This system provides medical reference information only. "
        "It does NOT provide diagnosis, prescription, or treatment advice. "
        "Consult a licensed physician for any health concerns."
    )
    return response


# ---- 生命周期事件 ----

@app.on_event("startup")
async def startup():
    """服务启动：连接 Milvus，预热模型"""
    logger.info("=" * 60)
    logger.info("全科医疗问诊RAG智能助手 启动中...")
    logger.info(f"模型路径: {settings.qwen_model_path}")
    from src.core.retrieval import backend_name
    logger.info(f"检索后端: {backend_name()}（VECTOR_BACKEND={settings.vector_backend or '未设置'}）")
    load_mode = (
        "GPTQ INT4 (Qwen-14B ~8G 显存)" if settings.use_gptq
        else "标准 transformers 加载"
    )
    logger.info(f"模型加载: {load_mode}")
    logger.info("=" * 60)

    # 数据库建表（幂等）：MySQL 真源，本地/CI 自动落 SQLite
    try:
        from src.storage.db import init_schema
        init_schema()
        logger.info(f"数据库就绪: {settings.database_url.split('@')[-1]}")
    except Exception as e:
        logger.error(f"数据库初始化失败（会话/反馈将不可用）: {e}")

    # 预连接检索库（按配置选定的后端，见 src/core/retrieval.get_retriever）
    try:
        from src.core.retrieval import get_retriever
        get_retriever().connect()
        logger.info(f"向量库连接就绪: {backend_name()}")
    except Exception as e:
        logger.warning(f"向量库连接未就绪 (不影响启动): {e}")

    # 预加载 BGE-M3 (首次调用时会懒加载)
    try:
        from src.core.retrieval import get_retriever
        _ = get_retriever().embedder
        logger.info("BGE-M3 已预加载")
    except Exception as e:
        logger.warning(f"BGE-M3 预加载跳过: {e}")

    logger.info("服务启动完成，等待请求...")
    logger.info(f"管理台入口: http://{settings.api_host}:{settings.api_port}/  (聊天 + 知识库上传/删除)")


@app.on_event("shutdown")
async def shutdown():
    """服务关闭"""
    logger.info("服务关闭中...")
    from src.monitoring.logger import request_logger
    request_logger.flush()


# ---- 直接运行入口 ----

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
