# ============================================================
# 全科医疗问诊RAG智能助手 - 全局配置
# ============================================================

import os
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """全局配置：模型路径、向量库连接、阈值参数"""

    # ========== 模型路径 (标准加载 / GPTQ INT4 可选) ==========
    qwen_model_path: str = os.getenv("QWEN_MODEL_PATH", "/models/Qwen-14B-Chat-GPTQ-Int4")
    bge_model_path: str = os.getenv("BGE_MODEL_PATH", "/models/bge-m3")

    # ========== LLM 对话总机 ==========
    # qwen: 真实模型后端；fake: 确定性假话务员（测试/CI/演示，不加载模型）
    llm_backend: str = os.getenv("LLM_BACKEND", "qwen")

    # ========== 设备配置 ( GPU优先，CPU fallback) ==========
    device: str = os.getenv("DEVICE", "cuda" if os.getenv("FORCE_CPU", "0") == "0" else "cpu")
    max_gpu_memory: int = 10  # GB, 量化后模型总占用 <10G
    use_gptq: bool = os.getenv("USE_GPTQ", "0").lower() in (
        "1", "true", "yes", "on"
    )  # 默认 0=标准 transformers 加载（CPU/复现即用）；1=GPTQ INT4（生产 docker compose 已置 1，本地/CI 无需开）
    use_milvus_lite: bool = os.getenv("USE_MILVUS_LITE", "1").lower() in (
        "1", "true", "yes", "on"
    )  # 0/false → Docker Milvus（生产/docker compose）；1/true → 嵌入式 Lite（本地开发/CI）

    # ========== 检索后端选择（milvus | milvus_lite | pgvector） ==========
    # 留空 = 沿用上面的 USE_MILVUS_LITE（兼容既有部署）；显式配置优先。
    vector_backend: str = os.getenv("VECTOR_BACKEND", "")
    # pgvector 连接串与表名；口令请在生产用环境变量覆盖
    pg_dsn: str = os.getenv("PG_DSN") or (
        "postgresql://medrag:medrag_dev2026@127.0.0.1:5432/medical_rag"
    )
    pg_table: str = os.getenv("PG_TABLE", "kb_chunks")
    pg_pool_max: int = int(os.getenv("PG_POOL_MAX", "8"))

    # ========== Milvus 向量库 ( 单机版, 两个Collection, 1024维) ==========
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_kb: str = os.getenv("MILVUS_COLLECTION_KB", "medical_knowledge_base")
    milvus_collection_pubmed: str = os.getenv("MILVUS_COLLECTION_PUBMED", "pubmed_literature")
    embedding_dim: int = 1024  # BGE-M3 向量维度 

    # ========== 服务配置 ==========
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs"))

    # ========== 流式输出（SSE） ==========
    # 高风险链路「先校验后回放」的分片字符数：一次事件推多少字。
    # 只影响下发粒度，不改变内容，也不人为 sleep 制造打字机效果。
    stream_replay_chunk_chars: int = int(os.getenv("STREAM_REPLAY_CHUNK_CHARS", "24"))
    # 单条流式问诊的总时长上限（秒）：超时后补一个 timeout 出口的 final 事件
    stream_timeout_seconds: int = int(os.getenv("STREAM_TIMEOUT_SECONDS", "90"))
    # SSE 心跳间隔（秒）：跨反向代理时防止空闲连接被掐断
    sse_heartbeat_seconds: int = int(os.getenv("SSE_HEARTBEAT_SECONDS", "15"))

    # ========== 数据持久化（MySQL 真源 / SQLite 本地与 CI） ==========
    # 生产 compose 注入 mysql+pymysql://...；本地开发与 CI 默认落 SQLite 文件，
    # 两者共用同一套 SQLAlchemy 模型与仓储代码（见 docs/adr/0008-storage-layering.md）
    # 注意用 `or` 而不是 getenv 默认值：.env 里写 DATABASE_URL=（留空）时
    # getenv 返回空串，create_engine("") 会直接炸
    database_url: str = os.getenv("DATABASE_URL") or (
        f"sqlite:///{(PROJECT_ROOT / 'volumes' / 'app.db').as_posix()}"
    )
    db_echo: bool = os.getenv("DB_ECHO", "0").lower() in ("1", "true", "yes", "on")
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    db_pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))

    # ========== Redis（会话热缓存 + Celery broker） ==========
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "1").lower() in (
        "1", "true", "yes", "on"
    )
    # 热会话在 Redis 的存活时间；MySQL 里的历史永久保留（复盘调优用）
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

    # ========== 异步任务（Celery / 进程内） ==========
    # celery: 投递到 broker 由独立 worker 执行（生产 compose）
    # inline: BackgroundTasks 进程内执行（本地开发 / 测试 / 无 Redis）
    task_mode: str = os.getenv("TASK_MODE", "inline")
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "")
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "")

    # ========== 会话配置  ==========
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h
    session_max_tokens: int = int(os.getenv("SESSION_MAX_TOKENS", "2000"))     # 触发压缩阈值
    session_compress_rounds: int = int(os.getenv("SESSION_COMPRESS_ROUNDS", "4"))  # >4轮压缩

    # ========== 检索配置  ==========
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))

    # ========== 分块配置 ( 阈值定制) ==========
    chunk_max_tokens: int = 800   #  从默认512调整到800
    chunk_min_tokens: int = 100   #  最小token阈值
    chunk_overlap_tokens: int = 50

    # ========== 重试与降级  ==========
    max_retry_generation: int = int(os.getenv("MAX_RETRY_GENERATION", "2"))   #  最多重试2次
    max_retry_tool_call: int = int(os.getenv("MAX_RETRY_TOOL_CALL", "2"))     #  工具调用重试2次

    # ========== LLM-Judge 配置 ( 三维均 ≥ 阈值分即通过, 含阈值; 默认6) ==========
    judge_score_threshold: int = int(os.getenv("JUDGE_SCORE_THRESHOLD", "6"))

    # ========== 关键词分类 ( 快速匹配) ==========
    classification_keywords_path: str = str(PROJECT_ROOT / "data" / "common_diseases_drugs.txt")

    # ========== Prompt 模板路径 ==========
    prompts_dir: str = str(PROJECT_ROOT / "prompts")

    # 数据目录
    pubmed_offline_dir: str = os.getenv("PUBMED_OFFLINE_DIR", str(PROJECT_ROOT / "data" / "pubmed"))
    medical_kb_dir: str = os.getenv("MEDICAL_KB_DIR", str(PROJECT_ROOT / "data" / "medical_kb"))

    # CPU模式标记
    force_cpu: str = os.getenv("FORCE_CPU", "0")

    @field_validator("database_url", mode="before")
    @classmethod
    def _fallback_database_url(cls, value):
        """DATABASE_URL 留空时回落 SQLite。

        必须用校验器而不是 os.getenv 默认值：BaseSettings 的取值顺序是
        "环境变量 > .env > 类默认值"，.env 里写 `DATABASE_URL=` 会把空串
        直接塞进来，create_engine("") 会在启动时崩掉。
        """
        if value:
            return value
        return f"sqlite:///{(PROJECT_ROOT / 'volumes' / 'app.db').as_posix()}"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # 允许 .env 中额外字段


# 全局单例
settings = Settings()
