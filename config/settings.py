# ============================================================
# 全科医疗问诊RAG智能助手 - 全局配置
# 来源：简历技术栈 + Q2-Q25 各项参数设置
# ============================================================

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """全局配置，所有值均对应问答文档中的明确参数"""

    # ========== 模型路径 ( GPTQ INT4量化) ==========
    qwen_model_path: str = os.getenv("QWEN_MODEL_PATH", "/models/Qwen-14B-Chat-GPTQ-Int4")
    bge_model_path: str = os.getenv("BGE_MODEL_PATH", "/models/bge-m3")

    # ========== 设备配置 ( GPU优先，CPU fallback) ==========
    device: str = os.getenv("DEVICE", "cuda" if os.getenv("FORCE_CPU", "0") == "0" else "cpu")
    max_gpu_memory: int = 10  # GB, 量化后模型总占用 <10G
    use_gptq: bool = False  # CPU模式不使用GPTQ，加载标准transformers模型
    use_milvus_lite: bool = True  # CPU模式用Milvus Lite嵌入式，无需Docker

    # ========== Milvus 向量库 ( 单机版, 两个Collection, 1024维) ==========
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_kb: str = os.getenv("MILVUS_COLLECTION_KB", "medical_knowledge_base")
    milvus_collection_pubmed: str = os.getenv("MILVUS_COLLECTION_PUBMED", "pubmed_literature")
    embedding_dim: int = 1024  # BGE-M3 向量维度 (项目需求)

    # ========== 服务配置 ==========
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs"))

    # ========== 会话配置 (项目需求) ==========
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h
    session_max_tokens: int = int(os.getenv("SESSION_MAX_TOKENS", "2000"))     # 触发压缩阈值
    session_compress_rounds: int = int(os.getenv("SESSION_COMPRESS_ROUNDS", "4"))  # >4轮压缩

    # ========== 检索配置 (项目需求) ==========
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))

    # ========== 分块配置 ( 阈值定制) ==========
    chunk_max_tokens: int = 800   #  从默认512调整到800
    chunk_min_tokens: int = 100   #  最小token阈值
    chunk_overlap_tokens: int = 50

    # ========== 重试与降级 (项目需求) ==========
    max_retry_generation: int = int(os.getenv("MAX_RETRY_GENERATION", "2"))   #  最多重试2次
    max_retry_tool_call: int = int(os.getenv("MAX_RETRY_TOOL_CALL", "2"))     #  工具调用重试2次

    # ========== LLM-Judge 配置 ( 三维均>6分) ==========
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

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # 允许 .env 中额外字段


# 全局单例
settings = Settings()
