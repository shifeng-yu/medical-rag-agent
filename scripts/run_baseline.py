# ============================================================
# Baseline RAG 对比脚本
# 用于与优化方案做 AB 测试 ( 同一台机器、统一接口格式)
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from typing import List, Dict
from loguru import logger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# ---- Baseline 请求/响应模型 ( 与优化方案格式一致) ----

class QueryRequest(BaseModel):
    query: str
    session_id: str = "baseline"


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    latency_ms: float
    session_id: str = "baseline"


# ---- Baseline RAG 实现 ( 固定512分块+单库) ----

class BaselineRAG:
    """
    Baseline RAG 
    - LlamaIndex 框架
    - 固定 512 字符分块 (通用分块)
    - 仅本地知识库单源检索
    """

    def __init__(self):
        self._index = None

    def build_index(self):
        """用 LlamaIndex 构建单库索引 """
        try:
            from llama_index.core import (
                VectorStoreIndex, SimpleDirectoryReader, Settings
            )
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.core.node_parser import SentenceSplitter

            # 固定 512 字符分块 
            Settings.text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
            )
            # Embedding
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="/models/bge-m3",
                device="cuda",
            )

            # 加载本地知识库
            from config.settings import settings
            kb_dir = settings.medical_kb_dir
            if not Path(kb_dir).exists():
                Path(kb_dir).mkdir(parents=True, exist_ok=True)

            documents = SimpleDirectoryReader(kb_dir).load_data()
            self._index = VectorStoreIndex.from_documents(documents)
            logger.info(f"Baseline 索引构建完成: {len(documents)} 文档")

        except Exception as e:
            logger.error(f"Baseline索引构建失败: {e}")
            raise

    def query(self, query_text: str) -> Dict:
        """执行 Baseline RAG 查询"""
        if self._index is None:
            self.build_index()

        start = time.perf_counter()

        query_engine = self._index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )
        response = query_engine.query(query_text)

        latency_ms = (time.perf_counter() - start) * 1000

        sources = [
            {"content": node.text[:200], "score": node.score or 0}
            for node in response.source_nodes
        ] if hasattr(response, "source_nodes") else []

        return {
            "answer": str(response),
            "sources": sources,
            "latency_ms": latency_ms,
        }


# ---- FastAPI 服务 ----

baseline = BaselineRAG()
app = FastAPI(title="Baseline RAG (对照实验)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("Baseline RAG 服务启动中...")
    baseline.build_index()


@app.post("/api/v1/chat", response_model=QueryResponse)
async def baseline_chat(req: QueryRequest):
    """Baseline 问诊接口 ( 格式与优化方案完全一致)"""
    result = baseline.query(req.query)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
    )


# ---- 离线评估脚本 ( MedQA 2134条) ----

def evaluate_on_medqa(test_file: str, api_url: str) -> Dict:
    """
    MedQA 测试集评估 
    """
    import json
    import requests

    logger.info(f"加载测试集: {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    correct = 0
    total = len(test_data)
    latencies = []

    for i, item in enumerate(test_data):
        query = item.get("question", "")
        expected = item.get("answer", "")

        try:
            resp = requests.post(
                f"{api_url}/api/v1/chat",
                json={"query": query},
                timeout=30,
            )
            data = resp.json()
            latencies.append(data.get("latency_ms", 0))

            # 简单匹配判断 ( 与标准答案比对)
            if expected and expected in data.get("answer", ""):
                correct += 1

        except Exception as e:
            logger.error(f"评估错误 [{i}]: {e}")

        if (i + 1) % 100 == 0:
            logger.info(f"评估进度: {i + 1}/{total}, 当前准确率: {correct / (i + 1):.2%}")

    top1_accuracy = correct / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    result = {
        "total_samples": total,
        "top1_accuracy": round(top1_accuracy, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "description": " 使用MedQA2134条样本, 固定512字符分块+单库检索",
    }
    logger.info(f"评估完成: {result}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, help="MedQA测试集路径, 进行评估")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="API地址")
    args = parser.parse_args()

    if args.eval:
        result = evaluate_on_medqa(args.eval, args.api)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        logger.info("启动 Baseline RAG 服务 (端口 8001)")
        uvicorn.run(app, host="0.0.0.0", port=8001)
