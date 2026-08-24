# ============================================================
# 本地医疗知识库数据导入脚本
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from typing import List, Dict
from loguru import logger
from pymilvus import Collection, connections
from config.settings import settings
from src.chunking.medical_qa_chunker import MedicalQAChunker


def load_documents(kb_dir: str) -> List[Dict]:
    """加载本地知识库文档 (支持 JSON/MD/TXT)"""
    docs = []
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        logger.error(f"知识库目录不存在: {kb_dir}")
        return docs

    for file_path in kb_path.rglob("*"):
        if file_path.suffix in [".json", ".md", ".txt"]:
            try:
                content = file_path.read_text(encoding="utf-8")
                docs.append({
                    "content": content,
                    "metadata": {
                        "doc_id": file_path.stem,
                        "title": file_path.stem,
                        "department": file_path.parent.name,
                        "source_type": "medical_kb",
                        "publish_time": "",
                    },
                })
            except Exception as e:
                logger.warning(f"文件读取失败: {file_path} - {e}")

    logger.info(f"已加载 {len(docs)} 个知识库文档")
    return docs


def ingest_to_milvus(
    chunks: List[Dict],
    collection_name: str,
    batch_size: int = 100,
):
    """将分块结果向量化并写入 Milvus """
    from sentence_transformers import SentenceTransformer

    # 加载 BGE-M3
    logger.info(f"加载 BGE-M3: {settings.bge_model_path}")
    model = SentenceTransformer(settings.bge_model_path, device=settings.device)

    # 连接 Milvus
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )
    collection = Collection(collection_name)

    total = len(chunks)
    inserted = 0
    start = time.perf_counter()

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        # BGE-M3 向量化 ( 1024维)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # 构建写入数据
        data = [
            embeddings,
            texts,
            [c["metadata"].get("title", "") for c in batch],
            [c["metadata"].get("department", "") for c in batch],
            [c["metadata"].get("publish_time", "") for c in batch],
            [c["metadata"].get("source_type", "medical_kb") for c in batch],
            [c["metadata"].get("doc_id", "") for c in batch],
        ]

        collection.insert(data)
        inserted += len(batch)
        logger.info(f"导入进度: {inserted}/{total}")

    collection.flush()
    latency = time.perf_counter() - start
    logger.info(
        f"导入完成: {inserted} 条 → {collection_name}, "
        f"耗时={latency:.1f}s"
    )


def main():
    kb_dir = settings.medical_kb_dir
    logger.info(f"开始导入本地知识库: {kb_dir}")

    # 1. 加载文档
    documents = load_documents(kb_dir)
    if not documents:
        logger.warning("知识库为空，生成示例数据...")
        documents = _generate_sample_docs()

    # 2. Q&A 分块 ( 定制分块策略)
    chunker = MedicalQAChunker()
    chunks = chunker.batch_chunk(documents)
    logger.info(f"分块完成: {len(documents)} 文档 → {len(chunks)} 块")

    # 3. 向量化 + 写入 Milvus
    ingest_to_milvus(chunks, settings.milvus_collection_kb)
    logger.info("本地知识库导入完成!")


def _generate_sample_docs() -> List[Dict]:
    """生成示例医疗知识文档 (确保可运行)"""
    samples = [
        {
            "content": (
                "患者：医生，我最近总是头疼，有时候还恶心想吐。\n"
                "医生：头疼持续多久了？有没有其他症状？\n"
                "患者：大概有一周了，早上起床的时候特别明显。\n"
                "医生：根据您的描述，可能是紧张性头痛，"
                "建议您注意休息，避免长时间看手机，"
                "如果持续不好转建议到神经内科就诊。\n"
                "注意事项：以上为常见头痛症状分析，"
                "不能替代专业医疗诊断，请及时就医。"
            ),
            "metadata": {
                "doc_id": "sample_headache",
                "title": "头痛常见问诊",
                "department": "神经内科",
                "publish_time": "2024-01",
                "source_type": "medical_kb",
            },
        },
        {
            "content": (
                "患者：感冒了吃什么药好得快？\n"
                "医生：普通感冒是自限性疾病，通常 7-10 天自愈。\n"
                "症状轻微的话，多喝水、多休息就可以。\n"
                "如果发热超过 38.5℃，可以服用对乙酰氨基酚退热；\n"
                "如果咳嗽严重，可以服用止咳糖浆缓解症状。\n"
                "注意：不要多种感冒药混用，避免药物过量！\n"
                "如果持续发热超过 3 天，请及时就医。\n"
                "来源：中国成人普通感冒诊治指南"
            ),
            "metadata": {
                "doc_id": "sample_cold",
                "title": "感冒常见问诊",
                "department": "呼吸内科",
                "publish_time": "2024-03",
                "source_type": "medical_kb",
            },
        },
        {
            "content": (
                "主诉：反复上腹痛 2 个月，饭后加重\n"
                "现病史：患者近 2 个月反复出现上腹部隐痛，"
                "饭后 1-2 小时加重，伴有嗳气、反酸、烧心。\n"
                "诊断：慢性胃炎\n"
                "用药指导：奥美拉唑 20mg 每日一次，空腹服用；"
                "铝碳酸镁咀嚼片，餐后 1 小时嚼服，每次 2 片。\n"
                "注意事项：忌烟酒、辛辣食物，规律饮食，不要空腹喝咖啡。\n"
                "随访建议：服药 2 周后复诊，评估症状改善情况。\n"
                "来源：中国慢性胃炎诊治指南"
            ),
            "metadata": {
                "doc_id": "sample_gastritis",
                "title": "慢性胃炎问诊",
                "department": "消化内科",
                "publish_time": "2024-06",
                "source_type": "medical_kb",
            },
        },
    ]
    return samples


if __name__ == "__main__":
    main()
