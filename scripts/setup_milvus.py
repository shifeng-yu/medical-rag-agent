# ============================================================
# Milvus Collection 初始化脚本
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)
from loguru import logger
from config.settings import settings


def create_collections():
    """创建两个 Milvus Collection """

    # 连接 Milvus
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )
    logger.info(f"已连接 Milvus: {settings.milvus_host}:{settings.milvus_port}")

    # 通用字段定义 ( 1024维, 元数据存储)
    common_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="publish_time", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
    ]

    # 创建本地医疗知识库 Collection
    collection_kb_name = settings.milvus_collection_kb
    if utility.has_collection(collection_kb_name):
        logger.warning(f"Collection 已存在，将删除重建: {collection_kb_name}")
        utility.drop_collection(collection_kb_name)

    schema_kb = CollectionSchema(
        fields=common_fields,
        description="本地权威医疗知识库 ",
    )
    collection_kb = Collection(name=collection_kb_name, schema=schema_kb)

    # 向量索引 (IP内积)
    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection_kb.create_index(
        field_name="embedding", index_params=index_params
    )
    collection_kb.load()
    logger.info(f"Collection 创建完成: {collection_kb_name}")

    # 创建 PubMed 离线文献库 Collection
    collection_pubmed_name = settings.milvus_collection_pubmed
    if utility.has_collection(collection_pubmed_name):
        logger.warning(f"Collection 已存在，将删除重建: {collection_pubmed_name}")
        utility.drop_collection(collection_pubmed_name)

    schema_pubmed = CollectionSchema(
        fields=common_fields,
        description="PubMed 离线文献库 ",
    )
    collection_pubmed = Collection(name=collection_pubmed_name, schema=schema_pubmed)
    collection_pubmed.create_index(
        field_name="embedding", index_params=index_params
    )
    collection_pubmed.load()
    logger.info(f"Collection 创建完成: {collection_pubmed_name}")

    # 验证
    logger.info(f"现有 Collections: {utility.list_collections()}")
    logger.info(f"  {collection_kb_name}: {collection_kb.num_entities} 条")
    logger.info(f"  {collection_pubmed_name}: {collection_pubmed.num_entities} 条")


if __name__ == "__main__":
    create_collections()
