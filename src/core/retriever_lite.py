# ============================================================
# Milvus Lite 嵌入式检索器 (CPU适配)
# 完全复现 retriever.py 的接口，底层用 milvus-lite
# 无需Docker，数据存本地文件
# ============================================================

import time
import asyncio
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger
from milvus_lite import MilvusLite, CollectionSchema, FieldSchema, DataType
from config.settings import settings


class LiteRetriever:
    """Milvus Lite 嵌入式双源检索器"""

    def __init__(self):
        self._embedder = None
        self._db: Optional[MilvusLite] = None
        self._db_path = str(Path(settings.log_dir).parent / "milvus_lite")
        self._collections_ready = False
        Path(self._db_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Milvus Lite 检索器 (db={self._db_path})")

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载 BGE-M3: {settings.bge_model_path}")
            self._embedder = SentenceTransformer(
                settings.bge_model_path, device=settings.device
            )
        return self._embedder

    def connect(self):
        if self._db is not None:
            return
        self._db = MilvusLite(self._db_path)
        dim = settings.embedding_dim

        # Schema: id (auto) + vector (1024d) + metadata fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="publish_time", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)

        for col_name in [settings.milvus_collection_kb, settings.milvus_collection_pubmed]:
            if not self._db.has_collection(col_name):
                col = self._db.create_collection(name=col_name, schema=schema)
                # Create IVF_FLAT index on vector field
                col.create_index(field_name="vector", index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                })
                logger.info(f"Collection created: {col_name}")

        self._collections_ready = True
        logger.info("Milvus Lite ready")

    def encode_query(self, query: str) -> List[float]:
        emb = self.embedder.encode(query, normalize_embeddings=True)
        return emb.tolist()

    # ========== 检索 ==========

    async def retrieve(
        self, query: str, classification: str = "both", top_k: int = None
    ) -> Tuple[List[Dict], List[Dict], float]:
        if top_k is None:
            top_k = settings.retrieval_top_k
        if not self._collections_ready:
            self.connect()

        query_vector = self.encode_query(query)
        start = time.perf_counter()

        if classification == "local":
            local_results = self._search(settings.milvus_collection_kb, query_vector, top_k, "local_kb")
            pubmed_results = []
        elif classification == "pubmed":
            local_results = []
            pubmed_results = self._search(settings.milvus_collection_pubmed, query_vector, top_k, "pubmed")
        else:
            local_results = self._search(settings.milvus_collection_kb, query_vector, top_k, "local_kb")
            pubmed_results = self._search(settings.milvus_collection_pubmed, query_vector, top_k, "pubmed")

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(f"检索: local={len(local_results)}, pubmed={len(pubmed_results)}, {latency_ms:.0f}ms")
        return local_results, pubmed_results, latency_ms

    def _search(self, col_name: str, vector: List[float], top_k: int, source: str) -> List[Dict]:
        try:
            col = self._db.get_collection(col_name)

            results = col.search(
                query_vectors=[vector],
                top_k=top_k,
                anns_field="vector",
                output_fields=["content", "department", "publish_time",
                               "title", "source_type", "doc_id"],
            )

            formatted = []
            if results:
                for hit in results[0]:  # results[0] = hits for first query vector
                    # hit is a dict with 'id', 'distance', 'entity' keys
                    entity = hit.get("entity", {})
                    formatted.append({
                        "content": entity.get("content", ""),
                        "score": float(hit.get("distance", 0)),
                        "source": source,
                        "department": entity.get("department", ""),
                        "publish_time": entity.get("publish_time", ""),
                        "title": entity.get("title", ""),
                        "doc_id": entity.get("doc_id", ""),
                    })
            return formatted
        except Exception as e:
            logger.error(f"检索失败 [{col_name}]: {e}")
            return []

    # ========== 数据写入 ==========

    def insert(self, col_name: str, chunks: List[Dict], batch_size: int = 50):
        if not self._collections_ready:
            self.connect()

        col = self._db.get_collection(col_name)
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedder.encode(
                texts, normalize_embeddings=True
            ).tolist()

            data = []
            for j, c in enumerate(batch):
                data.append({
                    "vector": embeddings[j],
                    "content": c["text"][:65535],
                    "title": c.get("metadata", {}).get("title", "")[:512],
                    "department": c.get("metadata", {}).get("department", "")[:64],
                    "publish_time": c.get("metadata", {}).get("publish_time", "")[:32],
                    "source_type": c.get("metadata", {}).get("source_type", "")[:32],
                    "doc_id": c.get("metadata", {}).get("doc_id", "")[:128],
                })

            col.insert(data)
            logger.info(f"写入 [{col_name}]: {min(i + batch_size, total)}/{total}")

        try:
            col.flush()
        except Exception:
            pass  # Windows 文件锁冲突，数据已经在内存中
        logger.info(f"写入完成 [{col_name}]: {total} 条")

    def expand_query(self, query: str) -> str:
        stopwords = {"的", "了", "是", "我", "你", "吗", "呢", "啊", "吧", "什么", "怎么"}
        words = [w for w in query if w not in stopwords]
        return "".join(words) if words else query


lite_retriever = LiteRetriever()
