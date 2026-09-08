# ============================================================
# Milvus Lite 嵌入式检索器 (CPU适配)
# 完全复现 retriever.py 的接口，底层用 milvus-lite
# 无需Docker，数据存本地文件
# ============================================================

import time
import asyncio
import json
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger
from config.settings import settings
from src.core.retrieval import expand_query as _expand_query


class LiteRetriever:
    """Milvus Lite 嵌入式双源检索器"""

    def __init__(self):
        self._embedder = None
        self._db = None  # MilvusLite 实例，惰性导入（仅在 connect 时加载）
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
        # 惰性导入: milvus_lite 仅在真正需要嵌入式向量库时加载，
        # 避免在无该依赖的环境（如 CI 测试）模块导入即失败。
        from milvus_lite import MilvusLite, CollectionSchema, FieldSchema, DataType
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
                # 与 Docker 版对齐: 统一 IP 指标（向量已归一化 → score≈余弦相似度）
                col.create_index(field_name="vector", index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 128},
                })
                logger.info(f"Collection created: {col_name} (IP)")
            # 已存在的集合在 milvus-lite 3.x 打开时默认 released，
            # query/search/delete 前必须先 load，否则抛 CollectionNotLoadedError。
            self._ensure_loaded(col_name)

        self._collections_ready = True
        logger.info("Milvus Lite ready")

    def _ensure_loaded(self, col_name: str):
        """取集合并确保处于 loaded 态（load 幂等，重复调用无害）。

        milvus-lite 3.x 对已存在的 collection 默认 released：
        不 load 直接 query/search/delete 会抛 CollectionNotLoadedError，
        造成"上传成功但检索为空 / 删除失效"的静默故障。
        """
        col = self._db.get_collection(col_name)
        if getattr(col, "load_state", "loaded") != "loaded":
            col.load()
        return col

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
            col = self._ensure_loaded(col_name)

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

    def insert(
        self, collection_name: str, chunks: List[Dict], batch_size: int = 100
    ):
        """将分块向量化并写入指定 collection（与 retriever.insert 同一 chunk 契约）"""
        if not self._collections_ready:
            self.connect()

        col = self._db.get_collection(collection_name)
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
            logger.info(f"写入 [{collection_name}]: {min(i + batch_size, total)}/{total}")

        # Windows 持久化: milvus-lite 的 flush 在 Windows 存在 rename bug
        # （os.rename 覆盖已存在文件 → FileExistsError）。实测发现失败一次
        # flush 会把数据搞到"查不到"（索引段损坏），因此 Windows 直接跳过
        # flush——数据驻留内存、可正常检索，仅进程重启后丢失（本地开发可接受）。
        # Linux/CI 正常 flush 持久化。
        if os.name != "nt":
            try:
                col.flush()
            except Exception as e:
                logger.warning(f"flush 失败（数据已在内存中）: {e}")
        logger.info(f"写入完成 [{collection_name}]: {total} 条")

    def expand_query(self, query: str) -> str:
        """检索失败后的查询扩展（共享实现，见 retrieval.expand_query）"""
        return _expand_query(query)

    # ========== 数据删除（按 doc_id，供上传替换/删除接口用） ==========

    def delete_by_doc_id(
        self,
        collection_name: str,
        doc_id: str,
        department: Optional[str] = None,
    ) -> int:
        """删除指定 collection 中某个 doc_id 的向量（可加 department 过滤）。

        milvus-lite 的 Collection.delete 只收主键列表（无 expr），因此
        分两步：query 按 doc_id 查出主键 → delete(pks)。doc_id 与
        retriever.delete_by_doc_id 同一语义（文件名去后缀）。返回删除条数；
        doc_id 不存在时返回 0。
        """
        if not self._collections_ready:
            self.connect()
        col = self._ensure_loaded(collection_name)
        # 主键 auto_id=INT64，doc_id 是 VARCHAR 标量字段
        safe_doc = doc_id.replace("\\", "\\\\").replace('"', '\\"')
        expr = f'doc_id == "{safe_doc}"'
        if department:
            safe_dept = department.replace("\\", "\\\\").replace('"', '\\"')
            expr += f' and department == "{safe_dept}"'
        rows = col.query(expr=expr, output_fields=["id", "doc_id", "department"])
        if department:
            rows = [r for r in rows if r.get("department") == department]
        pks = [r["id"] for r in rows if r.get("doc_id") == doc_id]
        deleted = 0
        if pks:
            deleted = col.delete(pks)
        logger.info(f"delete_by_doc_id: {doc_id} → {collection_name} 删 {deleted} 条")
        return deleted


lite_retriever = LiteRetriever()
