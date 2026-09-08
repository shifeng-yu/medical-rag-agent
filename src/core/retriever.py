# ============================================================
# Milvus 双源并行检索模块
# - BGE-M3 将 query 编码为 1024 维向量
# - 并行检索 medical_kb + pubmed_literature
# - 返回带元数据的 chunk 列表
# ============================================================

import time
import asyncio
from typing import List, Dict, Optional, Tuple
from loguru import logger
from pymilvus import (
    Collection, CollectionSchema, FieldSchema, DataType, connections, utility,
)
from config.settings import settings
from src.core.retrieval import expand_query as _expand_query


class DualSourceRetriever:
    """
    双源检索器 
    - 两个独立 Milvus Collection
    - BGE-M3 向量化 (1024维)
    - 并行检索 (asyncio.gather)
    """

    def __init__(self):
        self._embedder = None
        self._local_col: Optional[Collection] = None
        self._pubmed_col: Optional[Collection] = None
        self._connected = False
        logger.info("双源检索器已初始化")

    # ========== 连接管理  ==========

    def connect(self):
        """连接 Milvus ( 单机版)；collection 不存在时自动创建（非破坏）"""
        if self._connected:
            return
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
            )
            self.ensure_collections(force=False)
            self._connected = True
            logger.info(
                f"Milvus连接成功: {settings.milvus_host}:{settings.milvus_port}"
            )
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise

    def ensure_collections(self, force: bool = False):
        """确保两个 Collection 存在（缺则建）；force=True 时删除重建（管理命令用）。

        建表逻辑收敛在适配器内部——setup_milvus.py 不再各自维护一套 schema。
        """
        for name in (settings.milvus_collection_kb, settings.milvus_collection_pubmed):
            exists = utility.has_collection(name)
            if force and exists:
                logger.warning(f"Collection 强制重建: {name}")
                utility.drop_collection(name)
                exists = False
            if not exists:
                self._create_collection(name)

            col = Collection(name)
            col.load()
            if name == settings.milvus_collection_kb:
                self._local_col = col
            else:
                self._pubmed_col = col

    def _create_collection(self, name: str):
        """按统一 schema + IP 指标创建 collection（与 retriever_lite 对齐）。"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="publish_time", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        ]
        schema = CollectionSchema(fields=fields, description=f"medical-rag collection: {name}")
        col = Collection(name=name, schema=schema)
        col.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            },
        )
        logger.info(f"Collection 创建完成: {name} (IP / dim={settings.embedding_dim})")

    # ========== Embedding ( BGE-M3, 1024维) ==========

    @property
    def embedder(self):
        """懒加载 BGE-M3"""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"加载 BGE-M3: {settings.bge_model_path}")
                self._embedder = SentenceTransformer(
                    settings.bge_model_path,
                    device=settings.device,
                )
            except Exception as e:
                logger.error(f"BGE-M3 加载失败: {e}")
                raise
        return self._embedder

    def encode_query(self, query: str) -> List[float]:
        """将 query 编码为 1024 维向量 """
        embedding = self.embedder.encode(
            query, normalize_embeddings=True
        )
        return embedding.tolist()

    # ========== 检索 (真正并行检索) ==========

    async def retrieve(
        self,
        query: str,
        classification: str = "both",
        top_k: int = None,
    ) -> Tuple[List[Dict], List[Dict], float]:
        """
        双源并行检索 ( 改串行为并行)
        参数:
            query: 用户问题
            classification: 分类结果 (local/pubmed/both)
            top_k: 每库检索数量
        返回: (local_results, pubmed_results, latency_ms)
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        if not self._connected:
            self.connect()

        query_vector = self.encode_query(query)
        start = time.perf_counter()

        if classification == "local":
            # 仅本地库 ( 常见病快速匹配)
            local_results = await self._search_local(query_vector, top_k)
            pubmed_results = []
        elif classification == "pubmed":
            # 仅文献库 ( 罕见病/前沿)
            local_results = []
            pubmed_results = await self._search_pubmed(query_vector, top_k)
        else:
            # 并行双库 ( 真正并行)
            local_results, pubmed_results = await asyncio.gather(
                self._search_local(query_vector, top_k),
                self._search_pubmed(query_vector, top_k),
            )

        latency_ms = (time.perf_counter() - start) * 1000
        total = len(local_results) + len(pubmed_results)
        logger.info(
            f"检索完成: local={len(local_results)}, pubmed={len(pubmed_results)}, "
            f"总耗时={latency_ms:.0f}ms"
        )
        return local_results, pubmed_results, latency_ms

    async def _search_local(
        self, query_vector: List[float], top_k: int
    ) -> List[Dict]:
        """检索本地医疗知识库集合"""
        return await self._milvus_search(
            self._local_col, query_vector, top_k, source="local_kb"
        )

    async def _search_pubmed(
        self, query_vector: List[float], top_k: int
    ) -> List[Dict]:
        """检索 PubMed 离线文献库集合"""
        return await self._milvus_search(
            self._pubmed_col, query_vector, top_k, source="pubmed"
        )

    async def _milvus_search(
        self,
        collection: Collection,
        query_vector: List[float],
        top_k: int,
        source: str,
    ) -> List[Dict]:
        """执行 Milvus 向量检索 """
        if collection is None:
            return []

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.search(
                    data=[query_vector],
                    anns_field="embedding",  # 向量字段
                    param={"metric_type": "IP", "params": {"nprobe": 16}},
                    limit=top_k,
                    output_fields=[
                        "content", "department", "publish_time",
                        "title", "source_type", "doc_id",
                    ],
                ),
            )

            # 格式化结果
            formatted = []
            if results and results[0]:
                for hit in results[0]:
                    entity = hit.entity
                    formatted.append({
                        "content": getattr(entity, "content", ""),
                        "score": float(hit.score),
                        "source": source,
                        "department": getattr(entity, "department", ""),
                        "publish_time": getattr(entity, "publish_time", ""),
                        "title": getattr(entity, "title", ""),
                        "doc_id": getattr(entity, "doc_id", ""),
                    })
            return formatted

        except Exception as e:
            logger.error(f"Milvus检索错误 [{source}]: {e}")
            return []

    # ========== 数据写入（与 retriever_lite.insert 同一 chunk 契约） ==========

    def insert(
        self, collection_name: str, chunks: List[Dict], batch_size: int = 100
    ):
        """将分块向量化并写入指定 collection。

        chunk 格式: {"text": str, "metadata": {title, department,
        publish_time, source_type, doc_id}} —— 与 retriever_lite.insert 完全一致。
        """
        if not self._connected:
            self.connect()

        col = Collection(collection_name)
        col.load()
        total = len(chunks)
        inserted = 0
        start = time.perf_counter()

        for i in range(0, total, batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.encode_query_batch(texts)

            data = [
                embeddings,
                texts,
                [c["metadata"].get("title", "") for c in batch],
                [c["metadata"].get("department", "") for c in batch],
                [c["metadata"].get("publish_time", "") for c in batch],
                [c["metadata"].get("source_type", "medical_kb") for c in batch],
                [c["metadata"].get("doc_id", "") for c in batch],
            ]
            col.insert(data)
            inserted += len(batch)
            logger.info(f"导入进度: {inserted}/{total} → {collection_name}")

        col.flush()
        latency = time.perf_counter() - start
        logger.info(
            f"导入完成: {inserted} 条 → {collection_name}, 耗时={latency:.1f}s"
        )

    def encode_query_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码（灌库用，与 encode_query 共用 embedder）"""
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    # ========== 数据删除（按 doc_id，供上传替换/删除接口用） ==========

    def delete_by_doc_id(
        self,
        collection_name: str,
        doc_id: str,
        department: Optional[str] = None,
        timeout: Optional[float] = 30,
    ) -> int:
        """删除指定 collection 中某个 doc_id 的向量（可加 department 过滤）。

        doc_id = 文件名去后缀（与 ingest_kb.py 同规则，保证同树重灌不重复）。
        同名文档再上传 = 替换语义：先 delete_by_doc_id 清旧，再 insert 新。
        跨科室同名文件存在时传 department 精确删；缺省则按 doc_id 全局删。
        返回删除条数（不存在时返回 0，不报错）。
        """
        if not self._connected:
            self.connect()
        col = Collection(collection_name)
        col.load()
        # doc_id / department 是 VARCHAR 字段；转义双引号防注入/语法错
        safe_doc = doc_id.replace("\\", "\\\\").replace('"', '\\"')
        expr = f'doc_id == "{safe_doc}"'
        if department:
            safe_dept = department.replace("\\", "\\\\").replace('"', '\\"')
            expr += f' and department == "{safe_dept}"'
        result = col.delete(expr=expr, timeout=timeout)
        col.flush()
        deleted = result.delete_count if hasattr(result, "delete_count") else 0
        logger.info(f"delete_by_doc_id: {doc_id} → {collection_name} 删 {deleted} 条")
        return deleted

    def expand_query(self, query: str) -> str:
        """检索失败后的查询扩展 ( 自动调整检索关键词)"""
        return _expand_query(query)


# 全局单例
retriever = DualSourceRetriever()
