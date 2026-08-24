# ============================================================
# Milvus 双源并行检索模块
# 来源：项目需求
# - BGE-M3 将 query 编码为 1024 维向量
# - 并行检索 medical_kb + pubmed_literature
# - 返回带元数据的 chunk 列表
# ============================================================

import time
import asyncio
from typing import List, Dict, Optional, Tuple
from loguru import logger
from pymilvus import Collection, connections, utility
from config.settings import settings


class DualSourceRetriever:
    """
    双源检索器 (项目需求)
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

    # ========== 连接管理 (项目需求) ==========

    def connect(self):
        """连接 Milvus ( 单机版)"""
        if self._connected:
            return
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
            )
            self._local_col = Collection(settings.milvus_collection_kb)
            self._local_col.load()
            self._pubmed_col = Collection(settings.milvus_collection_pubmed)
            self._pubmed_col.load()
            self._connected = True
            logger.info(
                f"Milvus连接成功: {settings.milvus_host}:{settings.milvus_port}"
            )
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise

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
        """将 query 编码为 1024 维向量 (项目需求)"""
        embedding = self.embedder.encode(
            query, normalize_embeddings=True
        )
        return embedding.tolist()

    # ========== 检索 (项目需求/ 真正并行检索) ==========

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
        """执行 Milvus 向量检索 (项目需求)"""
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

    # ========== 降级检索 ( 失败时用通用知识) ==========

    def degrade_search(self, query: str) -> List[Dict]:
        """
        降级检索: 工具调用失败时返回空结果
        触发大模型用通用知识回答 (项目需求)
        """
        logger.warning(f"检索降级: 返回空结果, 将使用通用知识回答 | query={query[:60]}")
        return []

    # ========== 关键词增强检索 ( 失败重试时调整关键词) ==========

    def expand_query(self, query: str) -> str:
        """
        检索失败后的查询扩展 ( 自动调整检索关键词)
        简单实现: 去掉部分停用词, 提取核心实体
        """
        stopwords = {"的", "了", "是", "我", "你", "吗", "呢", "啊", "吧", "什么", "怎么"}
        words = [w for w in query if w not in stopwords]
        return "".join(words) if words else query


# 全局单例
retriever = DualSourceRetriever()
