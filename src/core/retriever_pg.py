# ============================================================
# pgvector 检索适配器（第三个后端）
#
# 与 retriever.py（Docker Milvus）、retriever_lite.py（Milvus Lite）
# 同契约：connect / retrieve / insert / delete_by_doc_id /
#         expand_query / encode_query / embedder。
# 由 retrieval.get_retriever() 按 VECTOR_BACKEND 选择，调用方不写分支。
#
# 为什么加它：目标口径是「换库无需改动代码、配置即可生效」。前两个适配器
# 都是 Milvus，只能证明"换部署形态"；这里落一个真异构后端（PostgreSQL +
# pgvector），才能证明"换引擎"（见 docs/adr/0009-third-vector-backend.md）。
#
# 表结构：单表 + collection 列（不是一集合一张表）——pgvector 的 HNSW 索引
# 对过滤列友好，单表能少一次建表/迁移，也便于跨库统计。
# 打分：1 - cosine_distance，与 Milvus 的 IP（向量已归一化）同向同量纲。
# ============================================================

import asyncio
import re
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from loguru import logger

from config.settings import settings
from src.core.retrieval import expand_query as _expand_query

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PgVectorRetriever:
    """PostgreSQL + pgvector 双源检索器。"""

    def __init__(self):
        self._pool = None
        self._embedder = None
        self._ready = False
        self._table = self._safe_ident(settings.pg_table)
        logger.info(f"pgvector 检索器 (dsn={self._mask(settings.pg_dsn)}, table={self._table})")

    # ---------- 基础 ----------

    @staticmethod
    def _mask(dsn: str) -> str:
        """日志里不打印口令。"""
        return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", dsn or "")

    @staticmethod
    def _safe_ident(name: str) -> str:
        """表名来自配置，必须校验成合法标识符（不能参数化）。"""
        if not _IDENT.match(name or ""):
            raise ValueError(f"非法表名: {name!r}（仅允许字母/数字/下划线）")
        return name

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"加载 BGE-M3: {settings.bge_model_path}")
            self._embedder = SentenceTransformer(
                settings.bge_model_path, device=settings.device
            )
        return self._embedder

    def _vec(self, values: List[float]) -> str:
        """Python list → pgvector 文本格式（'[0.1,0.2,...]'），配合 %s::vector。"""
        return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"

    # ---------- 连接与建表 ----------

    def connect(self):
        if self._pool is not None:
            return
        from psycopg2 import pool as pg_pool

        logger.info(f"连接 pgvector: {self._mask(settings.pg_dsn)}")
        self._pool = pg_pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=settings.pg_pool_max,
            dsn=settings.pg_dsn,
            connect_timeout=5,
        )
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    # 扩展需要超级用户或已预装；失败时给出可操作的报错
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cur.execute(self._ddl())
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS ix_{self._table}_collection_doc "
                        f"ON {self._table} (collection, doc_id)"
                    )
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS ix_{self._table}_embedding "
                        f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
                    )
        except Exception:
            self._pool.closeall()
            self._pool = None
            raise
        self._ready = True
        logger.info(f"pgvector ready (table={self._table}, dim={settings.embedding_dim})")

    def _ddl(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {self._table} (
            id            BIGSERIAL PRIMARY KEY,
            collection    VARCHAR(64)  NOT NULL,
            content       TEXT         NOT NULL,
            title         VARCHAR(512) NOT NULL DEFAULT '',
            department    VARCHAR(64)  NOT NULL DEFAULT '',
            publish_time  VARCHAR(32)  NOT NULL DEFAULT '',
            source_type   VARCHAR(32)  NOT NULL DEFAULT '',
            doc_id        VARCHAR(128) NOT NULL DEFAULT '',
            embedding     vector({settings.embedding_dim}) NOT NULL
        )
        """

    @contextmanager
    def _conn(self):
        if self._pool is None:
            raise RuntimeError("pgvector 未连接：先调用 connect()")
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close(self):
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            self._ready = False

    # ---------- 编码与检索 ----------

    def encode_query(self, query: str) -> List[float]:
        return self.embedder.encode(query, normalize_embeddings=True).tolist()

    async def retrieve(
        self, query: str, classification: str = "both", top_k: int = None
    ) -> Tuple[List[Dict], List[Dict], float]:
        if top_k is None:
            top_k = settings.retrieval_top_k
        if not self._ready:
            self.connect()

        vector = self.encode_query(query)
        start = time.perf_counter()
        kb, pubmed = settings.milvus_collection_kb, settings.milvus_collection_pubmed

        if classification == "local":
            local_results = await asyncio.to_thread(
                self._search, kb, vector, top_k, "local_kb"
            )
            pubmed_results: List[Dict] = []
        elif classification == "pubmed":
            local_results = []
            pubmed_results = await asyncio.to_thread(
                self._search, pubmed, vector, top_k, "pubmed"
            )
        else:
            # 双源并行：两条 SQL 互不依赖，串行会白白多花一倍时间
            local_results, pubmed_results = await asyncio.gather(
                asyncio.to_thread(self._search, kb, vector, top_k, "local_kb"),
                asyncio.to_thread(self._search, pubmed, vector, top_k, "pubmed"),
            )

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"检索(pg): local={len(local_results)}, pubmed={len(pubmed_results)}, {latency_ms:.0f}ms"
        )
        return local_results, pubmed_results, latency_ms

    def _search(
        self, collection: str, vector: List[float], top_k: int, source: str
    ) -> List[Dict]:
        literal = self._vec(vector)
        sql = f"""
            SELECT content, title, department, publish_time, doc_id,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self._table}
            WHERE collection = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (literal, collection, literal, top_k))
                    rows = cur.fetchall()
        except Exception as e:  # noqa: BLE001 - 单库检索失败不该拖垮整个问诊
            logger.error(f"pgvector 检索失败 [{collection}]: {e}")
            return []

        return [
            {
                "content": row[0] or "",
                "title": row[1] or "",
                "department": row[2] or "",
                "publish_time": row[3] or "",
                "doc_id": row[4] or "",
                # 余弦距离浮点误差可能让 1-d 略微越界，夹回 [0,1]
                "score": max(0.0, min(1.0, float(row[5] or 0))),
                "source": source,
            }
            for row in rows
        ]

    # ---------- 写入与删除 ----------

    def insert(
        self, collection_name: str, chunks: List[Dict], batch_size: int = 100
    ):
        """向量化并写入；与 Milvus 适配器同一 chunk 契约。"""
        if not self._ready:
            self.connect()

        from psycopg2.extras import execute_values

        sql = f"""
            INSERT INTO {self._table}
                (collection, content, title, department, publish_time,
                 source_type, doc_id, embedding)
            VALUES %s
        """
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedder.encode(
                texts, normalize_embeddings=True
            ).tolist()

            rows = []
            for j, chunk in enumerate(batch):
                meta = chunk.get("metadata", {}) or {}
                rows.append((
                    collection_name,
                    chunk["text"][:65535],
                    str(meta.get("title", ""))[:512],
                    str(meta.get("department", ""))[:64],
                    str(meta.get("publish_time", ""))[:32],
                    str(meta.get("source_type", ""))[:32],
                    str(meta.get("doc_id", ""))[:128],
                    self._vec(embeddings[j]),
                ))

            with self._conn() as conn:
                with conn.cursor() as cur:
                    # 模板里对最后一列加倍率转换，避免 "vector" 类型不匹配
                    execute_values(
                        cur, sql, rows,
                        template="(%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                    )
            logger.info(f"写入 [{collection_name}]: {min(i + batch_size, total)}/{total}")

        logger.info(f"写入完成 [{collection_name}]: {total} 条")

    def delete_by_doc_id(
        self,
        collection_name: str,
        doc_id: str,
        department: Optional[str] = None,
    ) -> int:
        sql = f"DELETE FROM {self._table} WHERE collection = %s AND doc_id = %s"
        params: List = [collection_name, doc_id]
        if department:
            sql += " AND department = %s"
            params.append(department)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                deleted = cur.rowcount
        logger.info(f"delete_by_doc_id: {doc_id} → {collection_name} 删 {deleted} 条")
        return int(deleted or 0)

    def expand_query(self, query: str) -> str:
        """检索失败后的查询扩展（共享实现，见 retrieval.expand_query）。"""
        return _expand_query(query)

    # ---------- 运维辅助 ----------

    def count(self, collection_name: str) -> int:
        """当前集合的向量条数（测试与运维排查用，不在检索契约内）。"""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT count(*) FROM {self._table} WHERE collection = %s",
                    (collection_name,),
                )
                return int(cur.fetchone()[0])


pg_retriever = PgVectorRetriever()
