# ============================================================
# pgvector 适配器测试
#
# 分两层：
#   1. 后端选择纯逻辑（无需数据库，CI 必跑）
#   2. 真库往返（insert → retrieve → delete），需要真实 PostgreSQL + pgvector，
#      通过 PG_TEST_DSN 开启，未设置则跳过（CI 无数据库时不会假失败）
#
# 真库测试用假编码器：不加载 BGE-M3，只验 SQL / 打分 / 删除语义。
# ============================================================

import asyncio
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from config.settings import settings


class TestBackendSelection:
    """VECTOR_BACKEND 解析：显式优先，未配置回落旧开关。"""

    def test_explicit_pgvector(self, monkeypatch):
        from src.core.retrieval import backend_name

        monkeypatch.setattr(settings, "vector_backend", "pgvector")
        assert backend_name() == "pgvector"

    def test_explicit_lite(self, monkeypatch):
        from src.core.retrieval import backend_name

        monkeypatch.setattr(settings, "vector_backend", "milvus_lite")
        assert backend_name() == "milvus_lite"

    def test_explicit_milvus(self, monkeypatch):
        from src.core.retrieval import backend_name

        monkeypatch.setattr(settings, "vector_backend", "milvus")
        assert backend_name() == "milvus"

    def test_fallback_to_legacy_switch(self, monkeypatch):
        from src.core.retrieval import backend_name

        monkeypatch.setattr(settings, "vector_backend", "")
        monkeypatch.setattr(settings, "use_milvus_lite", False)
        assert backend_name() == "milvus"
        monkeypatch.setattr(settings, "use_milvus_lite", True)
        assert backend_name() == "milvus_lite"


# ---------------- 真库往返 ----------------

PG_DSN = os.getenv("PG_TEST_DSN", "").strip()

pytestmark_db = pytest.mark.skipif(
    not PG_DSN, reason="未设置 PG_TEST_DSN（需要真实 PostgreSQL + pgvector）"
)


class _Vec:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeEmbedder:
    """含'苹果'→dim0，含'香蕉'→dim1，其余→dim2；余弦可区分。"""

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = []
        for text in items:
            vec = [0.0] * settings.embedding_dim
            if "苹果" in text:
                vec[0] = 1.0
            if "香蕉" in text:
                vec[1] = 1.0
            if not any(vec):
                vec[2] = 1.0
            out.append(vec)
        return _Vec(out[0]) if single else _Vec(out)


@pytest.fixture(scope="module")
def pg():
    pytest.importorskip("psycopg2")
    if not PG_DSN:
        pytest.skip("未设置 PG_TEST_DSN")

    from src.core.retriever_pg import PgVectorRetriever

    original_dsn = settings.pg_dsn
    settings.pg_dsn = PG_DSN

    retriever = PgVectorRetriever()
    # 独立表：不碰开发库里真实的 kb_chunks
    retriever._table = f"kb_chunks_test_{uuid.uuid4().hex[:6]}"
    retriever._embedder = _FakeEmbedder()
    retriever.connect()

    yield retriever

    # 清表 + 删表，避免在测试库留垃圾
    try:
        with retriever._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {retriever._table}")
    finally:
        retriever.close()
        settings.pg_dsn = original_dsn


@pytestmark_db
class TestPgRoundtrip:
    COLLECTION = "medical_knowledge_base"

    def _chunk(self, text, doc_id, department="营养科"):
        return {
            "text": text,
            "metadata": {
                "title": f"{doc_id} 指南",
                "department": department,
                "doc_id": doc_id,
                "source_type": "medical_kb",
                "publish_time": "2026-01",
            },
        }

    def test_contract_members_align_with_lite(self, pg):
        """三个适配器必须同形：成员与 retrieve/insert 签名一致。"""
        import inspect

        from src.core.retriever_lite import LiteRetriever

        required = {
            "connect", "retrieve", "insert", "expand_query",
            "encode_query", "embedder", "delete_by_doc_id",
        }
        pg_members = {m for m in dir(type(pg)) if not m.startswith("__")}
        assert required <= pg_members, required - pg_members

        for name in ("retrieve", "insert"):
            pg_params = list(inspect.signature(getattr(type(pg), name)).parameters)
            lite_params = list(inspect.signature(getattr(LiteRetriever, name)).parameters)
            assert pg_params[:3] == lite_params[:3], (name, pg_params, lite_params)

    def test_insert_retrieve_delete(self, pg):
        before = pg.count(self.COLLECTION)
        pg.insert(self.COLLECTION, [
            self._chunk("苹果富含膳食纤维，有助于肠道健康。", "apple"),
            self._chunk("香蕉含钾丰富，适合补钾人群。", "banana"),
            self._chunk("冠心病患者应规律复查血脂。", "chd", department="心内科"),
        ], batch_size=2)
        assert pg.count(self.COLLECTION) == before + 3

        local, pubmed, latency = asyncio.run(pg.retrieve("苹果有哪些营养", "local"))
        assert local and local[0]["doc_id"] == "apple"
        assert local[0]["score"] > 0.99          # 余弦同向 → 接近 1
        assert local[0]["source"] == "local_kb"
        assert local[0]["title"] == "apple 指南"
        assert pubmed == []
        assert latency >= 0

        # pubmed 集合为空：返回空列表而不是抛异常
        local2, pubmed2, _ = asyncio.run(pg.retrieve("香蕉", "pubmed"))
        assert local2 == [] and pubmed2 == []

        # 打分降序
        scores = [r["score"] for r in local]
        assert scores == sorted(scores, reverse=True)

        deleted = pg.delete_by_doc_id(self.COLLECTION, "apple", department="营养科")
        assert deleted == 1
        assert pg.count(self.COLLECTION) == before + 2

        pg.delete_by_doc_id(self.COLLECTION, "banana")
        pg.delete_by_doc_id(self.COLLECTION, "chd", department="心内科")
        assert pg.count(self.COLLECTION) == before
