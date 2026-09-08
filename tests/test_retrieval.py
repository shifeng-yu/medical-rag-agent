# ============================================================
# 检索接缝契约测试
# 覆盖 (候选 A · 检索接缝归位):
#   1. get_retriever() 唯一选择逻辑
#   2. 两个适配器（Docker Milvus / Milvus Lite）接口一致：方法齐全、签名一致
#   3. expand_query 单一实现（两适配器同源）
#   4. Lite 真往返: insert → retrieve（临时库 + 假编码器，IP 打分方向实证）
# ============================================================

import sys
import asyncio
import inspect
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from config.settings import settings


# ---------------- 1. 唯一选择逻辑 ----------------

class TestGetRetriever:
    def test_returns_cached_singleton(self):
        from src.core import retrieval
        r1 = retrieval.get_retriever()
        r2 = retrieval.get_retriever()
        assert r1 is r2  # 缓存单例

    def test_lite_mode_selects_lite(self, monkeypatch):
        from src.core import retrieval
        from src.core.retriever_lite import LiteRetriever
        monkeypatch.setattr(retrieval, "_retriever", None)
        monkeypatch.setattr(settings, "use_milvus_lite", True)
        assert isinstance(retrieval.get_retriever(), LiteRetriever)

    def test_docker_mode_selects_pymilvus(self, monkeypatch):
        pymilvus = pytest.importorskip("pymilvus")
        from src.core import retrieval
        from src.core.retriever import DualSourceRetriever
        monkeypatch.setattr(retrieval, "_retriever", None)
        monkeypatch.setattr(settings, "use_milvus_lite", False)
        assert isinstance(retrieval.get_retriever(), DualSourceRetriever)


# ---------------- 2. 适配器契约一致 ----------------

def _adapter_classes():
    from src.core.retriever import DualSourceRetriever
    from src.core.retriever_lite import LiteRetriever
    return [DualSourceRetriever, LiteRetriever]


REQUIRED_MEMBERS = {
    "connect", "retrieve", "insert", "expand_query",
    "encode_query", "embedder",  # embedder 是 property，类级存在即可
}


class TestAdapterParity:
    @pytest.fixture(params=[False, True], ids=["docker", "lite"])
    def adapter_class(self, request):
        if request.param:
            pytest.importorskip("milvus_lite")
        else:
            pytest.importorskip("pymilvus")
        classes = _adapter_classes()
        return classes[1] if request.param else classes[0]

    def test_required_members_present(self, adapter_class):
        missing = REQUIRED_MEMBERS - {m for m in dir(adapter_class) if not m.startswith("__")}
        assert not missing, f"{adapter_class.__name__} 缺少契约成员: {missing}"

    def test_retrieve_signature_aligned(self, adapter_class):
        params = list(inspect.signature(adapter_class.retrieve).parameters)
        assert params[:3] == ["self", "query", "classification"], params
        assert "top_k" in params

    def test_insert_signature_aligned(self, adapter_class):
        sig = inspect.signature(adapter_class.insert)
        params = list(sig.parameters)
        assert params[:3] == ["self", "collection_name", "chunks"], params
        assert sig.parameters["batch_size"].default == 100  # 两适配器统一

    def test_expand_query_shared_source(self):
        """两边 expand_query 行为一致（单一实现）"""
        from src.core.retriever import DualSourceRetriever
        from src.core.retriever_lite import LiteRetriever
        docker, lite = DualSourceRetriever(), LiteRetriever()
        for q in ["最近总是头痛怎么办", "我该吃什么药", "的了我你吗"]:
            assert docker.expand_query(q) == lite.expand_query(q)

    def test_expand_query_removes_stopwords_only(self):
        from src.core.retriever import DualSourceRetriever
        r = DualSourceRetriever()
        assert r.expand_query("我头痛的厉害") == "头痛厉害"
        assert r.expand_query("的我你") == "的我你"  # 全停用词 → 原样返回


# ---------------- 3. Lite 真往返（IP 打分方向实证） ----------------

class _FakeVec:
    """ndarray 的最小替身：只有真实调用链需要的 .tolist()"""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeEmbedder:
    """确定性假编码器：含'苹果'→第0维=1，含'香蕉'→第1维=1，其余为 0"""

    def __init__(self, dim: int):
        self._dim = dim

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = [[0.0] * self._dim for _ in items]
        for v, t in zip(vecs, items):
            if "苹果" in t:
                v[0] = 1.0
            if "香蕉" in t:
                v[1] = 1.0
        return _FakeVec(vecs[0]) if single else _FakeVec(vecs)


def _chunk(text, doc_id, source_type="local_kb"):
    return {
        "text": text,
        "metadata": {
            "title": doc_id, "department": "心内科",
            "publish_time": "2024", "source_type": source_type, "doc_id": doc_id,
        },
    }


class TestLiteRoundtrip:
    @pytest.fixture()
    def lite_roundtrip(self, monkeypatch):
        pytest.importorskip("milvus_lite")
        from src.core.retriever_lite import LiteRetriever

        retriever = LiteRetriever()
        tmpdir = tempfile.mkdtemp(prefix="medrag_retrieval_test_")
        monkeypatch.setattr(retriever, "_db_path", tmpdir)
        # 与 test_documents_api 同款隔离：清掉可能已绑真实库的句柄，强制按 tmp 重连
        monkeypatch.setattr(retriever, "_db", None)
        retriever._collections_ready = False
        retriever._embedder = _FakeEmbedder(settings.embedding_dim)
        return retriever

    def test_insert_then_retrieve_ranks_by_ip(self, lite_roundtrip):
        retriever = lite_roundtrip

        retriever.insert(settings.milvus_collection_kb, [
            _chunk("苹果富含维生素C", "apple_doc"),
            _chunk("香蕉富含钾元素", "banana_doc"),
        ])

        local, pubmed, latency = asyncio.run(
            retriever.retrieve("苹果怎么吃", classification="both", top_k=5)
        )
        assert latency >= 0
        assert len(local) == 2
        assert pubmed == []

        # IP 打分方向: 与"苹果"最相关的排最前，score 越大越相关
        assert local[0]["doc_id"] == "apple_doc"
        assert local[0]["score"] >= local[1]["score"]
        assert local[0]["score"] > 0
        # 结果字段契约
        assert set(local[0].keys()) == {
            "content", "score", "source", "department",
            "publish_time", "title", "doc_id",
        }

    def test_insert_both_collections_and_parallel_shape(self, lite_roundtrip):
        retriever = lite_roundtrip
        retriever.insert(settings.milvus_collection_kb, [_chunk("苹果的吃法", "a")])
        retriever.insert(settings.milvus_collection_pubmed, [
            _chunk("香蕉的文献研究", "p1", source_type="pubmed"),
        ])

        local, pubmed, _ = asyncio.run(
            retriever.retrieve("苹果", classification="both", top_k=3)
        )
        assert len(local) == 1 and len(pubmed) == 1
        assert pubmed[0]["source"] == "pubmed"
        assert local[0]["source"] == "local_kb"
