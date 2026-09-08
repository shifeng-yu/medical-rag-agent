# ============================================================
# 知识库文档上传 API 测试
# 覆盖:
#   1. multipart 上传 md → 后台任务 done、分块入库、检索可命中
#   2. 上传落盘到 collection 对应的 data 目录（doc_id=文件名去后缀）
#   3. 同名文件重传 = 替换（向量条数不翻倍、旧 doc_id 清空）
#   4. DELETE 删除文件 + 向量
#   5. json Q&A 结构按条切分入库
#   6. 非法: 扩展名 / department / collection(pubmed 不允许) / 大小 / 数量
# 无模型策略: 假编码器注入（与 test_retrieval 同款）+ 临时 Lite 库
# ============================================================

import sys
import time
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings


class _FakeVec:
    """ndarray 最小替身：只有调用链需要的 .tolist()"""

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


@pytest.fixture()
def doc_env(monkeypatch):
    """独立 app + 临时 Lite 库（假编码器）+ 临时知识库目录"""
    pytest.importorskip("milvus_lite")
    monkeypatch.setattr(settings, "use_milvus_lite", True)

    tmp_kb = tempfile.mkdtemp(prefix="medrag_kb_")
    tmp_pub = tempfile.mkdtemp(prefix="medrag_pub_")
    monkeypatch.setattr(settings, "medical_kb_dir", tmp_kb)
    monkeypatch.setattr(settings, "pubmed_offline_dir", tmp_pub)

    from src.core import retrieval
    monkeypatch.setattr(retrieval, "_retriever", None)
    retriever = retrieval.get_retriever()
    tmp_db = tempfile.mkdtemp(prefix="medrag_db_")
    monkeypatch.setattr(retriever, "_db_path", tmp_db)
    # 关键隔离：get_retriever 返回的是模块级单例 lite_retriever，
    # 若此前有测试（如 test_api 的 startup）已把它的 _db 绑到真实库，
    # 仅换 _db_path 不会重连——必须清掉句柄与 ready 标志，让 connect() 按新路径重建。
    monkeypatch.setattr(retriever, "_db", None)
    retriever._collections_ready = False
    retriever._embedder = _FakeEmbedder(settings.embedding_dim)

    from src.api.documents import router as documents_router
    app = FastAPI()
    app.include_router(documents_router)
    yield app, retriever, Path(tmp_kb), Path(tmp_pub)


def _upload(client, name="头痛.md", content="苹果富含维生素C，建议多食用。",
            department="神经内科", collection="local_kb", files=None):
    payload = files or [("files", (name, content.encode("utf-8"), "text/markdown"))]
    return client.post(
        "/api/v1/documents/upload",
        files=payload,
        data={"department": department, "collection": collection},
    )


def _wait_task(client, task_id, timeout=10.0):
    """轮询任务直到 done/error"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/v1/documents/tasks/{task_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"任务 {task_id} 超时未完成")


def _count_doc_vectors(retriever, col_name, doc_id):
    """按 doc_id 统计向量条数（query 过滤，tombstone 语义下可靠）

    milvus-lite 在 Windows 跳过 flush，delete 是 tombstone 语义：
    num_entities 物理计数可能滞后，用 query 按 doc_id 过滤才是真值。
    """
    col = retriever._db.get_collection(col_name)
    rows = col.query(
        expr=f'doc_id == "{doc_id}"',
        output_fields=["id"],
        limit=10000,
    )
    return len(rows)


class TestUploadDocument:
    def test_upload_then_ingest_and_retrieve(self, doc_env):
        app, retriever, tmp_kb, _ = doc_env
        client = TestClient(app)

        resp = _upload(client, name="头痛问答.md",
                       content="问：经常头痛怎么办？答：苹果富含维生素C。")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "processing"
        assert body["collection"] == "local_kb"
        assert body["accepted_files"] == 1

        task = _wait_task(client, body["task_id"])
        assert task["status"] == "done", task
        assert task["chunks"] > 0
        assert task["failed"] == 0

        # 文件已落盘到 medical_kb/神经内科/（文件名不带时间戳前缀）
        saved = list((tmp_kb / "神经内科").glob("*.md"))
        assert len(saved) == 1
        assert saved[0].name == "头痛问答.md"

        # 检索能命中刚灌入的内容
        import asyncio
        from src.core.retrieval import get_retriever
        local, _, _ = asyncio.run(get_retriever().retrieve(
            "苹果", classification="local", top_k=3
        ))
        assert any("苹果" in r["content"] for r in local), local

    def test_reupload_same_name_is_replace(self, doc_env):
        """同名重传 = 替换：向量不翻倍，旧内容被新内容覆盖"""
        app, retriever, tmp_kb, _ = doc_env
        client = TestClient(app)

        r1 = _upload(client, name="头痛.md", content="苹果相关旧内容。")
        t1 = _wait_task(client, r1.json()["task_id"])
        assert t1["status"] == "done"
        col = settings.milvus_collection_kb
        n1 = _count_doc_vectors(retriever, col, "头痛")

        r2 = _upload(client, name="头痛.md", content="香蕉相关新内容，覆盖旧版。")
        t2 = _wait_task(client, r2.json()["task_id"])
        assert t2["status"] == "done"
        n2 = _count_doc_vectors(retriever, col, "头痛")

        # 替换后同 doc_id 条数不翻倍（旧向量已清，只剩新内容的块）
        assert n2 <= max(1, n1), f"替换后应接近原量: {n1} -> {n2}"
        # 落盘只有一份文件
        assert len(list((tmp_kb / "神经内科").glob("头痛.md"))) == 1

        # 检索命中新内容（香蕉），旧内容（苹果）应被替换掉
        import asyncio
        from src.core.retrieval import get_retriever
        local, _, _ = asyncio.run(get_retriever().retrieve(
            "香蕉", classification="local", top_k=3
        ))
        assert any("香蕉" in r["content"] for r in local), local
        local2, _, _ = asyncio.run(get_retriever().retrieve(
            "苹果", classification="local", top_k=3
        ))
        assert not any("旧内容" in r["content"] for r in local2), local2

    def test_delete_document_removes_file_and_vectors(self, doc_env):
        app, retriever, tmp_kb, _ = doc_env
        client = TestClient(app)

        resp = _upload(client, name="可删除.md", content="苹果要被删掉。")
        task = _wait_task(client, resp.json()["task_id"])
        assert task["status"] == "done"
        col = settings.milvus_collection_kb
        assert _count_doc_vectors(retriever, col, "可删除") > 0

        d = client.delete("/api/v1/documents/可删除",
                          params={"department": "神经内科", "collection": "local_kb"})
        assert d.status_code == 200, d.text
        body = d.json()
        assert body["doc_id"] == "可删除"
        assert body["deleted_vectors"] > 0
        assert body["deleted_files"] == 1

        # 向量与文件都清了
        assert _count_doc_vectors(retriever, col, "可删除") == 0
        assert not (tmp_kb / "神经内科" / "可删除.md").exists()

    def test_upload_json_qa_list(self, doc_env):
        """json Q&A 列表按条切分入库"""
        app, retriever, tmp_kb, _ = doc_env
        client = TestClient(app)

        qa = [
            {"question": "苹果有什么好处？", "answer": "富含维生素C。"},
            {"question": "香蕉有什么好处？", "answer": "富含钾元素。"},
        ]
        import json as _json
        content = _json.dumps(qa, ensure_ascii=False)
        resp = _upload(client, name="水果问答.json", content=content)
        assert resp.status_code == 200, resp.text
        task = _wait_task(client, resp.json()["task_id"])
        assert task["status"] == "done", task
        assert task["chunks"] >= 2

        import asyncio
        from src.core.retrieval import get_retriever
        local, _, _ = asyncio.run(get_retriever().retrieve(
            "苹果", classification="local", top_k=5
        ))
        assert any("苹果有什么好处" in r["content"] for r in local), local

    def test_reject_pubmed_collection_upload(self, doc_env):
        """网页上传只开 local_kb；pubmed 上传必须 400（ADR-0005）"""
        app, *_ = doc_env
        client = TestClient(app)
        resp = _upload(client, collection="pubmed")
        assert resp.status_code == 400
        assert "local_kb" in resp.json()["detail"]

    def test_reject_unsupported_suffix(self, doc_env):
        app, *_ = doc_env
        client = TestClient(app)
        resp = _upload(client, name="病毒.exe", content="bad")
        assert resp.status_code == 400
        assert "不支持的文件类型" in resp.json()["detail"]

    def test_reject_unsafe_department(self, doc_env):
        app, *_ = doc_env
        client = TestClient(app)
        resp = _upload(client, department="../../etc")
        assert resp.status_code == 400
        assert "department" in resp.json()["detail"]

    def test_reject_too_many_files(self, doc_env):
        app, *_ = doc_env
        client = TestClient(app)
        files = [("files", (f"f{i}.md", "苹果内容。".encode("utf-8"), "text/markdown"))
                 for i in range(21)]
        resp = _upload(client, files=files)
        assert resp.status_code == 400
        assert "单批最多" in resp.json()["detail"]

    def test_reject_oversize_file(self, doc_env, monkeypatch):
        app, *_ = doc_env
        from src.api import documents as documents_mod
        monkeypatch.setattr(documents_mod, "_MAX_FILE_BYTES", 10)  # 缩小到 10B
        client = TestClient(app)
        resp = _upload(client, name="big.md", content="苹果内容" * 10)
        assert resp.status_code == 400
        assert "文件过大" in resp.json()["detail"]

    def test_task_query_and_list(self, doc_env):
        app, *_ = doc_env
        client = TestClient(app)
        resp = _upload(client, name="感冒.md", content="感冒多喝热水。")
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]
        task = _wait_task(client, task_id)
        assert task["status"] == "done"

        tasks = client.get("/api/v1/documents/tasks").json()
        assert any(t["task_id"] == task_id for t in tasks)

        docs = client.get("/api/v1/documents", params={"collection": "local_kb"}).json()
        assert docs["collection"] == "local_kb"
        assert len(docs["files"]) >= 1
        assert docs["files"][0]["path"].startswith("神经内科/")

    def test_missing_task_404(self, doc_env):
        app, *_ = doc_env
        client = TestClient(app)
        assert client.get("/api/v1/documents/tasks/nope").status_code == 404
