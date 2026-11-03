# ============================================================
# 知识库任务链路测试（不依赖 Milvus）
#
# 覆盖：
#   - 上传 → 任务落库 → 后台执行 → 状态可跨请求查询
#   - 文档元数据（kb_document）随入库写入
#   - 切分预览（dry-run）：出块、截断、且不写知识库目录
#   - 单文档重建 / 集合级重建：任务类型、文件数、调用路径
#
# 向量化用桩替换（_ingest_one 返回固定块数），只验任务编排与落库语义。
# ============================================================

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import settings
from src.storage import repository


@pytest.fixture()
def kb_client(monkeypatch, tmp_path):
    """独立 app + 临时知识库目录（每次测试一套，互不干扰）。"""
    monkeypatch.setattr(settings, "medical_kb_dir", str(tmp_path / "kb"))
    monkeypatch.setattr(settings, "pubmed_offline_dir", str(tmp_path / "pub"))

    from src.api.documents import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), tmp_path


@pytest.fixture()
def fake_ingest(monkeypatch):
    """替换真实向量化，记录调用参数，固定返回 3 块/文件。"""
    calls = []

    def _fake(path: Path, collection: str) -> int:
        calls.append((path.name, collection))
        return 3

    monkeypatch.setattr("src.api.documents._ingest_one", _fake)
    return calls


def _wait_task(client, task_id: str, timeout: float = 3.0) -> dict:
    """轮询到终态。上传/重建接口返回的是"受理瞬间"的快照，
    后台任务在响应之后才执行，断言终态必须再查一次。"""
    import time

    deadline = time.time() + timeout
    task = client.get(f"/api/v1/documents/tasks/{task_id}").json()
    while task["status"] not in ("done", "error") and time.time() < deadline:
        time.sleep(0.02)
        task = client.get(f"/api/v1/documents/tasks/{task_id}").json()
    return task


def _upload(client, name: str, content: str, department: str = "神经内科"):
    return client.post(
        "/api/v1/documents/upload",
        files=[("files", (name, content.encode("utf-8"), "text/markdown"))],
        data={"department": department, "collection": "local_kb"},
    )


class TestUploadTask:
    def test_task_persisted_and_document_recorded(self, kb_client, fake_ingest):
        client, tmp_path = kb_client
        doc_name = f"头痛-{uuid.uuid4().hex[:6]}.md"

        resp = _upload(client, doc_name, "# 头痛\n\n紧张性头痛常见诱因是睡眠不足。")
        assert resp.status_code == 200, resp.text
        task_id = resp.json()["task_id"]

        # 任务跑完（TestClient 在响应后同步执行 BackgroundTasks）
        task = client.get(f"/api/v1/documents/tasks/{task_id}").json()
        assert task["status"] == "done"
        assert task["chunks"] == 3
        assert task["kind"] == "upload"
        assert task["executor"] == "inline"

        # 落盘 + 元数据入库
        assert (tmp_path / "kb" / "神经内科" / doc_name).exists()
        docs = repository.list_documents("local_kb")
        assert any(
            d["doc_id"] == Path(doc_name).stem and d["chunk_count"] == 3
            for d in docs
        )
        assert fake_ingest[-1] == (doc_name, "local_kb")

    def test_task_list_is_not_process_memory(self, kb_client, fake_ingest):
        """任务状态来自 kb_task：换一个客户端（新请求上下文）仍能列出。"""
        client, _ = kb_client
        task_id = _upload(
            client, f"头痛-{uuid.uuid4().hex[:6]}.md", "# 头痛\n内容"
        ).json()["task_id"]

        listing = client.get("/api/v1/documents?collection=local_kb").json()
        assert any(t["task_id"] == task_id for t in listing["tasks"])

        tasks = client.get("/api/v1/documents/tasks?limit=50").json()
        assert any(t["task_id"] == task_id for t in tasks)

    def test_unknown_task_404(self, kb_client):
        client, _ = kb_client
        assert client.get("/api/v1/documents/tasks/nope").status_code == 404


class TestChunkPreview:
    def test_preview_returns_chunks_without_touching_kb(self, kb_client):
        client, tmp_path = kb_client
        question = "问：紧张性头痛的常见诱因有哪些？\n答：睡眠不足、精神压力、长时间伏案。\n"
        text = "# 紧张性头痛\n" + question * 8

        resp = client.post(
            "/api/v1/documents/preview",
            files=[("files", ("预览样本.md", text.encode("utf-8"), "text/markdown"))],
            data={"department": "神经内科", "max_chunks": "3"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["total_chunks"] >= 1
        assert body["department"] == "神经内科"
        file_info = body["files"][0]
        assert file_info["doc_id"] == "预览样本"
        assert file_info["chunk_count"] >= 1
        assert file_info["returned"] == min(file_info["chunk_count"], 3)
        assert len(file_info["chunks"]) == file_info["returned"]
        assert file_info["truncated"] is (file_info["chunk_count"] > 3)
        assert all(c["text"] for c in file_info["chunks"])

        # dry-run：不写知识库目录、不产生向量
        assert not (tmp_path / "kb").exists()

    def test_preview_rejects_bad_suffix(self, kb_client):
        client, _ = kb_client
        resp = client.post(
            "/api/v1/documents/preview",
            files=[("files", ("bad.exe", b"x", "application/octet-stream"))],
            data={"department": "神经内科"},
        )
        assert resp.status_code == 400


class TestRebuild:
    def test_single_document_rebuild(self, kb_client, fake_ingest):
        client, tmp_path = kb_client
        folder = tmp_path / "kb" / "心内科"
        folder.mkdir(parents=True)
        (folder / "冠心病.md").write_text("# 冠心病\n内容", encoding="utf-8")

        resp = client.post(
            "/api/v1/documents/冠心病/rebuild",
            params={"department": "心内科", "collection": "local_kb"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "rebuild_doc"
        assert body["doc_id"] == "冠心病"

        task = _wait_task(client, body["task_id"])
        assert task["status"] == "done"
        assert task["chunks"] == 3
        assert fake_ingest[-1] == ("冠心病.md", "local_kb")

    def test_rebuild_missing_document_404(self, kb_client):
        client, _ = kb_client
        resp = client.post(
            "/api/v1/documents/不存在/rebuild", params={"department": "心内科"}
        )
        assert resp.status_code == 404

    def test_collection_rebuild_counts_files(self, kb_client, fake_ingest):
        client, tmp_path = kb_client
        folder = tmp_path / "kb" / "呼吸科"
        folder.mkdir(parents=True)
        for name in ("a.md", "b.md"):
            (folder / name).write_text("# 内容", encoding="utf-8")

        resp = client.post(
            "/api/v1/documents/rebuild", params={"collection": "local_kb"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kind"] == "rebuild_collection"
        assert body["files"] == 2

        task = _wait_task(client, body["task_id"])
        assert task["status"] == "done"
        assert task["chunks"] == 6
        assert {c[0] for c in fake_ingest} >= {"a.md", "b.md"}

    def test_rebuild_empty_collection_404(self, kb_client):
        client, _ = kb_client
        resp = client.post(
            "/api/v1/documents/rebuild", params={"collection": "local_kb"}
        )
        assert resp.status_code == 404


def test_invalid_file_in_batch_writes_nothing(kb_client):
    """一批里有一个非法文件时，合法文件也不该落盘。

    否则磁盘上会留下"有文件、没任务、库里没向量"的孤儿，列表里显示未入库，
    用户还得手工清理。
    """
    client, tmp_path = kb_client
    resp = client.post(
        "/api/v1/documents/upload",
        files=[
            ("files", ("good.md", b"# ok", "text/markdown")),
            ("files", ("bad.exe", b"x", "application/octet-stream")),
        ],
        data={"department": "神经内科", "collection": "local_kb"},
    )
    assert resp.status_code == 400
    assert not (tmp_path / "kb").exists(), "校验失败时不应创建目录或落盘"
