# ============================================================
# 反馈与会话复盘接口测试
#
# 覆盖：反馈提交/覆盖写/归属校验、反馈看板、会话列表/回放/删除。
# 只挂载 sessions + feedback 两个路由，不触发 Milvus/模型启动。
# ============================================================

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.session.manager import session_manager
from src.storage import repository


@pytest.fixture(scope="module")
def client():
    from src.api.feedback import router as feedback_router
    from src.api.sessions import router as sessions_router

    app = FastAPI()
    app.include_router(feedback_router)
    app.include_router(sessions_router)
    with TestClient(app) as c:
        yield c


def _seed_turn(source_title: str = "头痛指南") -> tuple[str, int, int]:
    """造一个会话 + 一轮问答，返回 (session_id, user_mid, assistant_mid)。"""
    sid = f"fb-{uuid.uuid4().hex[:8]}"
    session_manager.get_or_create_session(sid)
    user_mid = session_manager.add_message(sid, "user", "最近总是头痛")
    assistant_mid = session_manager.add_message(
        sid, "assistant", "建议规律作息，必要时就医。",
        outcome="ok",
        judge_result={"layer": "both", "scores": {"factual_consistency": 9}},
        latency_ms=15.0,
        stream_mode="token",
        sources=[{
            "title": source_title, "source": "local_kb", "score": 0.91,
            "doc_id": "headache", "chunk_id": "abc123def456",
            "snippet": "紧张性头痛的常见诱因包括睡眠不足。",
            "department": "神经内科", "authority_tier": "tier_1",
            "authority_label": "临床指南",
        }],
    )
    return sid, user_mid, assistant_mid


class TestFeedback:
    def test_submit_and_overwrite(self, client):
        sid, _, mid = _seed_turn()

        resp = client.post("/api/v1/feedback", json={
            "message_id": mid, "rating": -1,
            "reason_tags": ["引用不相关"], "comment": "来源不对",
        })
        assert resp.status_code == 200, resp.text
        first = resp.json()
        assert first["message_id"] == mid
        assert first["session_id"] == sid
        assert first["rating"] == -1
        assert first["reason_tags"] == ["引用不相关"]

        # 改主意：同一条回答覆盖写，不新增一行
        resp = client.post("/api/v1/feedback", json={
            "message_id": mid, "rating": 1, "reason_tags": [], "comment": "",
        })
        assert resp.status_code == 200
        second = resp.json()
        assert second["id"] == first["id"]
        assert second["rating"] == 1

    def test_unknown_message_404(self, client):
        resp = client.post("/api/v1/feedback", json={
            "message_id": 99999999, "rating": 1,
        })
        assert resp.status_code == 404

    def test_cannot_rate_user_message(self, client):
        _, user_mid, _ = _seed_turn()
        resp = client.post("/api/v1/feedback", json={
            "message_id": user_mid, "rating": 1,
        })
        assert resp.status_code == 400

    def test_rating_must_be_pm1(self, client):
        _, _, mid = _seed_turn()
        resp = client.post("/api/v1/feedback", json={
            "message_id": mid, "rating": 0,
        })
        assert resp.status_code == 422

    def test_stats_shape_and_reason(self, client):
        tag = f"答非所问-{uuid.uuid4().hex[:6]}"
        _, _, mid = _seed_turn()
        client.post("/api/v1/feedback", json={
            "message_id": mid, "rating": -1, "reason_tags": [tag],
        })

        resp = client.get("/api/v1/feedback/stats?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data) == {
            "days", "total", "positive", "negative",
            "positive_rate", "top_reasons", "by_day",
        }
        assert data["total"] >= 1
        assert any(r["tag"] == tag for r in data["top_reasons"])


class TestSessionReview:
    def test_list_contains_session(self, client):
        sid, _, _ = _seed_turn()
        resp = client.get("/api/v1/sessions?limit=100")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        item = next(i for i in body["items"] if i["session_id"] == sid)
        assert item["title"] == "最近总是头痛"
        assert item["turn_count"] == 1

    def test_detail_has_messages_sources_and_feedback(self, client):
        sid, _, mid = _seed_turn(source_title="紧张性头痛指南")
        client.post("/api/v1/feedback", json={
            "message_id": mid, "rating": 1,
        })

        resp = client.get(f"/api/v1/sessions/{sid}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["messages"]) == 2

        assistant = body["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["sources"][0]["title"] == "紧张性头痛指南"
        assert assistant["sources"][0]["snippet"].startswith("紧张性头痛")
        assert assistant["feedback"]["rating"] == 1
        assert assistant["judge_scores"]["factual_consistency"] == 9

    def test_delete_removes_session(self, client):
        sid, _, _ = _seed_turn()
        resp = client.delete(f"/api/v1/sessions/{sid}")
        assert resp.status_code == 200
        assert repository.get_session(sid) is None
        assert client.get(f"/api/v1/sessions/{sid}").status_code == 404

    def test_unknown_session_404(self, client):
        assert client.get("/api/v1/sessions/not-exist").status_code == 404
