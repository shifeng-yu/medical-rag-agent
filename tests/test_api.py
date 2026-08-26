# ============================================================
# API 层集成测试（FastAPI TestClient）
# 覆盖: /health /chat /stats 路由的请求-响应契约
# 说明: 不依赖真实模型/Milvus，通过 mock workflow 验证路由与序列化
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """构造 TestClient（触发应用启动，模型缺失时仅告警不崩溃）"""
    from src.main import app
    with TestClient(app) as c:
        yield c


class TestHealth:
    """健康检查接口"""

    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "models_loaded" in data
        assert "milvus_connected" in data


class TestChat:
    """核心问诊接口（mock workflow，验证路由契约）"""

    @pytest.fixture(autouse=True)
    def mock_workflow(self, monkeypatch):
        """替换 workflow.run，避免依赖模型/向量库"""
        async def fake_run(query, session_id=None):
            return {
                "answer": "测试回答【来源：中国测试指南 Tier1】",
                "sources": [{
                    "title": "中国测试指南", "source": "local_kb",
                    "score": 0.9, "publish_time": "", "department": "心内科",
                }],
                "judge_result": {
                    "layer": "both",
                    "scores": {"fact_consistency": 9, "logic": 8, "usefulness": 9},
                },
                "latency_ms": 120.5,
                "session_id": "test-session-001",
            }
        monkeypatch.setattr(
            "src.core.workflow.workflow.run", fake_run
        )

    def test_chat_single(self, client):
        resp = client.post("/api/v1/chat", json={"query": "冠心病人能运动吗？"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]
        assert data["sources"][0]["source"] in ("local_kb", "pubmed")
        assert data["judge_result"]["layer"] == "both"
        assert data["judge_result"]["scores"]["fact_consistency"] >= 6
        assert data["latency_ms"] > 0
        assert data["session_id"]

    def test_chat_with_session_id(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"query": "那需要注意什么？", "session_id": "test-session-001"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test-session-001"

    def test_chat_empty_query_rejected(self, client):
        """空 query 应被请求模型拒绝（400）"""
        resp = client.post("/api/v1/chat", json={"query": ""})
        assert resp.status_code == 422  # pydantic min_length 校验


class TestStats:
    """运行统计接口"""

    def test_stats_ok(self, client):
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "success_rate" in data
        assert "avg_latency_ms" in data
        assert "judge_pass_rate" in data
