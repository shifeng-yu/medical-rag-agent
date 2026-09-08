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


class TestGracefulDegradation:
    """API 层降级回归：超时/过载不再因缺 session_id 崩成 500"""

    def test_timeout_returns_200_with_session(self, client, monkeypatch):
        """workflow 超时 → 200 + layer=timeout + 固定形状"""
        import asyncio
        from src.core.workflow import workflow

        async def hang(query, session_id=None):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(workflow, "run", hang)
        resp = client.post("/api/v1/chat", json={"query": "头痛怎么办"})

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["judge_result"]["layer"] == "timeout"
        assert data["session_id"]              # 历史 bug: 此处曾 KeyError
        assert data["sources"] == []
        assert data["answer"]

    def test_overload_returns_200_with_session(self, client, monkeypatch):
        """workflow 异常 → 200 + layer=overload + session_id 补全"""
        from src.core.workflow import workflow

        async def boom(query, session_id=None):
            raise RuntimeError("模型崩了")

        monkeypatch.setattr(workflow, "run", boom)
        resp = client.post("/api/v1/chat", json={"query": "头痛怎么办"})

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["judge_result"]["layer"] == "overload"
        assert data["session_id"]
        assert data["answer"]


class TestBatchRobustness:
    """批量路由容错：结果 dict 新增键不再导致崩溃"""

    def test_batch_tolerates_extra_keys(self, client, monkeypatch):
        """workflow 返回多带键（含 sources/judge_result 里多余字段）→ 仍 200"""
        from src.core.workflow import workflow

        async def fake_run(query, session_id=None):
            return {
                "answer": "批量回答",
                "sources": [{
                    "title": "指南", "source": "local_kb", "score": 0.9,
                    "publish_time": "", "department": "心内科",
                    "vector": [1, 2, 3],        # 多余键（旧代码 **splat 会崩）
                }],
                "judge_result": {
                    "layer": "both", "passed": True,  # 多余键
                },
                "latency_ms": 1.0,
                "session_id": "batch-session",
                "internal_note": "不应泄漏",          # 顶层多余键
            }

        monkeypatch.setattr(workflow, "run", fake_run)
        resp = client.post("/api/v1/chat/batch", json={
            "queries": [{"query": "问题一"}, {"query": "问题二"}],
        })

        assert resp.status_code == 200, resp.text
        items = resp.json()
        assert len(items) == 2
        assert items[0]["sources"][0]["source"] == "local_kb"
        assert items[0]["judge_result"]["layer"] == "both"
        assert items[0]["session_id"] == "batch-session"
        assert "vector" not in items[0]["sources"][0]  # 序列化时被剔除


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
