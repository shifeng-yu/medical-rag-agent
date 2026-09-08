# ============================================================
# workflow 出口回归测试
# 覆盖: run() 的 emergency / content_blocked / degraded / ok / error
#       五个出口全部收敛到问诊结果出口模块，形状固定、不漏记账
# 说明: 不打真实模型/向量库，所有节点以 monkeypatch 顶替
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.core import outcome
from src.monitoring.logger import request_logger
from src.session.manager import session_manager


@pytest.fixture(autouse=True)
def capture_effects(monkeypatch):
    """捕获会话写入与指标上报（不落真实会话/日志）"""
    written = []
    logged = []

    def fake_add_message(*args, **kwargs):
        written.append(args)

    def fake_log_request(metrics):
        logged.append(metrics)

    monkeypatch.setattr(session_manager, "add_message", fake_add_message)
    monkeypatch.setattr(request_logger, "log_request", fake_log_request)
    return written, logged


class TestWorkflowExits:
    """五个出口统一走收银台"""

    async def _run(self, *args, **kwargs):
        from src.core.workflow import workflow
        return await workflow.run(*args, **kwargs)

    def _recorded(self, capture_effects):
        written, logged = capture_effects
        return written, logged

    def test_emergency_exit_is_logged_and_written(self, monkeypatch, capture_effects):
        """急症拦截: 写会话 + 补上报（历史 bug: 漏 log_request）"""
        import src.core.workflow as wf

        monkeypatch.setattr(
            wf.safety_filter, "check_emergency",
            lambda q: (True, "疑似心梗", "请立即前往急诊"),
        )
        written, logged = self._recorded(capture_effects)

        import asyncio
        result = asyncio.run(wf.workflow.run("胸痛怎么办"))

        assert result["judge_result"]["layer"] == "emergency_blocked"
        assert result["judge_result"]["reason"] == "疑似心梗"
        assert set(result.keys()) == {
            "answer", "sources", "judge_result", "latency_ms", "session_id",
            "outcome",
        }
        assert result["outcome"] == "emergency_blocked"
        # 写会话（面向用户的正式答复）
        assert any(a[1] == "assistant" and a[2] == "请立即前往急诊" for a in written)
        # 补上报（历史 bug 修复点）
        assert len(logged) == 1
        assert logged[0].outcome == "emergency_blocked"
        assert logged[0].success is False

    def test_content_blocked_exit_is_logged(self, monkeypatch, capture_effects):
        """敏感内容拦截: 同样补上报"""
        import src.core.workflow as wf

        monkeypatch.setattr(wf.safety_filter, "check_emergency", lambda q: (False, None, None))
        monkeypatch.setattr(
            wf.safety_filter, "check_input_safety",
            lambda q: (True, "敏感内容", "这个我无法回答"),
        )
        written, logged = self._recorded(capture_effects)

        import asyncio
        result = asyncio.run(wf.workflow.run("如何自杀"))

        assert result["judge_result"]["layer"] == "content_blocked"
        assert len(logged) == 1
        assert logged[0].outcome == "content_blocked"
        assert any(a[1] == "assistant" for a in written)

    def _stub_classifier(self, monkeypatch):
        """顶替 classifier，避免 L1 词典兜底失败时触发真实 LLM"""
        import src.core.workflow as wf

        monkeypatch.setattr(wf.classifier, "classify", lambda q: "local")
        monkeypatch.setattr(wf.classifier, "should_trigger_high_judge", lambda q: False)
        monkeypatch.setattr(wf.classifier, "get_source_weight", lambda q, c: (0.7, 0.3))

    def test_degraded_exit_when_retrieval_empty(self, monkeypatch, capture_effects):
        """检索全空 → degraded 出口: 写会话、记为失败、形状固定"""
        import src.core.workflow as wf

        async def fake_retrieve(query, classification):
            return [], [], 0.0

        monkeypatch.setattr(wf.retriever, "retrieve", fake_retrieve)
        monkeypatch.setattr(wf.generator, "generate_degraded", lambda q: "检索暂不可用，基于通用知识回答")
        monkeypatch.setattr(wf.safety_filter, "check_emergency", lambda q: (False, None, None))
        monkeypatch.setattr(wf.safety_filter, "check_input_safety", lambda q: (False, None, None))
        self._stub_classifier(monkeypatch)
        written, logged = self._recorded(capture_effects)

        import asyncio
        result = asyncio.run(wf.workflow.run("最近总失眠怎么办"))

        assert result["judge_result"]["layer"] == "degraded"
        assert result["answer"].startswith("检索暂不可用")
        assert len(logged) == 1
        assert logged[0].outcome == "degraded"
        assert logged[0].success is False  # 新口径: 仅 ok 算 success
        assert any(a[1] == "assistant" for a in written)

    def test_ok_exit_shape_and_whitelist(self, monkeypatch, capture_effects):
        """正常出口: sources 走白名单，shape 固定"""
        import src.core.workflow as wf

        async def fake_retrieve(query, classification):
            return [{"title": "本地指南", "source": "local_kb", "content": "x"}], [], 0.0

        def fake_rerank(query, local, pubmed, source_weights=None):
            return [{
                "title": "本地指南", "source": "local_kb",
                "rerank_score": 0.9, "publish_time": "2024", "department": "心内科",
                "content": "不应泄漏",  # 非白名单字段
            }]

        def fake_generate(**kwargs):
            return "这是正常回答。", 10.0

        def fake_validate(**kwargs):
            return True, {"layer": "both", "scores": {"fact": 9}}, ""

        monkeypatch.setattr(wf.retriever, "retrieve", fake_retrieve)
        monkeypatch.setattr(wf.reranker, "rerank", fake_rerank)
        monkeypatch.setattr(wf.generator, "generate", fake_generate)
        monkeypatch.setattr(wf.judge, "validate", fake_validate)
        monkeypatch.setattr(wf.safety_filter, "check_emergency", lambda q: (False, None, None))
        monkeypatch.setattr(wf.safety_filter, "check_input_safety", lambda q: (False, None, None))
        monkeypatch.setattr(wf.safety_filter, "check_output_compliance", lambda a, r: (True, []))
        self._stub_classifier(monkeypatch)
        written, logged = self._recorded(capture_effects)

        import asyncio
        result = asyncio.run(wf.workflow.run("冠心病人能运动吗？"))

        assert result["judge_result"]["layer"] == "both"
        assert len(logged) == 1
        assert logged[0].outcome == "ok"
        assert logged[0].success is True
        assert any(a[1] == "assistant" for a in written)
        # sources 只含白名单键（含 reranker 打标的权威分级）
        assert set(result["sources"][0].keys()) == {
            "title", "source", "score", "publish_time", "department",
            "authority_tier", "authority_label",
        }
        assert "content" not in result["sources"][0]
        assert result["sources"][0]["score"] == 0.9

    def test_error_exit_no_session_write(self, monkeypatch, capture_effects):
        """内部异常 → error 出口: 不写会话（系统失败规则）、仍上报"""
        import src.core.workflow as wf

        def boom(q):
            raise RuntimeError("检索崩溃")

        monkeypatch.setattr(wf.safety_filter, "check_emergency", boom)
        written, logged = self._recorded(capture_effects)

        import asyncio
        result = asyncio.run(wf.workflow.run("你好"))

        assert result["judge_result"]["layer"] == "error"
        assert set(result.keys()) == {
            "answer", "sources", "judge_result", "latency_ms", "session_id",
            "outcome",
        }
        assert result["outcome"] == "error"
        assert len(logged) == 1
        assert logged[0].outcome == "error"
        assert logged[0].success is False
        assert logged[0].error_message == "检索崩溃"
        # 系统失败出口不写会话
        assert not any(a[1] == "assistant" for a in written)
