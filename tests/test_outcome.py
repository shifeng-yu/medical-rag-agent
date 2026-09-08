# ============================================================
# 问诊结果出口模块（outcome）单元测试
# 表格化覆盖：7 种出口的 写会话规则 / 指标口径 / 返回形状
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.core import outcome
from src.monitoring.logger import RequestMetrics, request_logger
from src.session.manager import session_manager


@pytest.fixture(autouse=True)
def capture_side_effects(monkeypatch):
    """收银台的副作用捕获器：不落真实会话/日志文件"""
    written = []
    logged = []

    def fake_add_message(*args, **kwargs):
        written.append(args)

    def fake_log_request(metrics):
        logged.append(metrics)

    monkeypatch.setattr(session_manager, "add_message", fake_add_message)
    monkeypatch.setattr(request_logger, "log_request", fake_log_request)
    return written, logged


# kind → (是否写会话, 是否 success)
EXIT_TABLE = [
    (outcome.EXIT_OK, True, True),
    (outcome.EXIT_EMERGENCY_BLOCKED, True, False),
    (outcome.EXIT_CONTENT_BLOCKED, True, False),
    (outcome.EXIT_DEGRADED, True, False),
    (outcome.EXIT_ERROR, False, False),
    (outcome.EXIT_OVERLOAD, False, False),
    (outcome.EXIT_TIMEOUT, False, False),
]


class TestFinishTable:
    """收银台对 7 种出口的行为表"""

    @pytest.mark.parametrize("kind,expect_write,expect_success", EXIT_TABLE)
    def test_exit_table(
        self, capture_side_effects, kind, expect_write, expect_success
    ):
        written, logged = capture_side_effects
        metrics = RequestMetrics("sid-1")

        result = outcome.finish(
            "sid-1", metrics, kind=kind, answer=f"答复-{kind}",
        )

        # 写会话规则
        writes = [a for a in written if a[1] == "assistant"]
        if expect_write:
            assert len(writes) == 1, f"{kind} 应写入会话"
            assert writes[0][2] == f"答复-{kind}"
        else:
            assert len(writes) == 0, f"{kind} 不应写入会话（系统失败出口）"

        # 指标口径
        assert len(logged) == 1, "每次出口恰好上报一次"
        m = logged[0]
        assert m.outcome == kind
        assert m.success is expect_success
        assert m.final_output == f"答复-{kind}"

        # 返回形状固定 6 键（含 outcome 出口标记）
        assert set(result.keys()) == {
            "answer", "sources", "judge_result", "latency_ms", "session_id",
            "outcome",
        }
        assert result["session_id"] == "sid-1"
        assert result["answer"] == f"答复-{kind}"
        assert result["outcome"] == kind
        assert result["sources"] == []
        assert result["latency_ms"] >= 0

    def test_judge_layer_defaults_to_kind(self, capture_side_effects):
        """未给 judge_result 时 layer 默认 = 出口 kind"""
        metrics = RequestMetrics("sid-1")
        result = outcome.finish(
            "sid-1", metrics, kind=outcome.EXIT_TIMEOUT, answer="超时",
        )
        assert result["judge_result"] == {"layer": outcome.EXIT_TIMEOUT}

    def test_judge_layer_preserved_when_provided(self, capture_side_effects):
        """给了带 layer 的 judge_result（如 ok 出口的裁判报告）则原样保留"""
        metrics = RequestMetrics("sid-1")
        judge_report = {"layer": "both", "scores": {"fact": 9}}
        result = outcome.finish(
            "sid-1", metrics,
            kind=outcome.EXIT_OK, answer="正常回答",
            judge_result=judge_report,
        )
        assert result["judge_result"] == judge_report

    def test_judge_layer_filled_when_missing(self, capture_side_effects):
        """给了不带 layer 的 payload（拦截原因）则补上 layer=kind"""
        metrics = RequestMetrics("sid-1")
        result = outcome.finish(
            "sid-1", metrics,
            kind=outcome.EXIT_EMERGENCY_BLOCKED,
            answer="请立即就医",
            judge_result={"reason": "疑似心梗"},
        )
        assert result["judge_result"]["layer"] == outcome.EXIT_EMERGENCY_BLOCKED
        assert result["judge_result"]["reason"] == "疑似心梗"

    def test_error_message_passthrough(self, capture_side_effects):
        """error 出口透传 error_message 到指标，不进返回形状"""
        metrics = RequestMetrics("sid-1")
        result = outcome.finish(
            "sid-1", metrics,
            kind=outcome.EXIT_ERROR, answer="兜底", error_message="boom",
        )
        _, logged = capture_side_effects
        assert logged[0].error_message == "boom"
        assert "error" not in result  # 返回形状无 error 键（仅 outcome 标记出口）
        assert result["outcome"] == outcome.EXIT_ERROR

    def test_invalid_kind_rejected(self, capture_side_effects):
        """未知 kind 直接抛错（fail fast，防止新出口漏进词汇表）"""
        metrics = RequestMetrics("sid-1")
        with pytest.raises(ValueError):
            outcome.finish("sid-1", metrics, kind="mystery", answer="x")

    def test_write_session_override(self, capture_side_effects):
        """write_session 可显式覆盖 kind 默认规则"""
        written, _ = capture_side_effects
        metrics = RequestMetrics("sid-1")
        outcome.finish(
            "sid-1", metrics,
            kind=outcome.EXIT_ERROR, answer="道歉", write_session=True,
        )
        assert any(a[1] == "assistant" for a in written)


class TestToSources:
    """sources 白名单映射"""

    def test_whitelist_fields_only(self):
        reranked = [{
            "title": "指南", "source": "local_kb",
            "rerank_score": 0.88, "publish_time": "2024", "department": "心内科",
            "authority_tier": "tier_1", "authority_label": "临床指南",
            "content": "不应泄漏的内部字段",  # 非白名单字段必须被剔除
        }]
        out = outcome.to_sources(reranked)
        assert out == [{
            "title": "指南", "source": "local_kb",
            "score": 0.88, "publish_time": "2024", "department": "心内科",
            "authority_tier": "tier_1", "authority_label": "临床指南",
        }]

    def test_limit_and_empty(self):
        assert outcome.to_sources([{"title": "a"}, {"title": "b"}], limit=1) == [
            {"title": "a", "source": "", "score": 0, "publish_time": "",
             "department": "", "authority_tier": "", "authority_label": ""},
        ]
        assert outcome.to_sources([]) == []
        assert outcome.to_sources(None) == []


class TestDegrade:
    """API 层降级出口（修复 session_id 缺失导致的 500）"""

    @pytest.mark.parametrize("kind", [outcome.EXIT_OVERLOAD, outcome.EXIT_TIMEOUT])
    def test_degrade_shape_and_no_session_write(self, capture_side_effects, kind):
        written, logged = capture_side_effects
        result = outcome.degrade("anon-1", kind, "请稍后重试")

        assert set(result.keys()) == {
            "answer", "sources", "judge_result", "latency_ms", "session_id",
            "outcome",
        }
        assert result["session_id"] == "anon-1"      # 历史 bug：此处曾缺失
        assert result["outcome"] == kind
        assert result["judge_result"]["layer"] == kind
        assert not any(a[1] == "assistant" for a in written)  # 不写会话
        assert logged[0].outcome == kind
        assert logged[0].success is False
