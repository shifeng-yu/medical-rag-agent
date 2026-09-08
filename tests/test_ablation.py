# ============================================================
# 消融评测单测：evaluate 统计口径 + AblationConfig → RunOptions 映射
# 说明: 不跑真实链路，monkeypatch workflow.run 验证统计逻辑
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.core.workflow import RunOptions
from scripts.ablation import AblationConfig, build_configs, evaluate


class TestConfigMapping:
    """AblationConfig → RunOptions 映射"""

    def test_full_maps_to_all_on_except_session(self):
        cfg = AblationConfig(name="full")
        opts = cfg.to_options()
        assert isinstance(opts, RunOptions)
        assert opts.use_judge is True
        assert opts.use_reranker is True
        assert opts.use_source_weight is True
        assert opts.use_session is False   # 评测模式

    def test_disabled_maps_to_off(self):
        cfg = AblationConfig(
            name="no-judge",
            use_judge=False, use_reranker=True, use_source_weight=False,
        )
        opts = cfg.to_options()
        assert opts.use_judge is False
        assert opts.use_source_weight is False
        assert opts.use_reranker is True

    def test_build_configs_honors_disable(self):
        cfgs = build_configs(["judge", "reranker"])
        assert len(cfgs) == 1
        assert cfgs[0].use_judge is False
        assert cfgs[0].use_reranker is False
        assert cfgs[0].use_source_weight is True


class TestEvaluateStats:
    """evaluate 的命中/出口统计口径"""

    def _monkeypatch_run(self, monkeypatch, answers: dict):
        """answers: query 前缀 → (answer, outcome)。返回记录到的 options 列表"""
        from src.core import workflow as wf_mod
        seen_options = []

        async def fake_run(query, session_id=None, options=None):
            seen_options.append(options)
            answer, outcome_kind = answers.get(query, ("默认答复", "ok"))
            return {
                "answer": answer,
                "sources": [],
                "judge_result": {"layer": "both"},
                "latency_ms": 1.0,
                "session_id": session_id,
                "outcome": outcome_kind,
            }

        monkeypatch.setattr(wf_mod.workflow, "run", fake_run)
        return seen_options

    def test_hit_when_expected_in_answer(self, monkeypatch):
        self._monkeypatch_run(monkeypatch, {
            "头痛怎么办": ("布洛芬可用于缓解头痛【来源：指南】", "ok"),
        })
        cfg = AblationConfig(name="full")
        result = evaluate([{"question": "头痛怎么办", "answer": "布洛芬"}], cfg)

        assert result["total_samples"] == 1
        assert result["correct"] == 1
        assert result["top1_accuracy"] == 1.0
        assert result["outcome_counts"] == {"ok": 1}

    def test_miss_when_answer_mismatch(self, monkeypatch):
        self._monkeypatch_run(monkeypatch, {
            "头痛怎么办": ("建议多休息【来源：指南】", "ok"),
        })
        result = evaluate([{"question": "头痛怎么办", "answer": "布洛芬"}],
                          AblationConfig(name="full"))
        assert result["correct"] == 0

    def test_non_ok_outcome_counts_as_miss_and_recorded(self, monkeypatch):
        """拦截/降级/error 出口：未命中 + outcome 如实记录"""
        self._monkeypatch_run(monkeypatch, {
            "胸闷怎么办": ("请立即前往急诊", "emergency_blocked"),
            "罕见病": ("降级回答", "degraded"),
            "正常病": ("含标准答案的合规回答【来源：指南】", "ok"),
        })
        cfg = AblationConfig(name="full")
        result = evaluate([
            {"question": "胸闷怎么办", "answer": "硝酸甘油"},
            {"question": "罕见病", "answer": "任何答案"},
            {"question": "正常病", "answer": "标准答案"},
        ], cfg)

        assert result["correct"] == 1                    # 只有 ok 出口命中
        assert result["outcome_counts"] == {
            "emergency_blocked": 1, "degraded": 1, "ok": 1,
        }

    def test_passes_options_and_samples_limit(self, monkeypatch):
        seen = self._monkeypatch_run(monkeypatch, {
            "q1": ("答案一【来源：指南】", "ok"),
            "q2": ("答案二【来源：指南】", "ok"),
        })
        cfg = AblationConfig(name="no-judge", use_judge=False)
        result = evaluate(
            [{"question": "q1", "answer": "答案一"},
             {"question": "q2", "answer": "答案二"}],
            cfg, max_samples=1,
        )

        assert result["total_samples"] == 1              # max_samples 截断
        assert len(seen) == 1
        assert seen[0].use_judge is False                # 透传了配置的开关
        assert seen[0].use_session is False
