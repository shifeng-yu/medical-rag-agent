# ============================================================
# benchmark 评测闭环测试（纯逻辑，免 GPU / 免模型 / 免向量库）
# 覆盖:
#   1. judge_high_risk：合规违规 / 编造实体判定
#   2. build_report_md：结果表格与口径文字
#   3. --check 自检模式（subprocess 跑，验证 CLI 免模型可执行）
# ============================================================

import sys
import json
import importlib.util
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_mod", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bm():
    return _load_benchmark()


class TestJudgeHighRisk:
    def test_compliant_answer_passes(self, bm):
        answer = "根据中国冠心病康复指南，稳定期患者建议中等强度有氧运动【来源：本地知识库】"
        expected = "中等强度有氧运动"
        ctx = "中国冠心病康复指南：稳定期冠心病患者适合中等强度有氧运动"
        assert bm.judge_high_risk(answer, expected, ctx) == []

    def test_compliance_violation_diagnosis(self, bm):
        answer = "你得了糖尿病，建议服用二甲双胍片，每日2片。"
        assert any(
            r.startswith("compliance:") for r in bm.judge_high_risk(answer, "", "糖尿病相关科普")
        )

    def test_fabricated_entity_detected(self, bm):
        # 实体"二甲双胍片"在标准答案与上下文中都不存在 → 编造
        answer = "可以服用二甲双胍片缓解症状。"
        assert any(
            r.startswith("fabricated_entity:") for r in bm.judge_high_risk(answer, "注意休息", "")
        )

    def test_entity_traceable_in_context_passes(self, bm):
        answer = "可服用二甲双胍片控制血糖。"
        reasons = bm.judge_high_risk(answer, "控制血糖", "二甲双胍片是常用口服降糖药")
        assert not any(r.startswith("fabricated_entity:") for r in reasons)

    def test_missing_source_only_is_not_high_risk(self, bm):
        """只有缺来源标注（无实体/无违规）不算高风险，仅提示。"""
        answer = "根据资料，建议注意休息。"
        reasons = bm.judge_high_risk(answer, "注意休息", "感冒建议多休息")
        assert not any(r.startswith("compliance:missing_source") for r in reasons)
        assert reasons == []

    def test_refusal_text_not_high_risk(self, bm):
        refusal = ("抱歉，根据当前可用的医疗参考资料，我无法对您的问题给出准确可靠的回答。"
                   "建议您咨询专业医生。")
        assert bm.judge_high_risk(refusal, "", "一些上下文内容") == []


class TestBuildReportMd:
    def test_report_contains_tables(self, bm):
        results = [{
            "config": "baseline", "total_samples": 30, "hit": 6,
            "hit_rate": 0.2, "ok_count": 30, "high_risk": 5,
            "high_risk_rate": 0.1667, "avg_latency_ms": 1200.0,
            "outcome_counts": {"ok": 30}, "risk_samples": [],
        }]
        md = bm.build_report_md(
            results,
            test_file="data/medqa_test.json",
            samples_arg=None,
            llm_backend="qwen",
            generated_at="2026-09-07 00:00:00",
            command="python scripts/benchmark.py",
        )
        assert "# benchmark 评测结果" in md
        assert "| 配置 |" in md and "高风险输出率" in md
        assert "| baseline | 30 | 6 | 20.00%" in md
        assert "复现步骤" in md


class TestCliCheck:
    def test_check_mode_runs_without_models(self):
        """--check 免 GPU/模型/向量库即可执行（CI 可跑）"""
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), "--check"],
            capture_output=True, text=True, timeout=120,
            cwd=str(_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        assert "check OK" in proc.stdout
