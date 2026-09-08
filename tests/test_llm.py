# ============================================================
# LLM 总机契约测试（模型对话收口）
# 覆盖: 任务参数表完整性 / FakeLLM 确定性 / complete 参数分发 /
#       后端接口一致性（真后端与假后端同一约定）
# 说明: 不加载真实模型——QwenBackend 只测接口形状，不触发懒加载
# ============================================================

import sys
import inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.core.llm import (
    LLMService, QwenBackend, FakeLLM, TASKS, get_llm,
)
from src.core.llm import _llm_instance as _CACHED  # noqa: F401  仅确保模块已加载


# ---------------- 任务参数表 ----------------

class TestTaskTable:
    """任务参数表必须覆盖所有业务任务"""

    def test_all_business_tasks_covered(self):
        """5 个业务任务（答题/降级/分类/裁判/摘要）都必须在表里"""
        assert set(TASKS.keys()) >= {
            "answer", "degraded", "classify", "judge", "summary",
        }

    def test_each_spec_has_full_params(self):
        """每个任务都定义了完整的采样参数三元组"""
        for task, spec in TASKS.items():
            assert set(spec.keys()) == {"max_tokens", "temperature", "do_sample"}, task
            assert spec["max_tokens"] > 0, task
            assert isinstance(spec["do_sample"], bool), task

    def test_unknown_task_rejected(self):
        """未注册的任务直接抛错（fail fast，防新调用点漏进表）"""
        llm = LLMService(FakeLLM())
        with pytest.raises(ValueError):
            llm.complete("mystery_task", "hi")


# ---------------- FakeLLM ----------------

class TestFakeLLM:
    """假话务员：确定性 + 按任务返回合理默认"""

    def test_classify_returns_local(self):
        fake = FakeLLM()
        out = fake.generate("classify", "分类：xxx", max_tokens=10,
                            temperature=0.0, do_sample=False)
        assert out == "本地库"          # classifier 解析"文献"→pubmed，否则 local

    def test_judge_returns_high_scores_json(self):
        fake = FakeLLM()
        import json
        raw = fake.generate("judge", "打分", max_tokens=256,
                            temperature=0.0, do_sample=False)
        parsed = json.loads(raw)       # judge.safe_json_parse 可解析
        assert all(parsed[k] >= 8 for k in
                   ("factual_consistency", "logical_coherence", "answer_helpfulness"))

    def test_answer_default_is_compliant_text(self):
        """默认答复：无医疗实体 + 带来源标注 → 能过 judge.rule_check"""
        from src.core.judge import judge
        fake = FakeLLM()
        out = fake.generate("answer", "问题", max_tokens=1024,
                            temperature=0.0, do_sample=False)
        assert "来源" in out or "知识库" in out
        passed, reason, _ = judge.rule_check(out, [{"content": "感冒是自限性疾病。"}])
        assert passed, f"FakeLLM 默认答复应过规则校验: {reason}"

    def test_deterministic(self):
        """同一输入多次调用输出一致"""
        fake = FakeLLM()
        a = fake.generate("answer", "p", max_tokens=100,
                          temperature=0.0, do_sample=False)
        b = fake.generate("answer", "p", max_tokens=100,
                          temperature=0.0, do_sample=False)
        assert a == b

    def test_custom_responses_override(self):
        """responses 定制优先于内置默认"""
        fake = FakeLLM(responses={"answer": "定制答复"})
        out = fake.generate("answer", "p", max_tokens=100,
                            temperature=0.0, do_sample=False)
        assert out == "定制答复"


# ---------------- complete 参数分发 ----------------

class _RecordingBackend:
    """记录每次 generate 收到的参数的探针后端"""

    def __init__(self):
        self.calls = []

    def generate(self, task, prompt, *, max_tokens, temperature, do_sample):
        self.calls.append({
            "task": task, "prompt": prompt,
            "max_tokens": max_tokens, "temperature": temperature,
            "do_sample": do_sample,
        })
        return f"resp-{task}"


class TestCompleteDispatch:
    """complete(task, prompt) 按任务表分发采样参数"""

    @pytest.mark.parametrize("task", sorted(TASKS))
    def test_task_defaults_reach_backend(self, task):
        backend = _RecordingBackend()
        llm = LLMService(backend)
        llm.complete(task, "hello")
        call = backend.calls[-1]
        assert call["task"] == task
        assert call["prompt"] == "hello"
        assert call["max_tokens"] == TASKS[task]["max_tokens"]
        assert call["temperature"] == TASKS[task]["temperature"]
        assert call["do_sample"] == TASKS[task]["do_sample"]

    def test_overrides_win_over_table(self):
        backend = _RecordingBackend()
        llm = LLMService(backend)
        llm.complete("answer", "p", max_tokens=64, temperature=0.7)
        call = backend.calls[-1]
        assert call["max_tokens"] == 64
        assert call["temperature"] == 0.7
        assert call["do_sample"] is False   # 未覆盖的字段沿用任务表

    def test_unknown_override_rejected(self):
        llm = LLMService(FakeLLM())
        with pytest.raises(TypeError):
            llm.complete("answer", "p", bogus_param=1)


# ---------------- 后端接口契约 ----------------

class TestBackendContract:
    """真后端与假后端遵循同一接口约定（不强制继承，靠测试锁死）"""

    def test_signatures_match(self):
        sig_fake = inspect.signature(FakeLLM.generate)
        sig_real = inspect.signature(QwenBackend.generate)
        assert list(sig_fake.parameters) == list(sig_real.parameters), (
            "两后端 generate 签名必须一致（task, prompt, *, max_tokens, "
            "temperature, do_sample）"
        )
        params = list(sig_fake.parameters)
        assert params == ["self", "task", "prompt", "max_tokens",
                          "temperature", "do_sample"]
        # max_tokens/temperature/do_sample 都是 keyword-only
        assert sig_fake.parameters["max_tokens"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig_fake.parameters["do_sample"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_qwen_backend_lazy_no_model_load(self):
        """QwenBackend 构造不触发模型加载（懒加载直到首次 generate）"""
        import src.core.llm as llm_mod
        original = llm_mod.QwenBackend._ensure_loaded
        called = {"n": 0}

        def spy(self):
            called["n"] += 1
            return original(self)

        llm_mod.QwenBackend._ensure_loaded = spy
        try:
            QwenBackend()          # 仅构造，不加载
            assert called["n"] == 0
        finally:
            llm_mod.QwenBackend._ensure_loaded = original


# ---------------- 工厂选择 ----------------

class TestGetLLM:
    """get_llm 按 settings.llm_backend 选择后端"""

    def test_fake_backend_selected(self, monkeypatch):
        import src.core.llm as llm_mod
        monkeypatch.setattr(llm_mod.settings, "llm_backend", "fake")
        monkeypatch.setattr(llm_mod, "_llm_instance", None)
        llm = get_llm()
        assert isinstance(llm.backend, FakeLLM)

    def test_unknown_backend_rejected(self, monkeypatch):
        import src.core.llm as llm_mod
        monkeypatch.setattr(llm_mod.settings, "llm_backend", "alien")
        monkeypatch.setattr(llm_mod, "_llm_instance", None)
        with pytest.raises(ValueError):
            get_llm()
