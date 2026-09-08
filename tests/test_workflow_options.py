# ============================================================
# workflow RunOptions 环节开关测试
# 覆盖: 关闭 judge / reranker / session 时主链路的实际行为
#       以及 LLM_BACKEND=fake 时整条链路不依赖模型跑通（smoke）
# ============================================================

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.core.workflow import workflow, RunOptions
from src.session.manager import session_manager
from src.monitoring.logger import request_logger


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


def _stub_nodes(monkeypatch, *, with_results=True):
    """顶替安全/分类/检索等节点，让 run() 能走到 generate 阶段"""
    import src.core.workflow as wf

    monkeypatch.setattr(wf.safety_filter, "check_emergency", lambda q: (False, None, None))
    monkeypatch.setattr(wf.safety_filter, "check_input_safety", lambda q: (False, None, None))
    monkeypatch.setattr(wf.safety_filter, "check_output_compliance", lambda a, r: (True, []))
    monkeypatch.setattr(wf.classifier, "classify", lambda q: "local")
    monkeypatch.setattr(wf.classifier, "should_trigger_high_judge", lambda q: True)
    monkeypatch.setattr(wf.classifier, "get_source_weight", lambda q, c: (0.7, 0.3))
    monkeypatch.setattr(wf.knowledge_grader, "normalize_query", lambda q: q)

    async def fake_retrieve(query, classification):
        if with_results:
            return [{
                "title": "指南", "source": "local_kb",
                "content": "感冒是自限性疾病，多喝水休息可缓解。",
                "score": 0.8,
            }], [], 0.0
        return [], [], 0.0

    monkeypatch.setattr(wf.retriever, "retrieve", fake_retrieve)
    return wf


def _probe(monkeypatch, **stubs):
    """构造计数器探针（返回 {名字: {"n": n}}），stubs 为 name→callable"""
    counters = {name: {"n": 0} for name in stubs}

    def make(name, fn):
        def wrapped(*args, **kwargs):
            counters[name]["n"] += 1
            return fn(*args, **kwargs)
        return wrapped

    for name, fn in stubs.items():
        monkeypatch.setattr(_probe_target(name), name, make(name, fn))
    return counters


def _probe_target(name):
    """探针打点的目标对象（模块级单例）"""
    import src.core.workflow as wf
    return {
        "rerank": wf.reranker,
        "validate": wf.judge,
        "generate": wf.generator,
        "get_or_create_session": session_manager,
    }[name]


class TestRunOptionsToggles:
    """环节开关行为探针"""

    def test_default_options_equal_none(self, monkeypatch, capture_effects):
        """不传 options 与显式 RunOptions() 行为一致（在线路径零变化）"""
        wf = _stub_nodes(monkeypatch)
        counters = _probe(
            monkeypatch,
            rerank=lambda q, l, p, source_weights=None: [
                {"title": "t", "source": "local_kb", "content": "c", "rerank_score": 0.5}],
            validate=lambda **kw: (True, {"layer": "both"}, ""),
            generate=lambda **kw: ("正常答复【来源：知识库】", 5.0),
        )

        r1 = asyncio.run(workflow.run("感冒怎么办"))
        r2 = asyncio.run(workflow.run("感冒怎么办", options=RunOptions()))

        # session_id / latency_ms 天然逐次不同，其余字段与出口应完全一致
        def _stable(r):
            return {k: v for k, v in r.items() if k not in ("session_id", "latency_ms")}

        assert _stable(r1) == _stable(r2)
        assert r1["outcome"] == r2["outcome"] == "ok"
        assert counters["rerank"]["n"] == 2
        assert counters["validate"]["n"] == 2

    def test_judge_off_skips_validate_and_marks_layer(self, monkeypatch, capture_effects):
        """use_judge=False: judge.validate 不被调用，judge_result 占位 ablation_off"""
        wf = _stub_nodes(monkeypatch)
        counters = _probe(
            monkeypatch,
            rerank=lambda q, l, p, source_weights=None: [
                {"title": "t", "source": "local_kb", "content": "c", "rerank_score": 0.5}],
            validate=lambda **kw: (True, {"layer": "both"}, ""),
            generate=lambda **kw: ("正常答复【来源：知识库】", 5.0),
        )

        result = asyncio.run(workflow.run(
            "感冒怎么办", options=RunOptions(use_judge=False),
        ))

        assert counters["validate"]["n"] == 0       # 裁判完全跳过
        assert result["judge_result"] == {"layer": "ablation_off"}
        assert result["outcome"] == "ok"

    def test_reranker_off_skips_rerank(self, monkeypatch, capture_effects):
        """use_reranker=False: rerank 不被调用，走原始拼接"""
        wf = _stub_nodes(monkeypatch)
        counters = _probe(
            monkeypatch,
            rerank=lambda q, l, p, source_weights=None: [],
            validate=lambda **kw: (True, {"layer": "both"}, ""),
            generate=lambda **kw: ("正常答复【来源：知识库】", 5.0),
        )

        result = asyncio.run(workflow.run(
            "感冒怎么办", options=RunOptions(use_reranker=False),
        ))

        assert counters["rerank"]["n"] == 0
        assert result["outcome"] == "ok"
        assert result["judge_result"]["layer"] == "both"   # judge 仍可单独保留

    def test_session_off_no_history_write(self, monkeypatch, capture_effects):
        """use_session=False（评测模式）: 不建会话、不写历史，答复也不写"""
        wf = _stub_nodes(monkeypatch)
        counters = _probe(
            monkeypatch,
            get_or_create_session=lambda sid=None: "created-sid",
            rerank=lambda q, l, p, source_weights=None: [
                {"title": "t", "source": "local_kb", "content": "c"}],
            validate=lambda **kw: (True, {"layer": "both"}, ""),
            generate=lambda **kw: ("正常答复【来源：知识库】", 5.0),
        )
        written, logged = capture_effects

        result = asyncio.run(workflow.run(
            "感冒怎么办", session_id="eval-1",
            options=RunOptions(use_session=False),
        ))

        assert counters["get_or_create_session"]["n"] == 0   # 不建会话
        assert not any(a[0] == "eval-1" for a in written)    # 不写任何历史
        assert result["outcome"] == "ok"
        assert result["session_id"] == "eval-1"
        assert len(logged) == 1                              # 指标照常上报

    def test_degraded_exit_in_eval_mode(self, monkeypatch, capture_effects):
        """评测模式下检索全空 → degraded 出口，不写会话"""
        wf = _stub_nodes(monkeypatch, with_results=False)
        monkeypatch.setattr(wf.generator, "generate_degraded", lambda q: "降级答复")

        written, _ = capture_effects
        result = asyncio.run(workflow.run(
            "罕见病", options=RunOptions(use_session=False),
        ))

        assert result["outcome"] == "degraded"
        assert not any(a[1] == "assistant" for a in written)


class TestFakeBackendSmoke:
    """LLM_BACKEND=fake：整条链路不加载真实模型即可跑到 ok 出口"""

    def test_end_to_end_ok_with_fake_llm(self, monkeypatch, capture_effects):
        import src.core.llm as llm_mod
        # 总机切到假后端（重置缓存单例）
        monkeypatch.setattr(llm_mod.settings, "llm_backend", "fake")
        monkeypatch.setattr(llm_mod, "_llm_instance", None)

        wf = _stub_nodes(monkeypatch)
        monkeypatch.setattr(wf.reranker, "rerank", lambda q, l, p, source_weights=None: [
            {"title": "指南", "source": "local_kb", "content": "感冒是自限性疾病。"}])
        monkeypatch.setattr(wf.judge, "validate", lambda **kw: (True, {"layer": "both"}, ""))
        # generator.generate 保持真身 → 内部经 get_llm() 打到 FakeLLM

        result = asyncio.run(workflow.run("最近总失眠怎么办"))

        assert result["outcome"] == "ok"
        assert "来源" in result["answer"] or "知识库" in result["answer"]
        assert result["session_id"]
        # 证明总机确实在用假后端（未走真实模型）
        assert isinstance(llm_mod.get_llm().backend, llm_mod.FakeLLM)
