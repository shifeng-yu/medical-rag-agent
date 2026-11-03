# ============================================================
# 流式问诊测试（协议 / 风险分级 / 事件序列 / 与阻塞链路一致性）
#
# 全部用桩替换检索、重排、生成、校验：不加载模型、不连向量库，
# 也不写真实会话与指标（CI 无 GPU 可全量跑）。
# ============================================================

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.core import stream_protocol as sp


# ---------------- 协议层 ----------------


class TestProtocol:
    def test_frame_keeps_data_single_line(self):
        """正文里的换行必须转义，不能把一帧劈成两帧。"""
        frame = sp.sse_frame(sp.ev(sp.EVENT_DELTA, text="第一行\n第二行"))
        lines = frame.split("\n")
        assert lines[0] == "event: delta"
        assert lines[1].startswith("data: ")
        assert len(lines) == 4  # event / data / 空行 / 尾部空串
        assert "\\n" in lines[1]

    def test_unknown_event_rejected(self):
        with pytest.raises(ValueError):
            sp.ev("no_such_event")

    def test_split_text(self):
        assert sp.split_text("", 10) == []
        assert sp.split_text("短句", 10) == ["短句"]
        assert sp.split_text("abcdefghij", 4) == ["abcd", "efgh", "ij"]
        assert sp.split_text("abcdef", 0) == ["abcdef"]  # size<=0 不切片

    def test_assess_risk_dose_is_high(self):
        assert sp.assess_risk("布洛芬一次吃几片") == sp.RISK_HIGH

    def test_assess_risk_pregnancy_is_high(self):
        assert sp.assess_risk("孕妇感冒了能吃什么") == sp.RISK_HIGH

    def test_assess_risk_general_question_is_low(self):
        """泛问不缓冲：否则绝大多数真实问诊都退化成非流式。"""
        assert sp.assess_risk("最近总是失眠怎么办") == sp.RISK_LOW

    def test_assess_risk_empty_is_high(self):
        assert sp.assess_risk("") == sp.RISK_HIGH


# ---------------- 桩 ----------------


@pytest.fixture
def stubs(monkeypatch):
    """把五个节点全部替换成确定性桩，保留真实编排逻辑。"""
    import src.core.workflow as wf

    async def fake_retrieve(query, classification):
        return ([{
            "title": "本地指南", "source": "local_kb",
            "content": "感冒是自限性疾病，多喝水休息。" * 100,
            "doc_id": "respiratory", "score": 0.8,
        }], [], 3.0)

    def fake_rerank(query, local, pubmed, source_weights=None):
        return [{
            "title": "本地指南", "source": "local_kb", "rerank_score": 0.91,
            "publish_time": "2024", "department": "呼吸科",
            "doc_id": "respiratory",
            "content": "感冒是自限性疾病，多喝水休息。" * 100,
        }]

    def fake_generate(**kwargs):
        return "第二次回答。", 5.0

    def fake_generate_stream(**kwargs):
        for piece in ["这是", "流式", "回答。"]:
            yield piece

    def fake_validate(**kwargs):
        return True, {
            "layer": "both",
            "scores": {"factual_consistency": 9},
        }, ""

    monkeypatch.setattr(wf.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(wf.retriever, "expand_query", lambda q: q + " 机制")
    monkeypatch.setattr(wf.reranker, "rerank", fake_rerank)
    monkeypatch.setattr(wf.generator, "generate", fake_generate)
    monkeypatch.setattr(wf.generator, "generate_stream", fake_generate_stream)
    monkeypatch.setattr(wf.generator, "generate_degraded", lambda q: "降级回答")
    monkeypatch.setattr(wf.judge, "validate", fake_validate)
    monkeypatch.setattr(
        wf.safety_filter, "check_emergency", lambda q: (False, None, None)
    )
    monkeypatch.setattr(
        wf.safety_filter, "check_input_safety", lambda q: (False, None, None)
    )
    monkeypatch.setattr(
        wf.safety_filter, "check_output_compliance", lambda a, r: (True, [])
    )
    monkeypatch.setattr(
        wf.safety_filter, "get_refusal_response", lambda k: "标准拒答"
    )
    monkeypatch.setattr(wf.classifier, "classify", lambda q: "local")
    monkeypatch.setattr(wf.classifier, "should_trigger_high_judge", lambda q: False)
    monkeypatch.setattr(
        wf.classifier, "get_source_weight", lambda q, c: (0.7, 0.3)
    )
    monkeypatch.setattr(wf.session_manager, "add_message", lambda *a, **k: None)
    monkeypatch.setattr(wf.outcome.request_logger, "log_request", lambda m: None)
    return wf


def collect(query, **kwargs):
    """跑一次流式问诊，收齐事件。"""
    import src.core.workflow as wf

    async def _collect():
        return [event async for event in wf.workflow.stream(query, **kwargs)]

    return asyncio.run(_collect())


def names(events):
    return [e["event"] for e in events]


def first_index(events, name):
    return names(events).index(name)


# ---------------- 事件序列 ----------------


class TestStreamSequence:
    def test_low_risk_streams_before_judge(self, stubs):
        """低风险：delta 出现在 judge 之前（边生成边推）。"""
        events = collect("最近总是失眠怎么办")

        assert names(events)[0] == sp.EVENT_META
        assert first_index(events, sp.EVENT_DELTA) < first_index(events, sp.EVENT_JUDGE)
        assert names(events)[-1] == sp.EVENT_FINAL

        final = events[-1]["data"]
        assert set(final.keys()) == {
            "answer", "sources", "judge_result", "latency_ms",
            "session_id", "outcome", "message_id",
        }
        assert final["answer"] == "这是流式回答。"
        assert final["outcome"] == "ok"

    def test_sources_arrive_before_generation(self, stubs):
        """来源先于正文：前端右侧溯源栏可在等待生成时就绪。"""
        events = collect("最近总是失眠怎么办")
        src_idx = first_index(events, sp.EVENT_SOURCES)
        assert src_idx < first_index(events, sp.EVENT_DELTA)

        sources = events[src_idx]["data"]["sources"]
        assert sources[0]["doc_id"] == "respiratory"
        assert len(sources[0]["chunk_id"]) == 12
        # 原文片段截断到 1000 字，不透传整篇
        assert len(sources[0]["snippet"]) == 1000

    def test_high_risk_buffers_until_judge(self, stubs):
        """高风险：judge 之前不得出现任何 delta（先校验后回放）。"""
        events = collect("布洛芬一次吃几片")

        judge_idx = first_index(events, sp.EVENT_JUDGE)
        assert not [
            e for e in events[:judge_idx] if e["event"] == sp.EVENT_DELTA
        ]
        # 校验通过后回放，内容完整
        deltas = [e["data"]["text"] for e in events if e["event"] == sp.EVENT_DELTA]
        assert "".join(deltas) == "第二次回答。"

    def test_risk_event_reports_mode(self, stubs):
        events = collect("布洛芬一次吃几片")
        risk = [e for e in events if e["event"] == sp.EVENT_RISK][0]["data"]
        assert risk == {"level": "high", "mode": "buffered"}

    def test_judge_retry_emits_replace(self, stubs, monkeypatch):
        """校验不过重生成：已下发的旧答案必须被 replace 整段替换。"""
        import src.core.workflow as wf

        calls = {"n": 0}

        def flaky_validate(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return False, {"layer": "judge", "scores": {"fact": 3}}, "不通过"
            return True, {"layer": "both", "scores": {"fact": 9}}, ""

        monkeypatch.setattr(wf.judge, "validate", flaky_validate)
        events = collect("最近总是失眠怎么办")

        replaces = [e for e in events if e["event"] == sp.EVENT_REPLACE]
        assert len(replaces) == 1
        assert replaces[0]["data"]["text"] == "第二次回答。"
        assert replaces[0]["data"]["reason"] == "judge_retry"
        assert events[-1]["data"]["answer"] == "第二次回答。"
        assert calls["n"] == 2

    def test_emergency_exit_has_no_sources(self, stubs, monkeypatch):
        import src.core.workflow as wf

        monkeypatch.setattr(
            wf.safety_filter, "check_emergency",
            lambda q: (True, "疑似心梗", "请立即前往急诊"),
        )
        events = collect("胸痛半小时")

        assert sp.EVENT_SOURCES not in names(events)
        final = events[-1]["data"]
        assert final["outcome"] == "emergency_blocked"
        assert final["answer"] == "请立即前往急诊"

    def test_run_and_stream_share_final(self, stubs):
        """阻塞接口与流式接口产出同一份 final（同一条链）。"""
        import src.core.workflow as wf

        async def _run():
            return await wf.workflow.run("最近总是失眠怎么办")

        result = asyncio.run(_run())
        # session_manager 只复用"已存在"的 id，故用首轮返回的 sid 复跑
        sid = result["session_id"]
        events = collect(
            "最近总是失眠怎么办", session_id=sid, emit_tokens=False,
        )
        final = events[-1]["data"]

        assert result["answer"] == final["answer"] == "第二次回答。"
        assert result["judge_result"] == final["judge_result"]
        assert final["session_id"] == sid

    def test_no_tokens_when_disabled(self, stubs):
        """评测/批量模式：不产出 delta（避免无谓的切片与序列化）。"""
        events = collect("最近总是失眠怎么办", emit_tokens=False)
        assert sp.EVENT_DELTA not in names(events)
