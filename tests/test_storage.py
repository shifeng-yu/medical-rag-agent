# ============================================================
# 存储层与 SessionManager 测试
#
# 覆盖：
#   - 会话采纳客户端 id / 空值与超长值换发
#   - 消息落库、turn_count / title 维护、来源随消息落库
#   - 会话回放（消息 + 来源）与级联删除
#   - 摘要落库（summary_upto 对齐）与上下文压缩
#   - 指标行、反馈覆盖写、反馈统计口径
#   - 任务状态跨请求可见（进程内不再保存）
# 数据库由 tests/conftest.py 统一指向临时 SQLite，缓存默认关闭。
# ============================================================

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.logger import RequestMetrics
from src.session.manager import session_manager
from src.storage import repository


def _sid(prefix: str = "s") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TestSessionStore:
    def test_adopts_client_session_id(self):
        """客户端给的 id 直接采纳（重启/换实例后可续聊）。"""
        sid = _sid()
        assert session_manager.get_or_create_session(sid) == sid
        assert repository.get_session(sid)["session_id"] == sid

    def test_empty_and_overlong_id_reissued(self):
        empty = session_manager.get_or_create_session("")
        assert len(empty) == 32

        long_id = "x" * 100
        assert session_manager.get_or_create_session(long_id) != long_id

    def test_add_message_returns_id_and_maintains_session(self):
        sid = _sid()
        session_manager.get_or_create_session(sid)

        user_mid = session_manager.add_message(sid, "user", "最近总是头痛")
        assert isinstance(user_mid, int) and user_mid > 0

        assistant_mid = session_manager.add_message(
            sid, "assistant", "建议规律作息，必要时就医。",
            outcome="ok",
            judge_result={"layer": "rule_only", "scores": {"factual": 9}},
            latency_ms=12.5,
            first_token_ms=180.0,
            stream_mode="token",
            sources=[{
                "title": "头痛指南", "source": "local_kb", "score": 0.9,
                "doc_id": "headache", "chunk_id": "abc123def456",
                "snippet": "原文片段", "department": "神经内科",
            }],
        )
        assert assistant_mid > user_mid

        session = repository.get_session(sid)
        assert session["turn_count"] == 1
        assert session["title"] == "最近总是头痛"

        detail = repository.session_detail(sid)
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
        assert detail["messages"][1]["sources"][0]["chunk_id"] == "abc123def456"
        assert detail["messages"][1]["judge_layer"] == "rule_only"

    def test_history_roundtrip_and_delete(self):
        sid = _sid()
        session_manager.get_or_create_session(sid)
        session_manager.add_message(sid, "user", "问题一")
        session_manager.add_message(sid, "assistant", "回答一", outcome="ok")

        history = session_manager.get_history(sid)
        assert [h["role"] for h in history] == ["user", "assistant"]

        session_manager.delete_session(sid)
        assert repository.get_session(sid) is None
        assert session_manager.get_history(sid) == []

    def test_summary_persisted_with_coverage(self):
        """摘要与覆盖条数一起落库：重算依据，避免摘要与历史错位。"""
        sid = _sid()
        session_manager.get_or_create_session(sid)
        repository.update_summary(sid, "患者主诉头痛两月", 4)

        session = repository.get_session(sid)
        assert session["summary"] == "患者主诉头痛两月"
        assert session["summary_upto"] == 4


class TestContextCompression:
    def test_compress_after_threshold(self, monkeypatch):
        """>4 轮触发压缩：摘要 + 近 4 轮 + 当前问题。"""
        import src.session.manager as manager_mod

        class _FakeLLM:
            def complete(self, task, prompt):
                assert task == "summary"
                return "患者反复头痛，已建议就医"

        monkeypatch.setattr(manager_mod, "get_llm", lambda: _FakeLLM())

        sid = _sid()
        session_manager.get_or_create_session(sid)
        for i in range(5):
            session_manager.add_message(sid, "user", f"第{i}次：头痛")
            session_manager.add_message(sid, "assistant", f"第{i}次：建议休息", outcome="ok")

        context, tokens = session_manager.build_context(sid, "还是要痛怎么办")

        assert "历史对话摘要" in context
        assert "患者反复头痛" in context
        assert "最近4轮的完整对话" in context
        assert "还是要痛怎么办" in context
        assert tokens > 0
        # 摘要已落库，且覆盖到 split_idx = (5-4)*2 = 2 条
        assert repository.get_session(sid)["summary_upto"] == 2


class TestMetricsAndFeedback:
    def test_record_request_marks_success(self):
        sid = _sid()
        metrics = RequestMetrics(sid)
        metrics.query = "头痛"
        metrics.outcome = "ok"
        metrics.success = True
        metrics.stream_mode = "token"
        metrics.first_token_ms = 120.0
        repository.record_request(metrics)

        # 直接查库确认落了一行（通过 ORM 会话，避免额外仓储接口）
        from sqlalchemy import select
        from src.storage.db import session_scope
        from src.storage.models import ChatRequest

        with session_scope() as s:
            rows = list(s.scalars(
                select(ChatRequest).where(ChatRequest.session_id == sid)
            ).all())
        assert len(rows) == 1
        assert rows[0].success == 1
        assert rows[0].first_token_ms == 120.0

    def test_feedback_overwrites_same_message(self):
        sid = _sid()
        session_manager.get_or_create_session(sid)
        mid = session_manager.add_message(sid, "assistant", "回答", outcome="ok")

        first = repository.save_feedback(mid, sid, -1, ["引用不相关"], "来源不对")
        second = repository.save_feedback(mid, sid, 1, [], "其实还行")

        assert first["id"] == second["id"], "同一条回答只保留一条反馈"
        assert second["rating"] == 1

    def test_feedback_stats_shape(self):
        sid = _sid()
        session_manager.get_or_create_session(sid)
        mid = session_manager.add_message(sid, "assistant", "回答", outcome="ok")
        repository.save_feedback(mid, sid, -1, ["答非所问"], "")

        stats = repository.feedback_stats(days=7)
        assert set(stats) == {
            "days", "total", "positive", "negative",
            "positive_rate", "top_reasons", "by_day",
        }
        assert stats["total"] >= 1
        assert any(r["tag"] == "答非所问" for r in stats["top_reasons"])


class TestTaskStore:
    def test_task_lifecycle_visible_across_instances(self):
        """任务状态在库里：API 与 worker 不共享内存也能读到同一份。"""
        task_id = uuid.uuid4().hex[:12]
        repository.create_task(
            task_id, "rebuild_doc", "local_kb",
            department="心内科", doc_id="冠心病", files=1, executor="celery",
        )
        task = repository.get_task(task_id)
        assert task["status"] == "processing"
        assert task["kind"] == "rebuild_doc"
        assert task["executor"] == "celery"

        repository.update_task(
            task_id, status="done", chunks=12, failed=0, errors=[],
        )
        task = repository.get_task(task_id)
        assert task["chunks"] == 12
        assert task["status"] == "done"

        listed = repository.list_tasks(limit=10, collection="local_kb")
        assert any(t["task_id"] == task_id for t in listed)


class TestWorkflowPersistence:
    """主链路（节点打桩）→ 收银台 → MySQL。

    前面几组测的是仓储与 SessionManager；这一组验证"真的接上了"：
    跑一次 workflow.run，用户消息、助手消息、来源、回合指标、message_id
    都必须出现在库里。
    """

    def test_run_writes_message_sources_and_request(self, monkeypatch):
        import asyncio

        import src.core.workflow as wf

        async def fake_retrieve(query, classification):
            return ([{
                "title": "失眠指南", "source": "local_kb", "score": 0.8,
                "doc_id": "insomnia", "content": "失眠与作息、压力相关。",
            }], [], 2.0)

        def fake_rerank(query, local, pubmed, source_weights=None):
            return [{
                "title": "失眠指南", "source": "local_kb", "rerank_score": 0.88,
                "doc_id": "insomnia", "department": "神经内科",
                "content": "失眠与作息、压力相关。建议规律作息。",
            }]

        def fake_generate(**kwargs):
            return "建议规律作息，必要时就医。", 8.0

        def fake_validate(**kwargs):
            return True, {
                "layer": "both", "scores": {"factual_consistency": 9},
            }, ""

        monkeypatch.setattr(wf.retriever, "retrieve", fake_retrieve)
        monkeypatch.setattr(wf.reranker, "rerank", fake_rerank)
        monkeypatch.setattr(wf.generator, "generate", fake_generate)
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
        monkeypatch.setattr(wf.classifier, "classify", lambda q: "local")
        monkeypatch.setattr(
            wf.classifier, "should_trigger_high_judge", lambda q: False
        )
        monkeypatch.setattr(
            wf.classifier, "get_source_weight", lambda q, c: (0.7, 0.3)
        )

        result = asyncio.run(wf.workflow.run("最近总是失眠怎么办"))
        sid = result["session_id"]

        assert result["outcome"] == "ok"
        assert result["message_id"].isdigit(), result["message_id"]

        detail = repository.session_detail(sid)
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]

        assistant = detail["messages"][1]
        assert assistant["id"] == int(result["message_id"])
        assert assistant["outcome"] == "ok"
        assert assistant["judge_layer"] == "both"
        assert assistant["sources"][0]["doc_id"] == "insomnia"
        assert assistant["sources"][0]["snippet"].startswith("失眠")
        assert detail["turn_count"] == 1
        assert detail["last_outcome"] == "ok"

        # 回合指标同样落库（含 ok 出口）
        from sqlalchemy import select

        from src.storage.db import session_scope
        from src.storage.models import ChatRequest

        with session_scope() as s:
            rows = list(s.scalars(
                select(ChatRequest).where(ChatRequest.session_id == sid)
            ).all())
        assert len(rows) == 1
        assert rows[0].outcome == "ok"
        assert rows[0].success == 1

    def test_degraded_exit_persists_judge_layer(self, monkeypatch):
        """非 ok 出口也要把归一化后的 layer 落库（否则回放看不到降级原因）。"""
        import asyncio

        import src.core.workflow as wf
        from config.settings import settings

        async def empty_retrieve(query, classification):
            return [], [], 0.0

        monkeypatch.setattr(settings, "max_retry_tool_call", 0)
        monkeypatch.setattr(wf.retriever, "retrieve", empty_retrieve)
        monkeypatch.setattr(wf.generator, "generate_degraded", lambda q: "检索暂不可用")
        monkeypatch.setattr(
            wf.safety_filter, "check_emergency", lambda q: (False, None, None)
        )
        monkeypatch.setattr(
            wf.safety_filter, "check_input_safety", lambda q: (False, None, None)
        )
        monkeypatch.setattr(wf.classifier, "classify", lambda q: "local")
        monkeypatch.setattr(
            wf.classifier, "should_trigger_high_judge", lambda q: False
        )

        result = asyncio.run(wf.workflow.run("最近总失眠怎么办"))
        assert result["outcome"] == "degraded"
        assert result["judge_result"]["layer"] == "degraded"

        detail = repository.session_detail(result["session_id"])
        assistant = detail["messages"][1]
        assert assistant["outcome"] == "degraded"
        assert assistant["judge_layer"] == "degraded"
        assert detail["last_outcome"] == "degraded"
