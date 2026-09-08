# ============================================================
# 问诊结果出口模块（收银台）
#
# 一次问诊回合如何收尾的全部知识集中于此：
#   - 出口（kind）词汇表：问诊回合的唯一收尾方式
#   - 写会话规则：哪些出口把答复写入会话历史
#   - 指标口径：outcome 记录出口、success 仅当 ok
#   - 返回形状：固定 6 键（answer/sources/judge_result/latency_ms/session_id/outcome）
#     outcome 键暴露出口 kind，供评测等下游判断本次回合的出门原因
#   - sources 白名单映射：客户端可见字段在此单一所有
#
# 背景与决策见 docs/adr/0001-workflow-outcome.md；
# 领域词条（问诊回合 / 问诊结果出口 / 出门原因）见 CONTEXT.md。
# ============================================================

import time
from typing import Dict, List, Optional, Any

from src.session.manager import session_manager
from src.monitoring.logger import RequestMetrics, request_logger


# ---------------- 出口词汇表 ----------------

EXIT_OK = "ok"                                  # 链路完整跑通
EXIT_EMERGENCY_BLOCKED = "emergency_blocked"    # 急症拦截
EXIT_CONTENT_BLOCKED = "content_blocked"        # 敏感内容拦截
EXIT_DEGRADED = "degraded"                      # 检索不可用 → 降级回答
EXIT_ERROR = "error"                            # 工作流内部异常
EXIT_OVERLOAD = "overload"                      # API 层过载
EXIT_TIMEOUT = "timeout"                        # API 层超时

ALL_EXITS = frozenset({
    EXIT_OK, EXIT_EMERGENCY_BLOCKED, EXIT_CONTENT_BLOCKED,
    EXIT_DEGRADED, EXIT_ERROR, EXIT_OVERLOAD, EXIT_TIMEOUT,
})

# 写会话规则：只有"面向用户的正式答复"才写入会话历史。
# 系统失败出口（error / overload / timeout）一律不写，避免污染多轮上下文
# （已知副作用：失败回合会在会话里留下孤儿的用户消息，见 ADR-0001）。
SESSION_WRITE_EXITS = frozenset({
    EXIT_OK, EXIT_EMERGENCY_BLOCKED, EXIT_CONTENT_BLOCKED, EXIT_DEGRADED,
})


# ---------------- sources 白名单 ----------------

SOURCE_FIELDS = (
    "title", "source", "score", "publish_time", "department",
    "authority_tier", "authority_label",
)


def to_sources(reranked: List[Dict], limit: int = 5) -> List[Dict]:
    """从重排结果挑选客户端可见字段。

    形状在此单一所有——workflow / routes 不再各自手选一遍。
    authority_tier / authority_label 由 reranker 在精排时打标
    （knowledge_grader 分级），随来源返回，实现"来源可分级可追溯"。
    """
    out: List[Dict[str, Any]] = []
    for r in (reranked or [])[:limit]:
        out.append({
            "title": r.get("title", ""),
            "source": r.get("source", ""),
            "score": r.get("rerank_score", 0),
            "publish_time": r.get("publish_time", ""),
            "department": r.get("department", ""),
            "authority_tier": r.get("authority_tier", ""),
            "authority_label": r.get("authority_label", ""),
        })
    return out


# ---------------- 收银台 ----------------

def _normalize_judge_result(kind: str, judge_result: Optional[Dict]) -> Dict:
    """judge_result 固定带 layer；调用方给了 layer 则尊重，否则默认 = 出口 kind。"""
    if judge_result is None:
        return {"layer": kind}
    if "layer" not in judge_result:
        return {**judge_result, "layer": kind}
    return judge_result


def finish(
    session_id: str,
    metrics: RequestMetrics,
    *,
    kind: str,
    answer: str,
    sources: Optional[List[Dict]] = None,
    judge_result: Optional[Dict] = None,
    error_message: str = "",
    write_session: Optional[bool] = None,
) -> Dict:
    """统一结账口：workflow 每个出口 / API 层降级都汇聚到这里。

    - 会话写入：默认按 kind 规则（SESSION_WRITE_EXITS），write_session 可显式覆盖
    - 指标：outcome 记出口；success 仅当 kind == ok；error_message 透传
    - 上报：每次出口恰好 log_request 一次（拦截类出口不再漏统计）
    - 返回：固定 6 键形状，outcome 键携带出口 kind
    """
    if kind not in ALL_EXITS:
        raise ValueError(f"未知的出口 kind: {kind!r}")

    metrics.outcome = kind
    metrics.success = (kind == EXIT_OK)
    metrics.final_output = answer
    if error_message:
        metrics.error_message = error_message

    should_write = (write_session if write_session is not None
                    else kind in SESSION_WRITE_EXITS)
    if should_write and answer:
        session_manager.add_message(session_id, "assistant", answer)

    request_logger.log_request(metrics)

    latency_ms = round((time.time() - metrics.start_time) * 1000, 2)
    return {
        "answer": answer,
        "sources": sources or [],
        "judge_result": _normalize_judge_result(kind, judge_result),
        "latency_ms": latency_ms,
        "session_id": session_id,
        "outcome": kind,
    }


def degrade(session_id: str, kind: str, message: str, reason: str = "") -> Dict:
    """API 层超时/过载降级出口：不写会话、记一次指标、返回固定形状。

    修复历史问题：此类响应此前手拼 dict 且漏 session_id，导致
    /chat 路由读取 result["session_id"] 时 KeyError 崩成 500。
    """
    if kind not in ALL_EXITS:
        raise ValueError(f"未知的出口 kind: {kind!r}")
    metrics = RequestMetrics(session_id)
    return finish(
        session_id, metrics,
        kind=kind, answer=message, error_message=reason or kind,
    )
