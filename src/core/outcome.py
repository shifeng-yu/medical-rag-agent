# ============================================================
# 问诊结果出口模块（收银台）
#
# 一次问诊回合如何收尾的全部知识集中于此：
#   - 出口（kind）词汇表：问诊回合的唯一收尾方式
#   - 写会话规则：哪些出口把答复写入会话历史
#   - 指标口径：outcome 记录出口、success 仅当 ok
#   - 存储：正式答复写 chat_message（含来源），每回合写 chat_request
#   - 返回形状：固定 7 键（answer/sources/judge_result/latency_ms/
#     session_id/outcome/message_id）。outcome 键暴露出口 kind，供评测等
#     下游判断本次回合的出门原因；message_id 供前端提交反馈时定位回答
#   - sources 白名单映射：客户端可见字段在此单一所有
#
# 背景与决策见 docs/adr/0001-workflow-outcome.md；
# 领域词条（问诊回合 / 问诊结果出口 / 出门原因）见 CONTEXT.md。
# ============================================================

import hashlib
import time
from typing import Dict, List, Optional, Any

from loguru import logger

from src.session.manager import session_manager
from src.monitoring.logger import RequestMetrics, request_logger

# 存储层惰性导入：问诊链路不该因为运维侧数据库配置问题而 import 失败。
# （模块级 import 只到 session_manager，真正的 SQL 在调用时才触达。）


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

# 客户端可见字段的唯一来源。9/7 随来源分级扩到 7 字段；
# 流式改造再加 3 个溯源字段：doc_id / chunk_id / snippet。
# snippet 是命中的原文片段（截断），支撑网页右侧"点开引用看原文"，
# 也是反馈定位到具体块的锚点。
SOURCE_FIELDS = (
    "title", "source", "score", "publish_time", "department",
    "authority_tier", "authority_label",
    "doc_id", "chunk_id", "snippet",
)

SNIPPET_MAX_CHARS = 1000


def _chunk_id(item: Dict) -> str:
    """由正文内容派生稳定块 ID。

    没有改 Milvus schema（不加字段、不重建索引），却要一个跨进程稳定的
    块标识供前端定位与反馈归因：取正文 sha1 前 12 位。同一块内容在任何
    后端（Docker Milvus / Lite / 将来 pgvector）下算出的 ID 都一致。
    """
    content = item.get("content", "") or ""
    if not content:
        return ""
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]


def to_sources(reranked: List[Dict], limit: int = 5) -> List[Dict]:
    """从重排结果挑选客户端可见字段。

    形状在此单一所有——workflow / routes 不再各自手选一遍。
    authority_tier / authority_label 由 reranker 在精排时打标
    （knowledge_grader 分级），随来源返回，实现"来源可分级可追溯"。
    snippet 为截断后的原文（≤ SNIPPET_MAX_CHARS 字）：只透出被引用的
    片段，不透传整篇文档与内部字段（向量、score 明细等）。
    """
    out: List[Dict[str, Any]] = []
    for r in (reranked or [])[:limit]:
        content = r.get("content", "") or ""
        out.append({
            "title": r.get("title", ""),
            "source": r.get("source", ""),
            "score": r.get("rerank_score", 0),
            "publish_time": r.get("publish_time", ""),
            "department": r.get("department", ""),
            "authority_tier": r.get("authority_tier", ""),
            "authority_label": r.get("authority_label", ""),
            "doc_id": r.get("doc_id", ""),
            "chunk_id": _chunk_id(r),
            "snippet": content[:SNIPPET_MAX_CHARS],
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


def _persist_turn(
    session_id: str,
    answer: str,
    kind: str,
    sources: Optional[List[Dict]],
    judge_result: Optional[Dict],
    latency_ms: float,
    metrics: RequestMetrics,
) -> Optional[int]:
    """把正式答复写入会话（MySQL 真源，Redis 追加由 SessionManager 负责）。

    落库失败只告警不抛出：数据库抖动不该让用户拿不到已经生成好的答复。
    返回 message_id；失败或无会话时返回 None。
    """
    try:
        return session_manager.add_message(
            session_id, "assistant", answer,
            outcome=kind,
            judge_result=judge_result or {},
            latency_ms=latency_ms,
            first_token_ms=metrics.first_token_ms,
            stream_mode=metrics.stream_mode,
            retry_count=metrics.retry_count,
            classification=metrics.classification_label,
            sources=sources or [],
        )
    except Exception as e:  # noqa: BLE001 - 存储失败不影响问诊结果
        logger.error(f"会话落库失败（回复照常返回）: {e}")
        return None


def _persist_metrics(metrics: RequestMetrics) -> None:
    """写一行回合指标（含拦截/失败出口），供 SQL 侧统计成功率。"""
    try:
        from src.storage import repository

        repository.record_request(metrics)
    except Exception as e:  # noqa: BLE001
        logger.error(f"指标落库失败（不影响回复）: {e}")


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
    - 存储：正式答复写 chat_message（含来源），每回合写一行 chat_request
    - 上报：每次出口恰好 log_request 一次（拦截类出口不再漏统计）
    - 返回：固定 7 键形状（6 键 + message_id），outcome 携带出口 kind，
      message_id 供前端提交反馈时定位回答
    """
    if kind not in ALL_EXITS:
        raise ValueError(f"未知的出口 kind: {kind!r}")

    metrics.outcome = kind
    metrics.success = (kind == EXIT_OK)
    metrics.final_output = answer
    if error_message:
        metrics.error_message = error_message

    latency_ms = round((time.time() - metrics.start_time) * 1000, 2)

    # 先归一化再落库：degraded / emergency_blocked 等出口没有打分，
    # layer 默认取出口 kind。若把归一化留到返回值，库里会存成空串，
    # 会话回放就看不到"这次是降级出口"。
    normalized_judge = _normalize_judge_result(kind, judge_result)

    should_write = (write_session if write_session is not None
                    else kind in SESSION_WRITE_EXITS)
    message_id = None
    if should_write and answer:
        message_id = _persist_turn(
            session_id, answer, kind, sources, normalized_judge,
            latency_ms, metrics,
        )

    # 评测模式（write_session=False）不写回合指标：消融/benchmark 跑几十上百条，
    # 混进线上统计会让成功率与延迟口径失真。JSONL 仍记一份，便于事后核对。
    if write_session is not False:
        _persist_metrics(metrics)
    request_logger.log_request(metrics)

    return {
        "answer": answer,
        "sources": sources or [],
        "judge_result": normalized_judge,
        "latency_ms": latency_ms,
        "session_id": session_id,
        "outcome": kind,
        # 统一成字符串：pydantic 的 str 字段不做 int→str 隐式转换，
        # 返回 int 会让 QueryResponse 在路由层直接校验失败
        "message_id": str(message_id) if message_id else "",
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
