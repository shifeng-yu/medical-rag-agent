# ============================================================
# 仓储层：所有 SQL 只出现在这里
#
# 约定：
#   - 对外一律返回 dict / 基础类型，不把 ORM 实例交给调用方
#     （会话关闭后 lazy load 会炸，跨层传递容易踩坑）；
#   - 事务边界由 db.session_scope() 负责，本层不 commit；
#   - 时间字段统一序列化为 ISO 字符串（秒精度），与前端/日志一致。
#
# 词条见 CONTEXT.md「会话存储」「反馈」「知识库文档」。
# ============================================================

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from src.storage.db import session_scope
from src.storage.models import (
    ChatFeedback,
    ChatMessage,
    ChatRequest,
    ChatSession,
    ChatSource,
    KbDocument,
    KbTask,
)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat(timespec="seconds") if value else None


# ---------------- 会话 ----------------


def ensure_session(session_id: str, title: str = "") -> Dict[str, Any]:
    """会话不存在则创建，返回会话快照。

    采纳调用方传入的 session_id（本地/重启后可续聊）；同名并发插入时
    由唯一主键兜底，冲突后回读已有行。
    """
    with session_scope() as s:
        row = s.get(ChatSession, session_id)
        if row is None:
            row = ChatSession(session_id=session_id, title=(title or "")[:120])
            s.add(row)
            try:
                s.flush()
            except IntegrityError:
                # 并发首次写入：主键冲突后回读已有行，不覆盖对方数据
                s.rollback()
                row = s.get(ChatSession, session_id)
        return {"session_id": row.session_id, "created_at": _iso(row.created_at)}


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    with session_scope() as s:
        row = s.get(ChatSession, session_id)
        if row is None:
            return None
        return {
            "session_id": row.session_id,
            "title": row.title,
            "turn_count": row.turn_count,
            "summary": row.summary,
            "summary_upto": row.summary_upto,
            "last_outcome": row.last_outcome,
            "created_at": _iso(row.created_at),
            "last_active_at": _iso(row.last_active_at),
        }


def update_summary(session_id: str, summary: str, upto: int) -> None:
    with session_scope() as s:
        row = s.get(ChatSession, session_id)
        if row is not None:
            row.summary = summary
            row.summary_upto = upto


def append_message(
    session_id: str,
    role: str,
    content: str,
    *,
    outcome: str = "",
    judge_result: Optional[Dict] = None,
    latency_ms: float = 0.0,
    first_token_ms: float = 0.0,
    stream_mode: str = "",
    retry_count: int = 0,
    classification: str = "",
    sources: Optional[List[Dict]] = None,
) -> int:
    """写一条消息（assistant 行顺带写来源），返回 message_id。

    来源只存白名单字段：与 outcome.to_sources 的输出同构，避免把向量、
    内部打分等字段写进库。
    """
    with session_scope() as s:
        judge = judge_result or {}
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            outcome=outcome,
            judge_layer=str(judge.get("layer", "") or "")[:32],
            judge_scores=judge.get("scores") or None,
            latency_ms=latency_ms,
            first_token_ms=first_token_ms,
            stream_mode=stream_mode,
            retry_count=retry_count,
            classification=classification,
        )
        s.add(msg)
        s.flush()
        message_id = int(msg.id)

        for rank, src in enumerate(sources or []):
            s.add(ChatSource(
                message_id=message_id,
                rank=rank,
                title=str(src.get("title", ""))[:512],
                source=str(src.get("source", ""))[:32],
                score=float(src.get("score", 0) or 0),
                publish_time=str(src.get("publish_time", ""))[:32],
                department=str(src.get("department", ""))[:64],
                authority_tier=str(src.get("authority_tier", ""))[:16],
                authority_label=str(src.get("authority_label", ""))[:64],
                doc_id=str(src.get("doc_id", ""))[:128],
                chunk_id=str(src.get("chunk_id", ""))[:32],
                snippet=str(src.get("snippet", "")),
            ))

        row = s.get(ChatSession, session_id)
        if row is None:
            row = ChatSession(session_id=session_id)
            s.add(row)
        row.last_active_at = datetime.now()
        if role == "assistant":
            row.turn_count = int(row.turn_count or 0) + 1
            # 会话列表要显示"最近一次出口"，在写消息的同一事务里更新，
            # 避免列表页再回查最后一条消息
            if outcome:
                row.last_outcome = outcome
        if role == "user" and not row.title:
            row.title = content.strip().replace("\n", " ")[:120]

        return message_id


def get_messages(
    session_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """按时间正序取消息（积木式：limit 取最近 N 条后仍按正序返回）。"""
    with session_scope() as s:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
        )
        rows = list(s.scalars(stmt).all())
        if limit is not None and len(rows) > limit:
            rows = rows[-limit:]
        return [
            {
                "id": int(r.id),
                "role": r.role,
                "content": r.content,
                "outcome": r.outcome,
                "judge_layer": r.judge_layer,
                "judge_scores": r.judge_scores,
                "latency_ms": r.latency_ms,
                "first_token_ms": r.first_token_ms,
                "stream_mode": r.stream_mode,
                "retry_count": r.retry_count,
                "classification": r.classification,
                "created_at": _iso(r.created_at),
            }
            for r in rows
        ]


def get_message(message_id: int) -> Optional[Dict[str, Any]]:
    """按 id 取单条消息（反馈提交时校验归属）。"""
    with session_scope() as s:
        row = s.get(ChatMessage, message_id)
        if row is None:
            return None
        return {
            "id": int(row.id),
            "session_id": row.session_id,
            "role": row.role,
            "content": row.content,
            "outcome": row.outcome,
            "created_at": _iso(row.created_at),
        }


def get_sources(message_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    """批量取来源，按 message_id 分组（避免 N+1 查询）。"""
    if not message_ids:
        return {}
    with session_scope() as s:
        stmt = (
            select(ChatSource)
            .where(ChatSource.message_id.in_(message_ids))
            .order_by(ChatSource.message_id.asc(), ChatSource.rank.asc())
        )
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for r in s.scalars(stmt).all():
            grouped.setdefault(int(r.message_id), []).append({
                "rank": r.rank,
                "title": r.title,
                "source": r.source,
                "score": r.score,
                "publish_time": r.publish_time,
                "department": r.department,
                "authority_tier": r.authority_tier,
                "authority_label": r.authority_label,
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "snippet": r.snippet,
            })
        return grouped


def list_sessions(limit: int = 20, offset: int = 0) -> Tuple[List[Dict], int]:
    with session_scope() as s:
        total = int(s.scalar(select(func.count()).select_from(ChatSession)) or 0)
        stmt = (
            select(ChatSession)
            .order_by(ChatSession.last_active_at.desc())
            .offset(max(0, offset))
            .limit(max(1, min(limit, 100)))
        )
        items = [{
            "session_id": r.session_id,
            "title": r.title,
            "turn_count": r.turn_count,
            "last_outcome": r.last_outcome,
            "created_at": _iso(r.created_at),
            "last_active_at": _iso(r.last_active_at),
        } for r in s.scalars(stmt).all()]
        return items, total


def delete_session(session_id: str) -> None:
    """硬删会话及其消息、来源、反馈（复盘数据不留残余）。"""
    with session_scope() as s:
        msg_ids = list(s.scalars(
            select(ChatMessage.id).where(ChatMessage.session_id == session_id)
        ).all())
        if msg_ids:
            s.execute(delete(ChatSource).where(ChatSource.message_id.in_(msg_ids)))
            s.execute(delete(ChatFeedback).where(ChatFeedback.message_id.in_(msg_ids)))
        s.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
        s.execute(delete(ChatRequest).where(ChatRequest.session_id == session_id))
        s.execute(delete(ChatSession).where(ChatSession.session_id == session_id))


def session_detail(session_id: str) -> Optional[Dict[str, Any]]:
    """会话回放：消息 + 来源 + 反馈，一次给全。"""
    base = get_session(session_id)
    if base is None:
        return None
    messages = get_messages(session_id)
    sources = get_sources([m["id"] for m in messages])
    feedback = get_feedback_map([m["id"] for m in messages])
    for m in messages:
        m["sources"] = sources.get(m["id"], [])
        m["feedback"] = feedback.get(m["id"])
    base["messages"] = messages
    return base


# ---------------- 观测 ----------------


def record_request(metrics) -> None:
    """每回合一行指标。失败出口也记，成功率才不会虚高。"""
    total_ms = (time.time() - metrics.start_time) * 1000
    with session_scope() as s:
        s.add(ChatRequest(
            session_id=metrics.session_id or "anonymous",
            query=(metrics.query or "")[:2000],
            outcome=metrics.outcome,
            success=1 if metrics.success else 0,
            classification=metrics.classification_label,
            latency_ms=round(total_ms, 2),
            first_token_ms=round(metrics.first_token_ms, 2),
            stream_mode=metrics.stream_mode,
            risk_level=metrics.risk_level,
            retrieval_latency_ms=round(metrics.retrieval_latency_ms, 2),
            retrieval_count=metrics.retrieval_count,
            retry_count=metrics.retry_count,
            judge_passed=1 if metrics.judge_passed else 0,
            error=(metrics.error_message or "")[:2000],
        ))


# ---------------- 反馈 ----------------


def _feedback_row(row: ChatFeedback) -> Dict[str, Any]:
    return {
        "id": int(row.id),
        "message_id": int(row.message_id),
        "session_id": row.session_id,
        "rating": int(row.rating),
        "reason_tags": row.reason_tags or [],
        "comment": row.comment or "",
        "created_at": _iso(row.created_at),
    }


def save_feedback(
    message_id: int,
    session_id: str,
    rating: int,
    reason_tags: Optional[List[str]] = None,
    comment: str = "",
) -> Dict[str, Any]:
    with session_scope() as s:
        row = s.scalar(
            select(ChatFeedback).where(ChatFeedback.message_id == message_id)
        )
        if row is None:
            row = ChatFeedback(
                message_id=message_id,
                session_id=session_id,
                rating=rating,
                reason_tags=reason_tags or [],
                comment=comment or "",
            )
            s.add(row)
            try:
                s.flush()
            except IntegrityError:
                # 并发重复提交：唯一键挡住第二次插入，回读已有行按更新处理
                s.rollback()
                row = s.scalar(
                    select(ChatFeedback).where(
                        ChatFeedback.message_id == message_id
                    )
                )
                if row is None:
                    raise
                row.rating = rating
                row.reason_tags = reason_tags or []
                row.comment = comment or ""
                row.created_at = datetime.now()
        else:
            # 允许改主意：同一条回答的反馈覆盖写，统计不重复计数
            row.rating = rating
            row.reason_tags = reason_tags or []
            row.comment = comment or ""
            row.created_at = datetime.now()
        s.flush()
        return _feedback_row(row)


def get_feedback_map(message_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not message_ids:
        return {}
    with session_scope() as s:
        stmt = select(ChatFeedback).where(ChatFeedback.message_id.in_(message_ids))
        return {int(r.message_id): _feedback_row(r) for r in s.scalars(stmt).all()}


def feedback_stats(days: int = 7) -> Dict[str, Any]:
    """反馈看板：总量、好评率、负反馈原因分布、按天趋势。"""
    since = datetime.now() - timedelta(days=max(1, min(days, 90)))
    with session_scope() as s:
        rows = list(s.scalars(
            select(ChatFeedback).where(ChatFeedback.created_at >= since)
        ).all())
        day_rows = s.execute(
            select(
                func.date(ChatFeedback.created_at).label("day"),
                ChatFeedback.rating,
                func.count().label("n"),
            )
            .where(ChatFeedback.created_at >= since)
            .group_by(func.date(ChatFeedback.created_at), ChatFeedback.rating)
            .order_by(func.date(ChatFeedback.created_at))
        ).all()

    total = len(rows)
    positive = sum(1 for r in rows if r.rating > 0)
    negative = total - positive
    reasons: Dict[str, int] = {}
    for r in rows:
        if r.rating > 0:
            continue
        for tag in (r.reason_tags or []):
            reasons[tag] = reasons.get(tag, 0) + 1

    by_day: Dict[str, Dict[str, int]] = {}
    for day, rating, n in day_rows:
        key = str(day)
        slot = by_day.setdefault(key, {"positive": 0, "negative": 0})
        slot["positive" if rating > 0 else "negative"] = int(n)

    return {
        "days": days,
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": round(positive / total, 4) if total else 0.0,
        "top_reasons": [
            {"tag": k, "count": v}
            for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:10]
        ],
        "by_day": [
            {"date": d, **v} for d, v in sorted(by_day.items())
        ],
    }


# ---------------- 知识库 ----------------


def upsert_document(
    collection: str,
    doc_id: str,
    department: str,
    path: str,
    size_bytes: int,
    chunk_count: int,
) -> None:
    with session_scope() as s:
        row = s.scalar(
            select(KbDocument).where(
                KbDocument.collection == collection,
                KbDocument.doc_id == doc_id,
                KbDocument.department == department,
            )
        )
        if row is None:
            row = KbDocument(
                collection=collection, doc_id=doc_id, department=department,
            )
            s.add(row)
        row.path = path[:512]
        row.size_bytes = int(size_bytes)
        row.chunk_count = int(chunk_count)
        row.updated_at = datetime.now()


def delete_document_meta(
    collection: str, doc_id: str, department: str
) -> None:
    with session_scope() as s:
        s.execute(
            delete(KbDocument).where(
                KbDocument.collection == collection,
                KbDocument.doc_id == doc_id,
                KbDocument.department == department,
            )
        )


def list_documents(collection: str) -> List[Dict[str, Any]]:
    with session_scope() as s:
        stmt = (
            select(KbDocument)
            .where(KbDocument.collection == collection)
            .order_by(KbDocument.department.asc(), KbDocument.doc_id.asc())
        )
        return [{
            "collection": r.collection,
            "doc_id": r.doc_id,
            "department": r.department,
            "path": r.path,
            "size_bytes": r.size_bytes,
            "chunk_count": r.chunk_count,
            "updated_at": _iso(r.updated_at),
        } for r in s.scalars(stmt).all()]


def create_task(
    task_id: str,
    kind: str,
    collection: str,
    department: str = "",
    doc_id: str = "",
    files: int = 0,
    executor: str = "inline",
) -> Dict[str, Any]:
    with session_scope() as s:
        row = KbTask(
            task_id=task_id,
            kind=kind,
            status="processing",
            collection=collection,
            department=department,
            doc_id=doc_id,
            files=files,
            executor=executor,
            errors=[],
        )
        s.add(row)
        s.flush()
        return _task_row(row)


def update_task(task_id: str, **fields) -> None:
    allowed = {
        "status", "chunks", "failed", "errors", "finished_at",
        "files", "executor", "doc_id", "department",
    }
    with session_scope() as s:
        row = s.get(KbTask, task_id)
        if row is None:
            logger.warning(f"任务不存在，忽略状态回写: {task_id}")
            return
        for key, value in fields.items():
            if key in allowed:
                setattr(row, key, value)


def _task_row(row: KbTask) -> Dict[str, Any]:
    return {
        "task_id": row.task_id,
        "kind": row.kind,
        "status": row.status,
        "collection": row.collection,
        "department": row.department or "",
        "doc_id": row.doc_id or "",
        "files": row.files or 0,
        "chunks": row.chunks or 0,
        "failed": row.failed or 0,
        "errors": row.errors or [],
        "executor": row.executor or "inline",
        "created_at": _iso(row.created_at),
        "finished_at": _iso(row.finished_at),
    }


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with session_scope() as s:
        row = s.get(KbTask, task_id)
        return _task_row(row) if row else None


def list_tasks(limit: int = 20, collection: str = "") -> List[Dict[str, Any]]:
    with session_scope() as s:
        stmt = select(KbTask).order_by(KbTask.created_at.desc())
        if collection:
            stmt = stmt.where(KbTask.collection == collection)
        stmt = stmt.limit(max(1, min(limit, 100)))
        return [_task_row(r) for r in s.scalars(stmt).all()]
