# ============================================================
# 数据模型（7 张表）
#
# 分域：
#   会话域   chat_session / chat_message / chat_source / chat_feedback
#   观测域   chat_request（每回合一行，含拦截/超时等不写会话的出口）
#   知识库域 kb_document / kb_task
#
# 设计取舍：
#   - 消息与来源拆两张表：来源是"一次回答命中多个块"的一对多，
#     合成 JSON 列会让"哪条引用被反馈"无法用 SQL 归因；
#   - chat_request 与 chat_message 不合并：失败出口（error/overload/
#     timeout）按规则不写会话历史，但仍要进统计，否则成功率会失真；
#   - 主键 id 用 BigInteger + SQLite 变体 Integer：SQLite 只有
#     INTEGER PRIMARY KEY 才自增；
#   - 时间统一用本地时间（与日志/监控一致），内网单时区部署。
# ============================================================

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now()


def _pk() -> Mapped[int]:
    return mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )


class Base(DeclarativeBase):
    pass


# ---------------- 会话域 ----------------


class ChatSession(Base):
    __tablename__ = "chat_session"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(128), default="")
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    # 上下文压缩摘要与其覆盖到的消息条数（重算依据，避免摘要与历史错位）
    summary: Mapped[str] = mapped_column(Text, default="")
    summary_upto: Mapped[int] = mapped_column(Integer, default=0)
    last_outcome: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, index=True
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"
    __table_args__ = (
        Index("ix_chat_message_session", "session_id", "id"),
    )

    id: Mapped[int] = _pk()
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    # 以下字段仅 assistant 行有值，用于复盘定位（不写会话的出口不落此表）
    outcome: Mapped[str] = mapped_column(String(32), default="")
    judge_layer: Mapped[str] = mapped_column(String(32), default="")
    judge_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    first_token_ms: Mapped[float] = mapped_column(Float, default=0.0)
    stream_mode: Mapped[str] = mapped_column(String(16), default="")
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    classification: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


class ChatSource(Base):
    __tablename__ = "chat_source"
    __table_args__ = (
        Index("ix_chat_source_message", "message_id", "rank"),
    )

    id: Mapped[int] = _pk()
    message_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), index=True
    )
    rank: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(512), default="")
    source: Mapped[str] = mapped_column(String(32), default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    publish_time: Mapped[str] = mapped_column(String(32), default="")
    department: Mapped[str] = mapped_column(String(64), default="")
    authority_tier: Mapped[str] = mapped_column(String(16), default="")
    authority_label: Mapped[str] = mapped_column(String(64), default="")
    doc_id: Mapped[str] = mapped_column(String(128), default="")
    chunk_id: Mapped[str] = mapped_column(String(32), default="")
    snippet: Mapped[str] = mapped_column(Text, default="")


class ChatFeedback(Base):
    __tablename__ = "chat_feedback"
    __table_args__ = (
        # 一条回答只保留最新一次反馈：前端允许改主意，避免重复统计
        UniqueConstraint("message_id", name="uq_chat_feedback_message"),
        Index("ix_chat_feedback_session", "session_id", "created_at"),
    )

    id: Mapped[int] = _pk()
    message_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), index=True
    )
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    rating: Mapped[int] = mapped_column(SmallInteger)  # 1 有用 / -1 没用
    reason_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


# ---------------- 观测域 ----------------


class ChatRequest(Base):
    """一次问诊回合的指标行（无论出口是什么都记一条）。"""

    __tablename__ = "chat_request"
    __table_args__ = (
        Index("ix_chat_request_created", "created_at"),
        Index("ix_chat_request_outcome", "outcome", "created_at"),
    )

    id: Mapped[int] = _pk()
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    query: Mapped[str] = mapped_column(Text, default="")
    outcome: Mapped[str] = mapped_column(String(32), default="")
    success: Mapped[int] = mapped_column(SmallInteger, default=0)
    classification: Mapped[str] = mapped_column(String(16), default="")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    first_token_ms: Mapped[float] = mapped_column(Float, default=0.0)
    stream_mode: Mapped[str] = mapped_column(String(16), default="")
    risk_level: Mapped[str] = mapped_column(String(16), default="")
    retrieval_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    judge_passed: Mapped[int] = mapped_column(SmallInteger, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


# ---------------- 知识库域 ----------------


class KbDocument(Base):
    """已入库文档的元数据快照（文件系统仍是源真，此表供后台展示与统计）。"""

    __tablename__ = "kb_document"
    __table_args__ = (
        UniqueConstraint(
            "collection", "doc_id", "department", name="uq_kb_document"
        ),
        Index("ix_kb_document_collection", "collection", "department"),
    )

    id: Mapped[int] = _pk()
    collection: Mapped[str] = mapped_column(String(32))
    doc_id: Mapped[str] = mapped_column(String(128))
    department: Mapped[str] = mapped_column(String(64), default="")
    path: Mapped[str] = mapped_column(String(512), default="")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class KbTask(Base):
    """灌库 / 重建任务。落库后跨进程可见（Celery worker 与 API 不共享内存）。"""

    __tablename__ = "kb_task"
    __table_args__ = (
        Index("ix_kb_task_created", "created_at"),
    )

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(24), default="upload")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    collection: Mapped[str] = mapped_column(String(32), default="local_kb")
    department: Mapped[str] = mapped_column(String(64), default="")
    doc_id: Mapped[str] = mapped_column(String(128), default="")
    files: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    executor: Mapped[str] = mapped_column(String(16), default="inline")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
