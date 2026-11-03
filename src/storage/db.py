# ============================================================
# 数据库连接（引擎 / 会话工厂 / 建表）
#
# 单一收口点：业务代码不再自己 create_engine，也不 import pymysql。
#   - 生产：DATABASE_URL=mysql+pymysql://user:pwd@mysql:3306/medical_rag
#   - 本地/CI：默认 sqlite:///volumes/app.db（零依赖，测试可直接跑）
# 两套后端共用同一套 SQLAlchemy 模型，SQL 方言差异由 ORM 吸收。
#
# 引擎按 settings.database_url 懒加载并按 URL 缓存：测试把 URL 指到临时
# 文件后调 reset_engine() 即可彻底隔离，不需要 reload 模块。
#
# 决策见 docs/adr/0008-storage-layering.md。
# ============================================================

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.storage.models import Base

_engine: Optional[Engine] = None
_engine_url: Optional[str] = None
_session_factory: Optional[sessionmaker] = None
_schema_ready = False
# RLock：init_schema 持锁期间会调 get_engine，后者也要拿同一把锁
_lock = threading.RLock()


def _ensure_sqlite_dir(url: str) -> None:
    """SQLite 文件库：建库前先确保父目录存在（volumes/ 可能被清空）。"""
    if not url.startswith("sqlite:///"):
        return
    raw = url[len("sqlite:///"):]
    if not raw or raw == ":memory:":
        return
    parent = Path(raw).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> Engine:
    """返回当前配置对应的引擎（URL 变化时重建，供测试隔离用）。"""
    global _engine, _engine_url, _session_factory

    url = settings.database_url
    if _engine is not None and _engine_url == url:
        return _engine

    with _lock:
        if _engine is not None and _engine_url == url:
            return _engine

        if _engine is not None:
            _engine.dispose()

        _ensure_sqlite_dir(url)
        kwargs = {"echo": settings.db_echo, "future": True}
        if url.startswith("sqlite"):
            # 后台线程（Celery/BackgroundTasks）与请求线程会共用引擎
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs.update(
                pool_pre_ping=True,
                pool_recycle=settings.db_pool_recycle,
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_pool_size * 2,
            )

        _engine = create_engine(url, **kwargs)
        _session_factory = sessionmaker(
            bind=_engine, expire_on_commit=False, future=True
        )
        _engine_url = url
        logger.info(f"数据库引擎已创建: {url.split('@')[-1]}")
        return _engine


def get_session_factory() -> sessionmaker:
    get_engine()
    assert _session_factory is not None
    return _session_factory


def init_schema() -> None:
    """建表（幂等）。首次访问数据层时自动调用，也可由 scripts/init_db.py 显式触发。"""
    global _schema_ready
    if _schema_ready:
        return
    with _lock:
        if _schema_ready:
            return
        try:
            Base.metadata.create_all(get_engine())
        except (OperationalError, ProgrammingError) as e:
            # 并发首启（API 与 Celery worker 同时建表）时对方可能刚好建完。
            # 表齐了就继续，别让一场竞态把服务卡死在启动阶段。
            logger.warning(f"建表遇到并发/权限问题，改为校验现有表: {e}")
            existing = set(inspect(get_engine()).get_table_names())
            missing = [t for t in Base.metadata.tables if t not in existing]
            if missing:
                raise
        _schema_ready = True
        logger.info("数据库表结构已就绪")


def reset_engine() -> None:
    """丢弃当前引擎（测试切库 / 配置热变更后调用）。"""
    global _engine, _engine_url, _session_factory, _schema_ready
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _engine_url = None
        _session_factory = None
        _schema_ready = False


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务边界：正常提交，异常回滚，最后关闭。

    仓储层之外的调用方一律走 with session_scope()，不要手工 commit。
    """
    init_schema()
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
