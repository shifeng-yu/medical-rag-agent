# ============================================================
# 全局测试夹具
#
# 存储隔离：所有测试统一落到会话级临时 SQLite 文件，绝不碰 volumes/app.db，
# 也不依赖 Redis（cache_enabled=False 时缓存层全部 no-op，读路径回源 MySQL/SQLite）。
#
# 之所以放在 session 级：引擎是进程级单例，按测试函数重建会反复建表；
# 单文件库 + 每个测试自造 session_id 已足够隔离。
# ============================================================

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_storage(tmp_path_factory):
    from config.settings import settings

    db_path = tmp_path_factory.mktemp("appdb") / "test.db"
    settings.database_url = f"sqlite:///{db_path.as_posix()}"
    settings.cache_enabled = False
    settings.task_mode = "inline"

    from src.storage import db as storage_db

    storage_db.reset_engine()
    storage_db.init_schema()

    yield

    storage_db.reset_engine()
