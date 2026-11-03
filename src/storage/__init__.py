"""存储子包：数据库、缓存、仓储。

对外只暴露三个入口（其余模块请勿直接 import models）：
  - src.storage.db.init_schema / reset_engine / session_scope
  - src.storage.repository（所有 SQL）
  - src.storage.cache.get_cache（会话热缓存）
"""
