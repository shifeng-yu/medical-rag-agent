# ============================================================
# Redis 会话热缓存（Cache-Aside）
#
# 定位：MySQL 是真源，Redis 只保存"最近 N 条消息"的热副本与 24h TTL。
#   - 读：先读 Redis，未命中回源 MySQL 并回填（get_history 返回值区分
#         "命中空列表"与"未命中"，用 None 表示未命中）
#   - 写：写 MySQL 成功后追加 Redis（append），DB 永远是权威
#   - 故障：Redis 连不上时全链路降级为直接读 MySQL，只告警一次，
#          不阻断问诊
#
# 为什么用 List 而不是 JSON blob：每条消息一次 RPUSH + LTRIM 即可，
# 不需要"读-改-写"，多进程写同一条会话时不会互相覆盖。
#
# 决策见 docs/adr/0008-storage-layering.md。
# ============================================================

import json
from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings

# 单会话缓存的消息上限：超过后从头部丢弃（更早的内容由 MySQL 兜底）
_MAX_CACHED_MESSAGES = 100


class SessionCache:
    """会话热缓存。不可用时所有方法都是安全的 no-op。"""

    def __init__(self) -> None:
        self._client = None
        # 运行期开关（不是构造期快照）：测试/运维把 cache_enabled 关掉后立即生效
        self._disabled = False
        self._warned = False

    # ---- 连接管理 ----

    def _get_client(self):
        if self._disabled or not settings.cache_enabled:
            return None
        if self._client is not None:
            return self._client
        try:
            import redis

            client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.3,
                socket_timeout=0.5,
            )
            client.ping()
            self._client = client
            logger.info(f"Redis 会话缓存已连接: {settings.redis_url}")
            return client
        except Exception as e:  # noqa: BLE001 - 缓存不可用不应阻断业务
            self._degrade(e)
            return None

    def _degrade(self, exc: Exception) -> None:
        """降级为纯 MySQL：只告警一次，避免刷屏。"""
        self._disabled = True
        self._client = None
        if not self._warned:
            self._warned = True
            logger.warning(f"Redis 不可用，会话缓存降级为直读 MySQL: {exc}")

    @staticmethod
    def _key(session_id: str) -> str:
        return f"sess:{session_id}:msgs"

    # ---- 读 ----

    def get_history(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """命中返回消息列表（可能为空列表），未命中/不可用返回 None。"""
        client = self._get_client()
        if client is None:
            return None
        try:
            raw = client.lrange(self._key(session_id), 0, -1)
            if not raw:
                return None
            return [json.loads(item) for item in raw]
        except Exception as e:  # noqa: BLE001
            self._degrade(e)
            return None

    # ---- 写 ----

    def exists(self, session_id: str) -> bool:
        """该会话是否已有热副本。

        add_message 依赖它判断"能否安全追加"：缓存未命中时追加会造出
        半截历史（只有最新几条），下次读缓存就丢失早期对话。
        """
        client = self._get_client()
        if client is None:
            return False
        try:
            return bool(client.exists(self._key(session_id)))
        except Exception as e:  # noqa: BLE001
            self._degrade(e)
            return False

    def set_history(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        """回填缓存（来自 MySQL）。整段替换，保证与库一致。"""
        client = self._get_client()
        if client is None:
            return
        key = self._key(session_id)
        try:
            pipe = client.pipeline()
            pipe.delete(key)
            tail = messages[-_MAX_CACHED_MESSAGES:]
            if tail:
                pipe.rpush(key, *[json.dumps(m, ensure_ascii=False) for m in tail])
            pipe.expire(key, settings.cache_ttl_seconds)
            pipe.execute()
        except Exception as e:  # noqa: BLE001
            self._degrade(e)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息。缓存未命中时不写入（等下次回填整段，避免半截历史）。"""
        client = self._get_client()
        if client is None:
            return
        key = self._key(session_id)
        try:
            pipe = client.pipeline()
            pipe.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
            pipe.ltrim(key, -_MAX_CACHED_MESSAGES, -1)
            pipe.expire(key, settings.cache_ttl_seconds)
            pipe.execute()
        except Exception as e:  # noqa: BLE001
            self._degrade(e)

    def delete(self, session_id: str) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            client.delete(self._key(session_id))
        except Exception as e:  # noqa: BLE001
            self._degrade(e)


_cache: Optional[SessionCache] = None


def get_cache() -> SessionCache:
    global _cache
    if _cache is None:
        _cache = SessionCache()
    return _cache


def reset_cache() -> None:
    """测试用：丢弃连接与降级标记，重新按当前配置建连。"""
    global _cache
    _cache = None
