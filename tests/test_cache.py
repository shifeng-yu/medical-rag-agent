# ============================================================
# Redis 读穿透（Cache-Aside）行为测试
#
# 用内存假客户端替代真实 Redis，验证四条关键语义：
#   1. 未命中 → 回源 MySQL 并回填
#   2. 命中 → 不再查库（把仓储打成异常也能返回）
#   3. 冷缓存下写消息不追加（避免造出"只有新消息"的半截历史）
#   4. 暖缓存下写消息才追加
# 默认测试环境关缓存（CACHE_ENABLED=0），这里单独打开。
# ============================================================

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from config.settings import settings
from src.session.manager import session_manager
from src.storage.cache import SessionCache


class _Pipe:
    def __init__(self, fake):
        self.fake = fake
        self.ops = []

    def delete(self, key):
        self.ops.append(("delete", key))
        return self

    def rpush(self, key, *values):
        self.ops.append(("rpush", key, values))
        return self

    def ltrim(self, key, start, end):
        self.ops.append(("ltrim", key, start, end))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))
        return self

    def execute(self):
        for op in self.ops:
            name = op[0]
            if name == "delete":
                self.fake.lists.pop(op[1], None)
            elif name == "rpush":
                self.fake.lists.setdefault(op[1], []).extend(op[2])
            elif name == "ltrim":
                key, start, end = op[1], op[2], op[3]
                items = self.fake.lists.get(key, [])
                self.fake.lists[key] = items[start:end] if end != -1 else items[start:]
            elif name == "expire":
                self.fake.ttl[op[1]] = op[2]
        self.ops.clear()
        return [True] * 4


class FakeRedis:
    """只实现会话缓存用到的 6 个命令。"""

    def __init__(self):
        self.lists = {}
        self.ttl = {}

    @staticmethod
    def _key(session_id):
        return f"sess:{session_id}:msgs"

    def lrange(self, key, start, end):
        return list(self.lists.get(key, []))

    def exists(self, key):
        return 1 if self.lists.get(key) else 0

    def delete(self, key):
        self.lists.pop(key, None)
        return 1

    def pipeline(self):
        return _Pipe(self)

    def ping(self):
        return True


@pytest.fixture()
def fake_cache(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", True)
    cache = SessionCache()
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_get_client", lambda: fake)
    monkeypatch.setattr(session_manager, "_cache", cache)
    return fake, cache


def _seed(sid: str) -> None:
    session_manager.get_or_create_session(sid)
    session_manager.add_message(sid, "user", "问题")
    session_manager.add_message(sid, "assistant", "回答", outcome="ok")


class TestReadThrough:
    def test_miss_reads_db_and_backfills(self, fake_cache):
        fake, _ = fake_cache
        sid = f"c-{uuid.uuid4().hex[:8]}"
        _seed(sid)
        key = FakeRedis._key(sid)
        assert key not in fake.lists

        history = session_manager.get_history(sid)
        assert [h["role"] for h in history] == ["user", "assistant"]
        # 回填：缓存里已有同样两条
        assert [json.loads(x)["content"] for x in fake.lists[key]] == ["问题", "回答"]
        assert fake.ttl[key] == settings.cache_ttl_seconds

    def test_hit_bypasses_repository(self, fake_cache, monkeypatch):
        fake, _ = fake_cache
        sid = f"c-{uuid.uuid4().hex[:8]}"
        _seed(sid)
        session_manager.get_history(sid)  # 预热

        def _boom(*args, **kwargs):
            raise AssertionError("命中缓存时不应回源数据库")

        import src.storage.repository as repo

        monkeypatch.setattr(repo, "get_messages", _boom)

        history = session_manager.get_history(sid)
        assert len(history) == 2

    def test_cold_cache_does_not_append(self, fake_cache):
        """冷缓存写消息不追加：否则会留下半截历史。"""
        fake, _ = fake_cache
        sid = f"c-{uuid.uuid4().hex[:8]}"
        session_manager.get_or_create_session(sid)
        session_manager.add_message(sid, "user", "新问题")

        assert FakeRedis._key(sid) not in fake.lists
        # 首次读取回源，拿到完整历史
        assert len(session_manager.get_history(sid)) == 1

    def test_warm_cache_appends(self, fake_cache):
        fake, _ = fake_cache
        sid = f"c-{uuid.uuid4().hex[:8]}"
        _seed(sid)
        session_manager.get_history(sid)  # 预热
        session_manager.add_message(sid, "user", "第三句")

        cached = fake.lists[FakeRedis._key(sid)]
        assert len(cached) == 3
        assert json.loads(cached[-1])["content"] == "第三句"


class TestDegrade:
    def test_redis_unavailable_falls_back_to_db(self, monkeypatch):
        """缓存不可用：读路径直接回源，不受影响。"""
        monkeypatch.setattr(settings, "cache_enabled", True)
        cache = SessionCache()
        cache._disabled = True  # 模拟连不上
        monkeypatch.setattr(session_manager, "_cache", cache)

        sid = f"c-{uuid.uuid4().hex[:8]}"
        _seed(sid)
        assert len(session_manager.get_history(sid)) == 2
        # 写入不抛异常，也不影响库
        session_manager.add_message(sid, "user", "再加一句")
        assert len(session_manager.get_history(sid)) == 3
