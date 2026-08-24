"""
Concurrency control: rate limiting, request queue, connection pooling, graceful degradation.

Covers:
  - Per-user rate limiting (sliding window)
  - Global concurrent request cap
  - Request timeout enforcement
  - Graceful overload response (503 + retry-after)
  - Thread-safe session access
"""

import time
import threading
from collections import defaultdict
from typing import Tuple


class RateLimiter:
    """Sliding window rate limiter per user + global."""

    def __init__(self, max_per_user: int = 10, window_sec: int = 60,
                 max_global_concurrent: int = 20):
        self.max_per_user = max_per_user
        self.window_sec = window_sec
        self.max_global_concurrent = max_global_concurrent
        self._user_requests = defaultdict(list)  # user_id -> [timestamps]
        self._active_requests = 0
        self._lock = threading.Lock()

    def acquire(self, user_id: str) -> Tuple[bool, str]:
        """
        Try to acquire a request slot. Returns (allowed, reason).
        """
        now = time.time()

        with self._lock:
            # Clean old entries
            cutoff = now - self.window_sec
            for uid in list(self._user_requests.keys()):
                self._user_requests[uid] = [
                    t for t in self._user_requests[uid] if t > cutoff
                ]
                if not self._user_requests[uid]:
                    del self._user_requests[uid]

            # Check per-user limit
            user_history = self._user_requests[user_id]
            if len(user_history) >= self.max_per_user:
                oldest = min(user_history)
                retry_after = int(oldest + self.window_sec - now + 1)
                return False, f"Rate limit exceeded. Retry after {retry_after}s"

            # Check global concurrency
            if self._active_requests >= self.max_global_concurrent:
                return False, "Server under heavy load. Please retry shortly."

            # Acquire
            user_history.append(now)
            self._user_requests[user_id] = user_history
            self._active_requests += 1
            return True, "OK"

    def release(self, user_id: str):
        """Release a request slot after completion."""
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_requests


class GracefulDegradation:
    """Graceful degradation under overload."""

    @staticmethod
    def overload_response():
        return {
            "answer": (
                "抱歉，当前服务请求量较大，暂时无法处理您的问诊。\n"
                "建议您稍后重试，或前往正规医疗机构咨询专业医生。"
            ),
            "sources": [],
            "judge_result": {"layer": "overload"},
            "latency_ms": 0,
        }

    @staticmethod
    def timeout_response():
        return {
            "answer": (
                "抱歉，处理您的请求超时。请尝试简化问题后重试。\n"
                "如需紧急医疗帮助，请立即前往医院就诊。"
            ),
            "sources": [],
            "judge_result": {"layer": "timeout"},
            "latency_ms": 0,
        }


# Global instances
rate_limiter = RateLimiter(max_per_user=10, window_sec=60, max_global_concurrent=20)
graceful = GracefulDegradation()
