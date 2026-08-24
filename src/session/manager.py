# ============================================================
# 会话缓存管理 + 动态上下文压缩模块
# 来源：独立会话缓存「动态上下文压缩」
# - Python 内存字典，key=session_id，24h TTL
# - Token 计数触发压缩（>2000 或 >4轮）
# - 早期对话摘要 + 近4轮保留
# ============================================================

import uuid
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from config.settings import settings
from src.utils.helpers import count_tokens, load_prompt_template


@dataclass
class SessionData:
    """单个会话的数据结构 (项目需求)"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    # 完整对话历史: [{"role": "user"/"assistant", "content": "..."}]
    history: List[Dict[str, str]] = field(default_factory=list)
    # 历史对话压缩摘要 ( 早期对话提取的核心信息)
    summary: str = ""
    # 是否已初始化（用于首次访问判断）


class SessionManager:
    """
    会话管理器 (项目需求)
    - 每个用户独立 session_id，内存字典存储
    - 24h 自动过期清理
    - 动态上下文压缩：token > 2000 或轮次 > 4 → 触发
    """

    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        # 启动后台过期清理
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("会话管理器已初始化")

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        获取或创建会话 ( 首次访问生成唯一 session_id)
        返回: session_id
        """
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                # 检查是否过期
                if time.time() - session.last_access > settings.session_ttl_seconds:
                    del self._sessions[session_id]
                else:
                    session.last_access = time.time()
                    return session_id

            # 创建新会话
            new_id = str(uuid.uuid4())
            self._sessions[new_id] = SessionData(session_id=new_id)
            logger.info(f"新会话创建: {new_id}")
            return new_id

    def add_message(self, session_id: str, role: str, content: str):
        """添加一条对话记录"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.history.append({"role": role, "content": content})
                session.last_access = time.time()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取完整对话历史"""
        with self._lock:
            session = self._sessions.get(session_id)
            return session.history.copy() if session else []

    # ---- 动态上下文压缩 (项目需求) ----

    def build_context(
        self, session_id: str, current_query: str
    ) -> Tuple[str, int]:
        """
        构建发送给大模型的完整上下文 (项目需求)
        策略:
        1. 计算总 token 数
        2. 如果 >2000 或 >4轮 → 触发压缩
        3. 格式: [历史摘要] + [近4轮对话] + [当前query]
        返回: (context_text, total_tokens)
        """
        session = self._sessions.get(session_id)
        if not session or not session.history:
            return current_query, count_tokens(current_query)

        history = session.history

        # 计算当前 token 总量 ( 历史 + 新 query)
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )
        total_text = history_text + f"\nuser: {current_query}"
        total_tokens = count_tokens(total_text)
        rounds = len(history) // 2  # 对话轮次（每轮=user+assistant）

        # 判断是否需要压缩 ( >2000 token 或 >4轮)
        need_compress = (
            total_tokens > settings.session_max_tokens
            or rounds >= settings.session_compress_rounds
        )

        if need_compress:
            return self._compress_and_build(session, history, current_query), count_tokens(
                self._compress_and_build(session, history, current_query)
            )
        else:
            parts = [f"{m['role']}: {m['content']}" for m in history]
            parts.append(f"user: {current_query}")
            return "\n".join(parts), total_tokens

    def _compress_and_build(
        self, session: SessionData, history: List[Dict], current_query: str
    ) -> str:
        """
        执行上下文压缩并构建输出 (项目需求)
        - 早期对话(>4轮) → 大模型提取摘要
        - 近4轮 → 完整保留
        - 拼接: [摘要] + [近4轮] + [当前query]
        """
        rounds = len(history) // 2
        compress_threshold = settings.session_compress_rounds  # 4

        if rounds > compress_threshold:
            # 分割：早期对话 vs 近4轮
            split_idx = (rounds - compress_threshold) * 2
            early_history = history[:split_idx]
            recent_history = history[split_idx:]

            # 尝试用已有摘要，或生成新摘要 ( 大模型提取核心信息)
            if not session.summary:
                session.summary = self._generate_summary(early_history)

            # 拼接 ( 标准结构)
            parts = []
            if session.summary:
                parts.append(f"[历史对话摘要：{session.summary}]")

            recent_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in recent_history
            )
            parts.append(f"[最近{compress_threshold}轮的完整对话：\n{recent_text}]")
            parts.append(f"[当前用户的新提问：{current_query}]")

            result = "\n\n".join(parts)
            logger.info(
                f"上下文压缩完成: 原始轮次={rounds}, "
                f"压缩后token≈{count_tokens(result)}, "
                f"摘要长度={len(session.summary)}字"
            )
            return result
        else:
            # 不需要压缩
            parts = [f"{m['role']}: {m['content']}" for m in history]
            parts.append(f"user: {current_query}")
            return "\n".join(parts)

    def _generate_summary(self, early_history: List[Dict]) -> str:
        """
        大模型生成历史对话摘要 (项目需求)
        Prompt: 只保留症状/病史/诊断结果/用药情况
        来源: 「把核心问诊信息提取出来，做成简洁摘要，不超过200字」
        """
        prompt_template = load_prompt_template("summary")
        if not prompt_template:
            prompt_template = (
                "下面是用户和问诊助手的历史对话，请你把这些对话里的核心问诊信息提取出来，"
                "做成一个简洁的摘要：\n"
                "1.只保留用户的症状、病史、历史的诊断结果、用药情况这些和问诊相关的核心信息\n"
                "2.删掉无关的冗余内容、客套话\n"
                "3.摘要要尽量简短，总长度不要超过200字\n\n"
                "历史对话：\n{history}"
            )

        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in early_history
        )
        prompt = prompt_template.replace("{history}", history_text)

        # 调用大模型生成摘要(同步调用封装)
        try:
            from src.core.generator import generate_text

            summary = generate_text(prompt, max_tokens=256)
            return summary[:200]  # 限200字 (项目需求)
        except Exception as e:
            logger.warning(f"摘要生成失败，使用简单截断: {e}")
            # 降级：取早期对话的文本截断
            return history_text[:200]

    # ---- 过期清理 ( 24h自动清理) ----

    def _cleanup_loop(self):
        """后台线程定期清理过期会话 ( 24h TTL)"""
        while True:
            time.sleep(3600)  # 每小时检查一次
            self._cleanup_expired()

    def _cleanup_expired(self):
        """清理过期会话"""
        now = time.time()
        with self._lock:
            expired_ids = [
                sid
                for sid, s in self._sessions.items()
                if now - s.last_access > settings.session_ttl_seconds
            ]
            for sid in expired_ids:
                del self._sessions[sid]
            if expired_ids:
                logger.info(f"清理过期会话: {len(expired_ids)} 个")

    def delete_session(self, session_id: str):
        """手动删除会话"""
        with self._lock:
            self._sessions.pop(session_id, None)


# 全局单例
session_manager = SessionManager()
