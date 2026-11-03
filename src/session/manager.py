# ============================================================
# 会话管理 + 动态上下文压缩
#
# 存储分层（ADR-0008）：
#   MySQL  chat_session / chat_message —— 真源，永久留存供复盘调优
#   Redis  sess:{id}:msgs —— 热副本，24h TTL，读穿透；不可用时直读 MySQL
#
# 本模块只做三件事：会话生命周期、历史读写（经缓存）、上下文压缩；
# 具体 SQL 全部在 src/storage/repository.py，缓存细节在 src/storage/cache.py。
#
# 压缩策略：token > 2000 或 > 4 轮 → 早期对话生成摘要，保留近 4 轮；
# 摘要与其覆盖到的消息条数一起落库（summary_upto），避免摘要与历史错位。
# ============================================================

import threading
import uuid
from typing import Dict, List, Optional, Tuple

from loguru import logger

from config.settings import settings
from src.core.llm import get_llm
from src.storage import repository
from src.storage.cache import get_cache
from src.utils.helpers import count_tokens, load_prompt_template

_SESSION_ID_MAX = 64


class SessionManager:
    """会话管理器（MySQL 真源 + Redis 读穿透）。"""

    def __init__(self):
        self._cache = get_cache()
        self._lock = threading.Lock()

    # ---- 会话生命周期 ----

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """取会话；不存在则创建。

        采纳调用方传来的 session_id（重启/换实例后仍能续聊）；超长或空值
        一律换发新 uuid，避免把非法值写进主键列。
        """
        sid = (session_id or "").strip()
        if not sid or len(sid) > _SESSION_ID_MAX:
            sid = uuid.uuid4().hex
        repository.ensure_session(sid)
        return sid

    def delete_session(self, session_id: str) -> None:
        """删除会话（消息、来源、反馈、指标一起清）。"""
        repository.delete_session(session_id)
        self._cache.delete(session_id)

    # ---- 消息读写 ----

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        **meta,
    ) -> Optional[int]:
        """写一条消息：先落 MySQL（真源），再追加 Redis（若该会话已在缓存中）。

        缓存未命中时故意不写：否则会造出"只有新消息"的半截历史，
        下次读缓存就再也看不到早期对话。等下次 get_history 回源整段回填。
        """
        message_id = repository.append_message(
            session_id, role, content, **meta
        )
        if self._cache.exists(session_id):
            self._cache.append_message(session_id, role, content)
        return message_id

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """读历史：Redis 命中即返；未命中回源 MySQL 并回填。"""
        cached = self._cache.get_history(session_id)
        if cached is not None:
            return cached

        rows = repository.get_messages(session_id)
        history = [{"role": r["role"], "content": r["content"]} for r in rows]
        self._cache.set_history(session_id, history)
        return history

    # ---- 动态上下文压缩 ----

    def build_context(
        self, session_id: str, current_query: str
    ) -> Tuple[str, int]:
        """
        构建发送给大模型的完整上下文

        策略:
        1. 计算总 token 数
        2. 如果 >2000 或 >4轮 → 触发压缩
        3. 格式: [历史摘要] + [近4轮对话] + [当前query]
        返回: (context_text, total_tokens)
        """
        history = self.get_history(session_id)
        if not history:
            return current_query, count_tokens(current_query)

        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )
        total_text = history_text + f"\nuser: {current_query}"
        total_tokens = count_tokens(total_text)
        rounds = len(history) // 2  # 对话轮次（每轮=user+assistant）

        need_compress = (
            total_tokens > settings.session_max_tokens
            or rounds >= settings.session_compress_rounds
        )

        if need_compress:
            context = self._compress_and_build(session_id, history, current_query)
            return context, count_tokens(context)

        parts = [f"{m['role']}: {m['content']}" for m in history]
        parts.append(f"user: {current_query}")
        return "\n".join(parts), total_tokens

    def _compress_and_build(
        self, session_id: str, history: List[Dict], current_query: str
    ) -> str:
        """
        执行上下文压缩并构建输出
        - 早期对话(>4轮) → 大模型提取摘要（落库，后续复用）
        - 近4轮 → 完整保留
        - 拼接: [摘要] + [近4轮] + [当前query]
        """
        rounds = len(history) // 2
        compress_threshold = settings.session_compress_rounds  # 4

        if rounds <= compress_threshold:
            parts = [f"{m['role']}: {m['content']}" for m in history]
            parts.append(f"user: {current_query}")
            return "\n".join(parts)

        # 分割：早期对话 vs 近4轮
        split_idx = (rounds - compress_threshold) * 2
        early_history = history[:split_idx]
        recent_history = history[split_idx:]

        session = repository.get_session(session_id) or {}
        summary = session.get("summary") or ""
        summary_upto = int(session.get("summary_upto") or 0)

        # 摘要覆盖范围与当前切分点不一致（新增长轮次）→ 重算并落库
        if not summary or summary_upto != split_idx:
            summary = self._generate_summary(early_history)
            if summary:
                repository.update_summary(session_id, summary, split_idx)

        parts = []
        if summary:
            parts.append(f"[历史对话摘要：{summary}]")

        recent_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in recent_history
        )
        parts.append(f"[最近{compress_threshold}轮的完整对话：\n{recent_text}]")
        parts.append(f"[当前用户的新提问：{current_query}]")

        result = "\n\n".join(parts)
        logger.info(
            f"上下文压缩完成: 原始轮次={rounds}, "
            f"压缩后token≈{count_tokens(result)}, "
            f"摘要长度={len(summary)}字"
        )
        return result

    def _generate_summary(self, early_history: List[Dict]) -> str:
        """
        大模型生成历史对话摘要
        Prompt: 只保留症状/病史/诊断结果/用药情况
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

        # 调用大模型生成摘要（走 LLM 总机，任务 summary）
        try:
            summary = get_llm().complete("summary", prompt)
            return summary[:200]  # 限200字
        except Exception as e:
            logger.warning(f"摘要生成失败，使用简单截断: {e}")
            return history_text[:200]


# 全局单例
session_manager = SessionManager()
