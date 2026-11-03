# ============================================================
# 会话复盘接口（管理后台用）
#
#   GET    /api/v1/sessions            会话列表（按活跃时间倒序）
#   GET    /api/v1/sessions/{id}       单会话完整回放（消息 + 来源 + 反馈）
#   DELETE /api/v1/sessions/{id}       删除会话（消息/来源/反馈/指标一起清）
#
# 只读为主：复盘是"看"，不提供改写历史的能力——审计数据一旦可改就失去
# 参考价值。删除是唯一例外（隐私/清理需求）。
# ============================================================

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import SessionDetailResponse, SessionListResponse
from src.session.manager import session_manager
from src.storage import repository

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse)
async def list_sessions(limit: int = 20, offset: int = 0):
    """列出会话：标题取首条提问，带轮次与最近一次出口。"""
    items, total = repository.list_sessions(limit=limit, offset=offset)
    return SessionListResponse(total=total, items=items)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(session_id: str):
    """会话回放：消息按时间正序，assistant 行带来源与反馈。"""
    data = repository.session_detail(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    return SessionDetailResponse(**data)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """删除会话及其派生数据（不可恢复）。"""
    session_manager.delete_session(session_id)
    logger.info(f"[API] 会话已删除: {session_id}")
    return {"status": "deleted", "session_id": session_id}
