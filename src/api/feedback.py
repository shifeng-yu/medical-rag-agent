# ============================================================
# 用户反馈接口
#
#   POST /api/v1/feedback        对某条回答点赞/点踩（可带原因标签与文本）
#   GET  /api/v1/feedback/stats  反馈看板（好评率、负反馈原因分布、按天趋势）
#
# 设计要点：
#   - 反馈必须绑定到具体 message_id：只有绑定到"哪一次回答、命中了哪些块"，
#     才能回答"是检索错了还是生成错了"这个复盘核心问题；
#   - 同一条回答只保留最新一次反馈（唯一键 + 覆盖写），避免重复计数；
#   - 不做登录：内网单租户，session_id 即归属，message_id 由服务端校验存在性。
# ============================================================

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import FeedbackItem, FeedbackRequest, FeedbackStatsResponse
from src.storage import repository

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])

# 负反馈原因标签白名单：前端只提供这几个选项，后端只做长度兜底
_REASON_TAG_MAX = 20
_REASON_TAGS_MAX = 6


@router.post("", response_model=FeedbackItem)
async def submit_feedback(req: FeedbackRequest):
    """提交/更新一条回答的反馈。

    message_id 来自 /chat 或 /chat/stream 的 final 事件；不存在或不是
    assistant 消息一律 404——避免把反馈挂到用户自己的提问上。
    """
    message = repository.get_message(req.message_id)
    if message is None:
        raise HTTPException(
            status_code=404, detail=f"消息不存在: {req.message_id}"
        )
    if message["role"] != "assistant":
        raise HTTPException(
            status_code=400, detail="只能对助手回答提交反馈"
        )

    tags = [
        str(t).strip()[:_REASON_TAG_MAX]
        for t in (req.reason_tags or [])[:_REASON_TAGS_MAX]
        if str(t).strip()
    ]

    item = repository.save_feedback(
        message_id=req.message_id,
        session_id=message["session_id"],
        rating=req.rating,
        reason_tags=tags,
        comment=(req.comment or "").strip()[:1000],
    )
    logger.info(
        f"[反馈] message={req.message_id} rating={req.rating} "
        f"tags={tags} session={message['session_id']}"
    )
    return FeedbackItem(**item)


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(days: int = 7):
    """反馈看板：`days` 为统计窗口（1–90 天）。"""
    return FeedbackStatsResponse(**repository.feedback_stats(days=days))
