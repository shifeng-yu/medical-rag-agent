# ============================================================
# FastAPI 路由定义
# ============================================================

from typing import List
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    BatchQueryRequest,
    HealthResponse,
    StatsResponse,
    SourceInfo,
    JudgeResult,
)
from src.core.workflow import workflow
from src.monitoring.logger import request_logger

router = APIRouter(prefix="/api/v1", tags=["medical-rag"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口
    """
    milvus_ok = False
    try:
        from config.settings import settings
        if settings.use_milvus_lite:
            from src.core.retriever_lite import lite_retriever
            lite_retriever.connect()
        else:
            from src.core.retriever import retriever
            retriever.connect()
        milvus_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        models_loaded=True,
        milvus_connected=milvus_ok,
    )


@router.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    """
    核心问诊接口
    输入: query + 可选 session_id
    输出: answer + sources + judge_result + latency
    """
    from src.core.concurrency import rate_limiter, graceful
    import asyncio

    user_id = req.session_id or "anonymous"
    allowed, reason = rate_limiter.acquire(user_id)
    if not allowed:
        logger.warning(f"[API] 限流: {reason} | user={user_id}")
        raise HTTPException(status_code=429, detail=reason)

    logger.info(f"[API] 收到问诊请求: {req.query[:80]}... | session={req.session_id}")

    try:
        result = await asyncio.wait_for(
            workflow.run(query=req.query, session_id=req.session_id),
            timeout=60.0  # 60s timeout per request
        )
    except asyncio.TimeoutError:
        logger.error(f"[API] 请求超时: {req.query[:60]}")
        result = graceful.timeout_response()
    except Exception as e:
        logger.error(f"[API] 工作流执行异常: {e}")
        result = graceful.overload_response()
    finally:
        rate_limiter.release(user_id)

    # 构建响应
    sources = [
        SourceInfo(
            title=s.get("title", ""),
            source=s.get("source", ""),
            score=s.get("score", 0),
            publish_time=s.get("publish_time", ""),
            department=s.get("department", ""),
        )
        for s in result.get("sources", [])
    ]

    judge_data = result.get("judge_result", {})
    judge_result = JudgeResult(
        layer=judge_data.get("layer", "unknown"),
        scores=judge_data.get("scores"),
        reason=judge_data.get("reason") or judge_data.get("feedback"),
        details=judge_data.get("details"),
    )

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        judge_result=judge_result,
        latency_ms=result["latency_ms"],
        session_id=result["session_id"],
    )


@router.post("/chat/batch", response_model=List[QueryResponse])
async def batch_chat(req: BatchQueryRequest):
    """批量问诊接口"""
    results = []
    for item in req.queries:
        try:
            result = await workflow.run(
                query=item.query,
                session_id=item.session_id,
            )
            results.append(QueryResponse(
                answer=result["answer"],
                sources=[
                    SourceInfo(**s) for s in result.get("sources", [])
                ],
                judge_result=JudgeResult(
                    **(result.get("judge_result", {"layer": "unknown"}))
                ),
                latency_ms=result["latency_ms"],
                session_id=result["session_id"],
            ))
        except Exception as e:
            logger.error(f"[API] 批量处理异常: {e}")
            results.append(QueryResponse(
                answer=f"处理失败: {str(e)}",
                sources=[],
                judge_result=JudgeResult(layer="error"),
                latency_ms=0,
                session_id=item.session_id or "",
            ))
    return results


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    获取运行统计 
    返回: 总请求数/成功率/平均响应时间/校验通过率
    """
    stats = request_logger.get_stats()
    return StatsResponse(**stats)


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清除指定会话 ( 24h过期或手动清除)"""
    from src.session.manager import session_manager
    session_manager.delete_session(session_id)
    logger.info(f"[API] 会话已清除: {session_id}")
    return {"status": "deleted", "session_id": session_id}
