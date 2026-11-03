# ============================================================
# FastAPI 路由定义
# ============================================================

import asyncio
import contextlib
import time
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
from src.core import outcome
from src.monitoring.logger import request_logger

router = APIRouter(prefix="/api/v1", tags=["medical-rag"])


def _consume_task_result(task: "asyncio.Task") -> None:
    """取一次被取消任务的结果。

    取消后无人 await 的 Task 会在 GC 时打印 "Task exception was never
    retrieved"；这里主动消费掉，保持日志干净。任务已取消时 exception()
    会抛 CancelledError，一并吞掉。
    """
    with contextlib.suppress(BaseException):
        task.exception()


def _build_query_response(result: dict) -> QueryResponse:
    """把问诊结果出口模块产出的固定 6 键形状（含 outcome 出口标记）映射为响应模型。

    显式逐字段映射（不做 **splat 展开）：即使未来结果 dict 新增键，
    单条 / 批量路由也不会因此崩掉。
    """
    sources = [
        SourceInfo(
            title=s.get("title", ""),
            source=s.get("source", ""),
            score=s.get("score", 0),
            publish_time=s.get("publish_time", ""),
            department=s.get("department", ""),
            authority_tier=s.get("authority_tier", ""),
            authority_label=s.get("authority_label", ""),
            doc_id=s.get("doc_id", ""),
            chunk_id=s.get("chunk_id", ""),
            snippet=s.get("snippet", ""),
        )
        for s in result.get("sources", [])
    ]
    judge_data = result.get("judge_result", {}) or {}
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
        message_id=str(result.get("message_id") or ""),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口
    """
    milvus_ok = False
    try:
        from src.core.retrieval import get_retriever
        get_retriever().connect()
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
        result = outcome.degrade(
            req.session_id or "anonymous",
            outcome.EXIT_TIMEOUT,
            graceful.timeout_message(),
        )
    except Exception as e:
        logger.error(f"[API] 工作流执行异常: {e}")
        result = outcome.degrade(
            req.session_id or "anonymous",
            outcome.EXIT_OVERLOAD,
            graceful.overload_message(),
            reason=str(e),
        )
    finally:
        rate_limiter.release(user_id)

    return _build_query_response(result)


@router.post("/chat/stream")
async def chat_stream(req: QueryRequest):
    """流式问诊接口（SSE）。

    事件词表见 src/core/stream_protocol.py：
        meta → stage* → risk → sources → delta* → judge → final
    低风险问题边生成边推（校验不过发 replace 整段替换）；
    高风险问题在 judge 之前不发 delta（先校验后回放）。

    为什么不用 EventSource：浏览器原生 EventSource 只支持 GET，塞不进
    请求体；前端用 fetch + ReadableStream 解析同样的帧格式。
    """
    from src.core import stream_protocol as sp
    from src.core.concurrency import graceful, rate_limiter
    from src.core.workflow import workflow
    from config.settings import settings

    user_id = req.session_id or "anonymous"
    allowed, reason = rate_limiter.acquire(user_id)
    if not allowed:
        logger.warning(f"[API] 限流(stream): {reason} | user={user_id}")
        raise HTTPException(status_code=429, detail=reason)

    async def frames():
        deadline = time.monotonic() + settings.stream_timeout_seconds
        session_id = req.session_id or "anonymous"
        iterator = workflow.stream(
            query=req.query, session_id=req.session_id
        ).__aiter__()
        pending = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(iterator.__anext__())

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending.cancel()
                    yield sp.sse_frame(sp.ev(
                        sp.EVENT_ERROR, code="timeout", message="问诊超时",
                    ))
                    result = outcome.degrade(
                        session_id, outcome.EXIT_TIMEOUT,
                        graceful.timeout_message(),
                    )
                    yield sp.sse_frame(sp.ev(sp.EVENT_FINAL, **result))
                    break

                # 心跳：等待下一个事件超过间隔就先发注释帧保活。
                # 用 asyncio.wait 而非 wait_for —— 后者超时会取消任务，
                # 相当于每个心跳都把工作流掐断一次。
                tick = min(remaining, max(1, settings.sse_heartbeat_seconds))
                done, _ = await asyncio.wait({pending}, timeout=tick)
                if not done:
                    yield sp.sse_comment()
                    continue

                task, pending = pending, None
                try:
                    event = task.result()
                except StopAsyncIteration:
                    break

                if event["event"] == sp.EVENT_META:
                    session_id = event["data"].get("session_id") or session_id
                yield sp.sse_frame(event)
        except asyncio.CancelledError:
            logger.info(f"[API] 客户端断开流式问诊: session={session_id}")
            raise
        except Exception as e:
            logger.error(f"[API] 流式问诊异常: {e}")
            yield sp.sse_frame(sp.ev(
                sp.EVENT_ERROR, code="stream_failed", message=str(e),
            ))
        finally:
            # 收尾不做任何 await：客户端断开时本生成器处于取消态，
            # finally 里再 await 会被立刻打断，反而漏掉限流槽释放。
            if pending is not None:
                pending.cancel()
                pending.add_done_callback(_consume_task_result)
            rate_limiter.release(user_id)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
            results.append(_build_query_response(result))
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
