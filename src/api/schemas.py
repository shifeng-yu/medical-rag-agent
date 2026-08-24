# ============================================================
# FastAPI Pydantic 请求/响应模型
# 来源：「API输入输出格式与baseline一致」
# ============================================================

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# ---- 请求模型 ----

class QueryRequest(BaseModel):
    """问诊请求 ( 与baseline统一格式)"""
    query: str = Field(
        ...,
        description="用户的问诊问题",
        min_length=1,
        max_length=2000,
        example="我最近经常头痛，伴有恶心，是什么问题？",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，不传则自动创建新会话 (项目需求)",
        example="550e8400-e29b-41d4-a716-446655440000",
    )


class BatchQueryRequest(BaseModel):
    """批量问诊请求"""
    queries: List[QueryRequest] = Field(
        ..., description="批量问诊请求列表", max_length=50
    )


# ---- 响应模型 ----

class SourceInfo(BaseModel):
    """检索来源信息 ( 标注来源)"""
    title: str = Field(default="", description="文档标题")
    source: str = Field(description="来源: local_kb / pubmed")
    score: float = Field(description="相关性分数")
    publish_time: str = Field(default="", description="发布时间")
    department: str = Field(default="", description="对应科室")


class JudgeResult(BaseModel):
    """校验结果 (项目需求)"""
    layer: str = Field(description="校验层级: rule / judge / both / degraded / error")
    scores: Optional[Dict[str, float]] = Field(
        default=None, description="LLM-Judge三维打分"
    )
    reason: Optional[str] = Field(default=None, description="未通过原因")
    details: Optional[Dict] = Field(default=None, description="规则校验详情")


class QueryResponse(BaseModel):
    """问诊响应 ( 与baseline统一格式)"""
    answer: str = Field(description="生成的问诊回答")
    sources: List[SourceInfo] = Field(
        default_factory=list, description="检索来源列表"
    )
    judge_result: JudgeResult = Field(description="幻觉校验结果")
    latency_ms: float = Field(description="总响应耗时(毫秒)")
    session_id: str = Field(description="会话ID")


class ErrorResponse(BaseModel):
    """标准化错误响应"""
    error: str = Field(description="错误类型")
    message: str = Field(description="错误详情")
    retry_after: int = Field(default=0, description="建议重试等待秒数")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    models_loaded: bool = Field(default=True)
    milvus_connected: bool = Field(default=True)


class StatsResponse(BaseModel):
    """统计指标响应 (项目需求)"""
    total_requests: int
    success_rate: float
    avg_latency_ms: float
    judge_pass_rate: float
