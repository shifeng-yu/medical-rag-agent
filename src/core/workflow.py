# ============================================================
# 主工作流编排模块
# 复现 RAGFlow 可视化图工作流 (项目需求) 的 Python 等价实现
#
# DAG 流程:
#   START → classify → [并行] retrieve_local + retrieve_pubmed
#        → rerank → assemble → generate
#        → rule_check → llm_judge → [判断] pass? → RETURN
#                                       ↓ fail (retry<2)
#                                      regenerate (含 query 扩展)
#                                       ↓ fail (retry>=2) or tool fail
#                                      degradation
# ============================================================

import time
import asyncio
from typing import Dict, Optional, List, Tuple
from loguru import logger
from config.settings import settings
from src.core.classifier import classifier
# CPU/GPU 自适应: 有Milvus Docker用pymilvus, 否则用Milvus Lite
from config.settings import settings
if settings.use_milvus_lite:
    from src.core.retriever_lite import lite_retriever as retriever
else:
    from src.core.retriever import retriever
from src.core.reranker import reranker
from src.core.generator import generator
from src.core.judge import judge
from src.core.safety import safety_filter
from src.core.knowledge_grader import knowledge_grader
from src.session.manager import session_manager
from src.monitoring.logger import request_logger, RequestMetrics


class MedicalRAGWorkflow:
    """
    全科医疗问诊 RAG 智能助手主工作流 (项目需求)
    严格复现 RAGFlow 图工作流的节点调度逻辑
    """

    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        主入口: 执行完整 RAG 问诊流程 (项目需求)
        对应 RAGFlow 工作流的所有节点

        返回:
        {
            "answer": str,
            "sources": [...],
            "judge_result": dict,
            "latency_ms": float,
            "session_id": str,
        }
        """
        metrics = RequestMetrics(session_id or "unknown")
        metrics.query = query

        try:
            # ---- 1. 会话管理 (项目需求) ----
            session_id = session_manager.get_or_create_session(session_id)
            metrics.session_id = session_id
            session_manager.add_message(session_id, "user", query)

            # ---- 1b. 安全合规检查（急症拦截 + 敏感内容过滤） ----
            is_emergency, em_label, em_response = safety_filter.check_emergency(query)
            if is_emergency:
                session_manager.add_message(session_id, "assistant", em_response)
                return {
                    "answer": em_response, "sources": [],
                    "judge_result": {"layer": "emergency_blocked", "reason": em_label},
                    "latency_ms": (time.time() - metrics.start_time) * 1000,
                    "session_id": session_id,
                }
            is_blocked, block_label, block_response = safety_filter.check_input_safety(query)
            if is_blocked:
                session_manager.add_message(session_id, "assistant", block_response)
                return {
                    "answer": block_response, "sources": [],
                    "judge_result": {"layer": "content_blocked", "reason": block_label},
                    "latency_ms": (time.time() - metrics.start_time) * 1000,
                    "session_id": session_id,
                }

            # 医学术语标准化
            normalized_query = knowledge_grader.normalize_query(query)

            # 构建对话上下文 ( 含压缩逻辑)
            conversation_context, conv_tokens = session_manager.build_context(
                session_id, query
            )

            # ---- 2. 问题分类 (项目需求/ 分类条件分支节点) ----
            classification = classifier.classify(query)
            metrics.classification_label = classification
            need_high_judge = classifier.should_trigger_high_judge(query)

            # 获取来源权重 ( 场景化来源权重)
            source_weights = classifier.get_source_weight(query, classification)

            # ---- 3. 双源并行检索 (项目需求/ 并行检索节点) ----
            for attempt in range(settings.max_retry_tool_call + 1):
                try:
                    local_results, pubmed_results, retrieval_latency = (
                        await retriever.retrieve(query, classification)
                    )
                    metrics.retrieval_latency_ms = retrieval_latency
                    metrics.retrieval_count = len(local_results) + len(pubmed_results)
                    break
                except Exception as e:
                    logger.warning(
                        f"检索失败 (attempt {attempt + 1}): {e}"
                    )
                    if attempt >= settings.max_retry_tool_call:
                        # 降级 (项目需求)
                        local_results, pubmed_results = [], []
                        metrics.retrieval_latency_ms = 0
                    await asyncio.sleep(0.5)

            all_empty = not local_results and not pubmed_results
            if all_empty:
                # 工具完全不可用 → 降级
                metrics.success = True
                degraded = generator.generate_degraded(query)
                session_manager.add_message(session_id, "assistant", degraded)
                request_logger.log_request(metrics)
                return {
                    "answer": degraded,
                    "sources": [],
                    "judge_result": {"layer": "degraded"},
                    "latency_ms": (time.time() - metrics.start_time) * 1000,
                    "session_id": session_id,
                }

            # ---- 4. 重排序 (项目需求3/ BGE-M3重排序节点) ----
            reranked = reranker.rerank(
                query, local_results, pubmed_results,
                source_weights=source_weights,
            )
            context_text = reranker.build_context_text(reranked)

            # ---- 5. 生成 + 校验 + 重试循环 ( 生成→校验→判断→重生成) ----
            final_answer = ""
            judge_result = {}
            gen_latency = 0.0

            for retry in range(settings.max_retry_generation + 1):
                metrics.retry_count = retry

                # 5a. 大模型生成节点 (项目需求)
                if retry > 0:
                    # 根据校验反馈重生成 / 调整检索关键词
                    expanded_query = (
                        retriever.expand_query(query)
                        if retry == 1
                        else query
                    )
                    # 重新检索 ( 自动调整检索关键词)
                    if retry == 1:
                        try:
                            local_results2, pubmed_results2, _ = (
                                await retriever.retrieve(expanded_query, "both")
                            )
                            reranked2 = reranker.rerank(
                                query, local_results2, pubmed_results2,
                                source_weights=source_weights,
                            )
                            context_text = reranker.build_context_text(reranked2)
                        except Exception:
                            pass  # 保持原上下文

                answer, gen_latency = generator.generate(
                    query=query,
                    context_text=context_text,
                    conversation_context=conversation_context,
                )
                metrics.generation_latency_ms += gen_latency

                # 5b. 幻觉校验节点 (项目需求/ 规则+LLM-Judge)
                passed, j_result, feedback = judge.validate(
                    query=query,
                    answer=answer,
                    retrieved_contexts=reranked,
                    trigger_high_judge=need_high_judge,
                )
                judge_result = j_result

                if "scores" in j_result:
                    metrics.judge_scores = j_result["scores"]
                    metrics.judge_passed = passed

                if passed:
                    final_answer = answer
                    break
                else:
                    logger.info(
                        f"[重试 {retry + 1}] 校验未通过: {feedback[:100]}"
                    )
                    # 校验不通过 → 自动回退到生成节点
                    if retry >= settings.max_retry_generation:
                        # 仍不通过 → 返回无法回答
                        final_answer = (
                            "抱歉，根据当前可用的医疗参考资料，我无法对您的问题"
                            "给出准确可靠的回答。建议您咨询专业医生获取进一步帮助。"
                        )
                        break

            # ---- 5c. 输出合规校验（拦截诊断/处方/剂量） ----
            output_ok, violations = safety_filter.check_output_compliance(
                final_answer, reranked
            )
            if not output_ok:
                for v in violations:
                    logger.warning(f"[合规拦截] {v['type']}: {v['detail']}")
                final_answer = safety_filter.get_refusal_response("out_of_scope")

            # ---- 6. 存储对话记录 (项目需求) ----
            session_manager.add_message(session_id, "assistant", final_answer)

            # ---- 7. 记录监控指标 (项目需求) ----
            metrics.success = True
            metrics.final_output = final_answer
            request_logger.log_request(metrics)

            total_latency = (time.time() - metrics.start_time) * 1000
            logger.info(
                f"[工作流完成] latency={total_latency:.0f}ms, "
                f"retry={metrics.retry_count}, "
                f"classification={classification}, "
                f"judge={metrics.judge_passed}"
            )

            return {
                "answer": final_answer,
                "sources": [
                    {
                        "title": r.get("title", ""),
                        "source": r.get("source", ""),
                        "score": r.get("rerank_score", 0),
                        "publish_time": r.get("publish_time", ""),
                        "department": r.get("department", ""),
                    }
                    for r in reranked[:5]
                ],
                "judge_result": judge_result,
                "latency_ms": round(total_latency, 2),
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"工作流异常: {e}")
            metrics.success = False
            metrics.error_message = str(e)
            request_logger.log_request(metrics)

            # 最终降级
            fallback = (
                "抱歉，系统当前遇到技术问题，暂时无法处理您的问诊请求。"
                "建议您稍后重试或咨询专业医生。"
            )
            return {
                "answer": fallback,
                "sources": [],
                "judge_result": {"layer": "error"},
                "latency_ms": (time.time() - metrics.start_time) * 1000,
                "session_id": session_id,
                "error": str(e),
            }

    # ========== 同步封装 (FastAPI 调用) ==========

    def run_sync(self, query: str, session_id: Optional[str] = None) -> Dict:
        """同步运行工作流 (供 FastAPI 路由调用)"""
        return asyncio.run(self.run(query, session_id))


# 全局单例
workflow = MedicalRAGWorkflow()
