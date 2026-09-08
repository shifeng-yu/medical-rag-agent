# ============================================================
# 主工作流编排模块
# 复现 RAGFlow 可视化图工作流的 Python 等价实现
#
# DAG 流程:
#   START → classify → [并行] retrieve_local + retrieve_pubmed
#        → rerank → assemble → generate
#        → rule_check → llm_judge → [判断] pass? → RETURN
#                                       ↓ fail (retry<2)
#                                      regenerate (含 query 扩展)
#                                       ↓ fail (retry>=2) or tool fail
#                                      degradation
#
# RunOptions：一次运行的环节开关（默认全开 == 在线行为）。
# 消融实验（scripts/ablation.py）通过 options 关闭 judge / reranker /
# source_weight / session，复用同一条链路做对照，不再复制第二份实现。
# ============================================================

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger
from config.settings import settings
from src.core.classifier import classifier
from src.core.retrieval import get_retriever, merge_dual_results
# 检索后端按配置选定（唯一选择逻辑在 retrieval.get_retriever，import 期决定）
retriever = get_retriever()
from src.core.reranker import reranker
from src.core.generator import generator
from src.core.judge import judge
from src.core.safety import safety_filter
from src.core.knowledge_grader import knowledge_grader
from src.core import outcome
from src.session.manager import session_manager
from src.monitoring.logger import RequestMetrics

import re as _re
# 短查询/问候模式 - 不触发检索链路
_GREETING_RE = _re.compile(
    r"^(\s*(你好|您好|hi|hello|hey|嗨|哈喽|在吗|喂|谢谢|再见|拜拜|哦|嗯|好的|收到|ok|OK)[\s\W]*)+$",
    _re.IGNORECASE,
)


@dataclass
class RunOptions:
    """一次问诊回合的环节开关。默认全开 == 与无 options 时完全一致。

    - use_judge:        幻觉校验（规则层 + LLM-Judge）。关闭后判定恒放行，
                        judge_result 占位 {"layer": "ablation_off"}
    - use_reranker:     两阶段重排。关闭后直接按原始检索顺序截断拼接
    - use_source_weight:场景化来源权重。仅在重排开启时有意义，
                        关闭后两库等权 (0.5, 0.5)
    - use_session:      会话管理（建会话/写历史/拼对话上下文）。
                        关闭 = 评测模式：不建会话、不写历史、上下文为空，
                        供消融/批量评测使用，避免污染会话存储
    """

    use_judge: bool = True
    use_reranker: bool = True
    use_source_weight: bool = True
    use_session: bool = True


class MedicalRAGWorkflow:
    """
    全科医疗问诊 RAG 智能助手主工作流 
    严格复现 RAGFlow 图工作流的节点调度逻辑
    """

    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        options: Optional[RunOptions] = None,
    ) -> Dict:
        """
        主入口: 执行完整 RAG 问诊流程 
        对应 RAGFlow 工作流的所有节点

        参数:
            options: 环节开关（None = 全开 = 在线行为）。
                     消融实验通过它关闭 judge/reranker/source_weight/session。

        返回:
        {
            "answer": str,
            "sources": [...],
            "judge_result": dict,
            "latency_ms": float,
            "session_id": str,
            "outcome": str,   # 出口 kind（ok/emergency_blocked/...），见 src/core/outcome.py
        }
        """
        options = options or RunOptions()
        metrics = RequestMetrics(session_id or "unknown")
        metrics.query = query

        # 出口统一收银：write_session 仅在会话模式开启时按 kind 规则落写
        def _finish(*, kind: str, answer: str, **kw) -> Dict:
            return outcome.finish(
                session_id, metrics, kind=kind, answer=answer,
                write_session=None if options.use_session else False,
                **kw,
            )

        try:
            # ---- 1. 会话管理  ----
            if options.use_session:
                session_id = session_manager.get_or_create_session(session_id)
                metrics.session_id = session_id
                session_manager.add_message(session_id, "user", query)
            else:
                # 评测模式：不建会话、不写历史，session_id 原样透传
                metrics.session_id = session_id or "eval"

            # ---- 1b. 安全合规检查（急症拦截 + 敏感内容过滤） ----
            is_emergency, em_label, em_response = safety_filter.check_emergency(query)
            if is_emergency:
                return _finish(
                    kind=outcome.EXIT_EMERGENCY_BLOCKED,
                    answer=em_response,
                    judge_result={"reason": em_label},
                )
            is_blocked, block_label, block_response = safety_filter.check_input_safety(query)
            if is_blocked:
                return _finish(
                    kind=outcome.EXIT_CONTENT_BLOCKED,
                    answer=block_response,
                    judge_result={"reason": block_label},
                )

            # ---- 1c. 短查询/问候/无效输入短路 ----
            # 避免"你好""hi""嗯"等无意义输入触发完整检索/生成链路
            stripped = query.strip()
            if len(stripped) < 3 or _GREETING_RE.search(stripped):
                return _finish(
                    kind=outcome.EXIT_OK,
                    answer="你好，我是全科医疗问诊助手。请描述具体的症状（如\"胸闷 2 小时\"\"最近经常头晕\"），我会结合知识库给出科普参考。",
                    judge_result={"reason": "greeting_or_too_short"},
                )

            # 医学术语标准化
            normalized_query = knowledge_grader.normalize_query(query)

            # 构建对话上下文 ( 含压缩逻辑)
            if options.use_session:
                conversation_context, conv_tokens = session_manager.build_context(
                    session_id, query
                )
            else:
                conversation_context, conv_tokens = "", 0

            # ---- 2. 问题分类 (分类条件分支节点) ----
            classification = classifier.classify(query)
            metrics.classification_label = classification
            need_high_judge = classifier.should_trigger_high_judge(query)

            # 来源权重 ( 场景化权重，可在消融中关闭 → 两库等权)
            source_weights = (0.5, 0.5)
            if options.use_reranker and options.use_source_weight:
                source_weights = classifier.get_source_weight(query, classification)

            # ---- 3. 双源并行检索 (并行检索节点) ----
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
                        # 降级 
                        local_results, pubmed_results = [], []
                        metrics.retrieval_latency_ms = 0
                    await asyncio.sleep(0.5)

            all_empty = not local_results and not pubmed_results
            if all_empty:
                # 工具完全不可用 → 降级
                degraded = generator.generate_degraded(query)
                return _finish(
                    kind=outcome.EXIT_DEGRADED,
                    answer=degraded,
                )

            # ---- 3b. 术语标准化补充检索（扩词式接入） ----
            # normalized_query 与原始 query 不一致（发生别名→标准术语替换）时，
            # 用标准化词补检索一次并并入结果：主 query 保持原样不丢口语原意，
            # 标准词兜底提高对"标准术语知识库"的召回。补充检索失败只告警不致命。
            if normalized_query != query:
                try:
                    extra_local, extra_pubmed, extra_lat = (
                        await retriever.retrieve(normalized_query, classification)
                    )
                    metrics.retrieval_latency_ms += extra_lat
                    local_results, pubmed_results = merge_dual_results(
                        local_results, pubmed_results,
                        extra_local, extra_pubmed,
                    )
                    logger.info(
                        f"术语标准化扩词检索: '{query}' → '{normalized_query}' "
                        f"(local+{len(extra_local)}, pubmed+{len(extra_pubmed)})"
                    )
                except Exception as e:
                    logger.warning(f"术语标准化补充检索失败(忽略): {e}")

            # ---- 4. 重排序 (BGE-M3重排序节点，可消融) ----
            if options.use_reranker:
                reranked = reranker.rerank(
                    query, local_results, pubmed_results,
                    source_weights=source_weights,
                )
            else:
                # 无重排对照组：按原始检索顺序直接拼接截断
                reranked = (local_results + pubmed_results)[: settings.rerank_top_k]
            context_text = reranker.build_context_text(reranked)

            # ---- 5. 生成 + 校验 + 重试循环 ( 生成→校验→判断→重生成) ----
            final_answer = ""
            judge_result = {}
            gen_latency = 0.0

            for retry in range(settings.max_retry_generation + 1):
                metrics.retry_count = retry

                # 5a. 大模型生成节点 
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
                            if options.use_reranker:
                                reranked2 = reranker.rerank(
                                    query, local_results2, pubmed_results2,
                                    source_weights=source_weights,
                                )
                            else:
                                reranked2 = (
                                    local_results2 + pubmed_results2
                                )[: settings.rerank_top_k]
                            context_text = reranker.build_context_text(reranked2)
                        except Exception:
                            pass  # 保持原上下文

                answer, gen_latency = generator.generate(
                    query=query,
                    context_text=context_text,
                    conversation_context=conversation_context,
                )
                metrics.generation_latency_ms += gen_latency

                # 5b. 幻觉校验节点 (规则+LLM-Judge，可消融)
                if options.use_judge:
                    passed, j_result, feedback = judge.validate(
                        query=query,
                        answer=answer,
                        retrieved_contexts=reranked,
                        trigger_high_judge=need_high_judge,
                    )
                else:
                    # 无校验对照组：恒放行，judge_result 占位标记消融状态
                    passed, j_result, feedback = (
                        True, {"layer": "ablation_off"}, ""
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

            # ---- 6. 统一收尾（写会话 + 上报指标 + 拼装结果）----
            result = _finish(
                kind=outcome.EXIT_OK,
                answer=final_answer,
                sources=outcome.to_sources(reranked),
                judge_result=judge_result,
            )

            logger.info(
                f"[工作流完成] latency={result['latency_ms']}ms, "
                f"retry={metrics.retry_count}, "
                f"classification={classification}, "
                f"judge={metrics.judge_passed}"
            )
            return result

        except Exception as e:
            logger.error(f"工作流异常: {e}")
            # 系统失败出口：不写会话、记一次指标、返回固定形状
            fallback = (
                "抱歉，系统当前遇到技术问题，暂时无法处理您的问诊请求。"
                "建议您稍后重试或咨询专业医生。"
            )
            return _finish(
                kind=outcome.EXIT_ERROR,
                answer=fallback,
                error_message=str(e),
            )

    # ========== 同步封装 (FastAPI 调用) ==========

    def run_sync(self, query: str, session_id: Optional[str] = None) -> Dict:
        """同步运行工作流 (供 FastAPI 路由调用)"""
        return asyncio.run(self.run(query, session_id))


# 全局单例
workflow = MedicalRAGWorkflow()
