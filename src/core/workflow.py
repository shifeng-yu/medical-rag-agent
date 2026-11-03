# ============================================================
# 主工作流编排模块
# 复现 RAGFlow 可视化图工作流的 Python 等价实现
#
# DAG 流程:
#   START → safety → classify → [并行] retrieve_local + retrieve_pubmed
#        → rerank → assemble → generate
#        → rule_check → llm_judge → [判断] pass? → RETURN
#                                       ↓ fail (retry<2)
#                                      regenerate (含 query 扩展)
#                                       ↓ fail (retry>=2) or tool fail
#                                      degradation
#
# 两个驱动，一条链：
#   stream() 是原语，产出 SSE 事件（见 src/core/stream_protocol.py）；
#   run()    消费 stream() 并只取 final —— 阻塞接口与评测链路不会
#            另写一份 RAG 流程（ADR-0004 只维护一条链）。
#   emit_tokens=False 时不开 token 流（评测/批量），链路节点完全一致。
#
# 结构约定：stream() 只做调度，依次跑四个节点方法
#   _open_turn → _retrieve_and_rank → _generate_and_judge → _check_compliance
# 每个节点是 async generator，自己产出事件、把结果写进 _Turn；
# 需要提前结束的节点把 final 事件放进 turn.final_event，stream() 见了就收尾。
# 加节点/调顺序只改 stream()，不必在几百行里找状态变量。
#
# RunOptions：一次运行的环节开关（默认全开 == 在线行为）。
# 消融实验（scripts/ablation.py）通过 options 关闭 judge / reranker /
# source_weight / session，复用同一条链路做对照，不再复制第二份实现。
# ============================================================

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple

from loguru import logger

from config.settings import settings
from src.core import outcome
from src.core import stream_protocol as sp
from src.core.classifier import classifier
from src.core.generator import generator
from src.core.judge import judge
from src.core.knowledge_grader import knowledge_grader
# 检索后端按配置选定（唯一选择逻辑在 retrieval.get_retriever，import 期决定）
from src.core.reranker import reranker
from src.core.retrieval import get_retriever, merge_dual_results
retriever = get_retriever()
from src.core.safety import safety_filter
from src.monitoring.logger import RequestMetrics
from src.session.manager import session_manager

import re as _re
# 短查询/问候模式 - 不触发检索链路
_GREETING_RE = _re.compile(
    r"^(\s*(你好|您好|hi|hello|hey|嗨|哈喽|在吗|喂|谢谢|再见|拜拜|哦|嗯|好的|收到|ok|OK)[\s\W]*)+$",
    _re.IGNORECASE,
)

GREETING_TEXT = (
    "你好，我是全科医疗问诊助手。请描述具体的症状"
    "（如\"胸闷 2 小时\"\"最近经常头晕\"），我会结合知识库给出科普参考。"
)

# 校验不过且重试耗尽 / 合规拦截时的标准拒答
REFUSAL_TEXT = (
    "抱歉，根据当前可用的医疗参考资料，我无法对您的问题"
    "给出准确可靠的回答。建议您咨询专业医生获取进一步帮助。"
)

FALLBACK_TEXT = (
    "抱歉，系统当前遇到技术问题，暂时无法处理您的问诊请求。"
    "建议您稍后重试或咨询专业医生。"
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


@dataclass
class _Turn:
    """一次问诊回合的可变状态。

    存在的意义：节点之间要传的东西太多（分类/风险/上下文/来源/最终答案/
    是否待替换…），靠嵌套函数改外部变量会把 stream() 变成 300 行的状态机。
    统一放这里，节点只读写字段，调度只负责顺序。
    """

    query: str
    options: RunOptions
    metrics: RequestMetrics
    emit_tokens: bool
    session_id: Optional[str] = None

    # 前置节点产出
    normalized_query: str = ""
    conversation_context: str = ""
    classification: str = ""
    need_high_judge: bool = False
    risk: str = ""
    live: bool = False
    source_weights: Tuple[float, float] = (0.5, 0.5)

    # 检索节点产出
    reranked: List[Dict] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    context_text: str = ""

    # 生成/校验产出
    final_answer: str = ""
    judge_result: Dict = field(default_factory=dict)
    # 前端已显示的内容被判定不合格：下一次成功必须整段替换
    pending_replace: bool = False

    # 提前出口（拦截/降级）：stream() 见到就收尾
    final_event: Optional[Dict] = None

    def finish(self, kind: str, answer: str, **extra) -> Dict:
        """统一结账（写会话 / 记指标 / 拼 7 键返回）。"""
        return outcome.finish(
            self.session_id,
            self.metrics,
            kind=kind,
            answer=answer,
            write_session=None if self.options.use_session else False,
            **extra,
        )


# ---------------- 同步迭代器 → 异步事件循环 ----------------

_DONE = object()


def _touch_first_token(metrics: RequestMetrics) -> None:
    """记录首字延迟（首个真正下发给前端的字符）。

    只记第一次：高风险链路记的是"校验通过后第一个字"，低风险记的是
    "模型吐出的第一个字"——两者口径不同，靠 metrics.stream_mode 区分。
    """
    if metrics.first_token_ms == 0.0:
        metrics.first_token_ms = round(
            (time.time() - metrics.start_time) * 1000, 2
        )


async def _aiter_sync(iterator) -> AsyncIterator[str]:
    """把同步生成器搬进事件循环。

    Qwen 的流式实现是「后台推理线程 + 阻塞队列」，这里每个 next() 用
    asyncio.to_thread 等待，避免一条正在生成的连接卡住其余连接的心跳
    与事件下发。

    刻意不在 finally 里 close() 上游生成器：客户端断开时 next() 可能仍在
    工作线程里执行，从事件循环线程 close() 会撞上 "generator already
    executing"。让它在推理线程里自然结束、由 GC 回收更安全（生成器的
    finally 仍会执行，用来回收推理线程）。
    """
    while True:
        item = await asyncio.to_thread(next, iterator, _DONE)
        if item is _DONE:
            break
        yield item


class MedicalRAGWorkflow:
    """全科医疗问诊 RAG 智能助手主工作流。

    stream() 只调度，节点方法见下方四个 `_xxx`；两者都不直接拼返回 dict，
    收尾一律走 _Turn.finish → outcome.finish。
    """

    # ==================== 流式原语（只调度） ====================

    async def stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        options: Optional[RunOptions] = None,
        *,
        emit_tokens: bool = True,
    ) -> AsyncIterator[Dict]:
        """产出流式事件（事件词表见 stream_protocol）。

        emit_tokens=False：不开 token 流，只产出 meta/stage/sources/judge/final。
        评测与 /chat 走这条路径，链路与在线流式完全一致。
        """
        turn = _Turn(
            query=query,
            options=options or RunOptions(),
            metrics=RequestMetrics(session_id or "unknown"),
            emit_tokens=emit_tokens,
            session_id=session_id,
        )
        turn.metrics.query = query
        turn.metrics.stream_mode = "off" if not emit_tokens else "token"

        try:
            for node in (
                self._open_turn,
                self._retrieve_and_rank,
                self._generate_and_judge,
                self._check_compliance,
            ):
                async for event in node(turn):
                    yield event
                if turn.final_event is not None:
                    yield turn.final_event
                    return

            yield self._finalize(turn)

        except Exception as e:
            logger.error(f"工作流异常: {e}")
            # 系统失败出口：不写会话、记一次指标、返回固定形状。
            # 已经下发过部分正文时必须用 replace 覆盖，避免前端残留半句。
            result = turn.finish(
                outcome.EXIT_ERROR, FALLBACK_TEXT, error_message=str(e)
            )
            if emit_tokens:
                yield sp.ev(sp.EVENT_REPLACE, text=FALLBACK_TEXT, reason="error")
            yield sp.ev(sp.EVENT_FINAL, **result)

    # ==================== 节点 1：会话与入口拦截 ====================

    async def _open_turn(self, turn: _Turn) -> AsyncIterator[Dict]:
        """建会话 → 急症/敏感/问候短路 → 术语标准化 + 上下文。"""
        metrics, options, query = turn.metrics, turn.options, turn.query

        # ---- 会话管理 ----
        if options.use_session:
            turn.session_id = session_manager.get_or_create_session(turn.session_id)
            metrics.session_id = turn.session_id
            session_manager.add_message(turn.session_id, "user", query)
        else:
            # 评测模式：不建会话、不写历史，session_id 原样透传
            metrics.session_id = turn.session_id or "eval"

        yield sp.ev(sp.EVENT_META, session_id=turn.session_id)

        # ---- 安全合规检查（急症拦截 + 敏感内容过滤） ----
        yield sp.ev(sp.EVENT_STAGE, node=sp.NODE_SAFETY, status="start")
        is_emergency, em_label, em_response = safety_filter.check_emergency(query)
        if is_emergency:
            async for event in self._exit(
                turn, outcome.EXIT_EMERGENCY_BLOCKED, em_response,
                judge_result={"reason": em_label},
            ):
                yield event
            return

        is_blocked, block_label, block_response = safety_filter.check_input_safety(query)
        if is_blocked:
            async for event in self._exit(
                turn, outcome.EXIT_CONTENT_BLOCKED, block_response,
                judge_result={"reason": block_label},
            ):
                yield event
            return

        # ---- 短查询/问候/无效输入短路 ----
        # 避免"你好""hi""嗯"等无意义输入触发完整检索/生成链路
        stripped = query.strip()
        if len(stripped) < 3 or _GREETING_RE.search(stripped):
            async for event in self._exit(
                turn, outcome.EXIT_OK, GREETING_TEXT,
                judge_result={"reason": "greeting_or_too_short"},
            ):
                yield event
            return

        yield sp.ev(sp.EVENT_STAGE, node=sp.NODE_SAFETY, status="done")

        # 医学术语标准化（别名 → 标准术语，供补检索用）
        turn.normalized_query = knowledge_grader.normalize_query(query)

        # 构建对话上下文（含压缩逻辑）
        if options.use_session:
            turn.conversation_context, _ = session_manager.build_context(
                turn.session_id, query
            )

    # ==================== 节点 2：分类 → 双源检索 → 重排 ====================

    async def _retrieve_and_rank(self, turn: _Turn) -> AsyncIterator[Dict]:
        """问题分类、流式风险分级、双源检索（含扩词补检）、重排与来源下发。"""
        query, options = turn.query, turn.options
        metrics = turn.metrics

        # ---- 问题分类（分类条件分支节点） ----
        yield sp.ev(sp.EVENT_STAGE, node=sp.NODE_CLASSIFY, status="start")
        classify_start = time.perf_counter()
        turn.classification = await asyncio.to_thread(classifier.classify, query)
        metrics.classification_label = turn.classification
        turn.need_high_judge = classifier.should_trigger_high_judge(query)
        yield sp.ev(
            sp.EVENT_STAGE, node=sp.NODE_CLASSIFY, status="done",
            label=turn.classification,
            ms=round((time.perf_counter() - classify_start) * 1000, 2),
        )

        # 流式策略：高风险先校验后回放，低风险边生成边推
        turn.risk = sp.assess_risk(
            query, turn.classification, turn.need_high_judge
        )
        turn.live = turn.emit_tokens and turn.risk == sp.RISK_LOW
        metrics.risk_level = turn.risk
        metrics.stream_mode = (
            ("token" if turn.live else "buffered")
            if turn.emit_tokens else "off"
        )
        yield sp.ev(sp.EVENT_RISK, level=turn.risk, mode=metrics.stream_mode)

        # 来源权重（场景化权重，可在消融中关闭 → 两库等权）
        if options.use_reranker and options.use_source_weight:
            turn.source_weights = classifier.get_source_weight(
                query, turn.classification
            )

        # ---- 双源并行检索（并行检索节点） ----
        yield sp.ev(sp.EVENT_STAGE, node=sp.NODE_RETRIEVE, status="start")
        retrieve_start = time.perf_counter()
        local_results, pubmed_results = [], []
        for attempt in range(settings.max_retry_tool_call + 1):
            try:
                local_results, pubmed_results, retrieval_latency = (
                    await retriever.retrieve(query, turn.classification)
                )
                metrics.retrieval_latency_ms = retrieval_latency
                metrics.retrieval_count = len(local_results) + len(pubmed_results)
                break
            except Exception as e:
                logger.warning(f"检索失败 (attempt {attempt + 1}): {e}")
                if attempt >= settings.max_retry_tool_call:
                    # 降级
                    local_results, pubmed_results = [], []
                    metrics.retrieval_latency_ms = 0
                else:
                    await asyncio.sleep(0.5)
        yield sp.ev(
            sp.EVENT_STAGE, node=sp.NODE_RETRIEVE, status="done",
            count=metrics.retrieval_count,
            ms=round((time.perf_counter() - retrieve_start) * 1000, 2),
        )

        if not local_results and not pubmed_results:
            # 工具完全不可用 → 降级
            degraded = await asyncio.to_thread(generator.generate_degraded, query)
            async for event in self._exit(turn, outcome.EXIT_DEGRADED, degraded):
                yield event
            return

        # ---- 术语标准化补充检索（扩词式接入） ----
        # normalized_query 与原始 query 不一致（发生别名→标准术语替换）时，
        # 用标准化词补检索一次并并入结果：主 query 保持原样不丢口语原意，
        # 标准词兜底提高对"标准术语知识库"的召回。补充检索失败只告警不致命。
        if turn.normalized_query != query:
            try:
                extra_local, extra_pubmed, extra_lat = (
                    await retriever.retrieve(turn.normalized_query, turn.classification)
                )
                metrics.retrieval_latency_ms += extra_lat
                local_results, pubmed_results = merge_dual_results(
                    local_results, pubmed_results,
                    extra_local, extra_pubmed,
                )
                logger.info(
                    f"术语标准化扩词检索: '{query}' → '{turn.normalized_query}' "
                    f"(local+{len(extra_local)}, pubmed+{len(extra_pubmed)})"
                )
            except Exception as e:
                logger.warning(f"术语标准化补充检索失败(忽略): {e}")

        # ---- 重排序（BGE-M3重排序节点，可消融） ----
        yield sp.ev(sp.EVENT_STAGE, node=sp.NODE_RERANK, status="start")
        rerank_start = time.perf_counter()
        if options.use_reranker:
            turn.reranked = await asyncio.to_thread(
                reranker.rerank, query, local_results, pubmed_results,
                source_weights=turn.source_weights,
            )
        else:
            # 无重排对照组：按原始检索顺序直接拼接截断
            turn.reranked = (local_results + pubmed_results)[
                : settings.rerank_top_k
            ]
        turn.context_text = reranker.build_context_text(turn.reranked)
        turn.sources = outcome.to_sources(turn.reranked)
        yield sp.ev(
            sp.EVENT_STAGE, node=sp.NODE_RERANK, status="done",
            ms=round((time.perf_counter() - rerank_start) * 1000, 2),
        )
        # 来源先于正文下发：前端右侧溯源栏可以在等待生成时就绪
        yield sp.ev(
            sp.EVENT_SOURCES, sources=turn.sources, count=len(turn.sources)
        )

    # ==================== 节点 3：生成 → 校验 → 重试 ====================

    async def _generate_and_judge(self, turn: _Turn) -> AsyncIterator[Dict]:
        """生成 → 校验 → 不过则重生成（≤2 次）；仍不过给标准拒答。"""
        query, options, metrics = turn.query, turn.options, turn.metrics

        for retry in range(settings.max_retry_generation + 1):
            metrics.retry_count = retry

            if retry > 0:
                yield sp.ev(
                    sp.EVENT_STAGE, node=sp.NODE_GENERATE,
                    status="retry", attempt=retry,
                )
                # 根据校验反馈重生成 / 调整检索关键词
                if retry == 1:
                    await self._expand_and_reretrieve(turn)

            # ---- 大模型生成节点 ----
            yield sp.ev(
                sp.EVENT_STAGE, node=sp.NODE_GENERATE,
                status="start", attempt=retry,
            )
            gen_start = time.perf_counter()
            # 重生成时屏幕上已有被判不合格的旧答案：不能再追加 delta
            # （会拼出两段答案），改为攒完一次性 replace。
            attempt_live = turn.live and not turn.pending_replace
            if attempt_live:
                # 低风险：边生成边推，累积后交给校验
                buffer = []
                iterator = generator.generate_stream(
                    query=query,
                    context_text=turn.context_text,
                    conversation_context=turn.conversation_context,
                )
                async for piece in _aiter_sync(iterator):
                    _touch_first_token(metrics)
                    buffer.append(piece)
                    yield sp.ev(sp.EVENT_DELTA, text=piece)
                answer = "".join(buffer)
            else:
                # 高风险：先攒完再校验（阻塞调用搬进线程，避免卡事件循环）
                answer, _ = await asyncio.to_thread(
                    generator.generate,
                    query=query,
                    context_text=turn.context_text,
                    conversation_context=turn.conversation_context,
                )
            metrics.generation_latency_ms += (
                time.perf_counter() - gen_start
            ) * 1000

            # ---- 幻觉校验节点（规则 + LLM-Judge，可消融） ----
            if options.use_judge:
                passed, j_result, feedback = await asyncio.to_thread(
                    judge.validate,
                    query=query,
                    answer=answer,
                    retrieved_contexts=turn.reranked,
                    trigger_high_judge=turn.need_high_judge,
                )
            else:
                # 无校验对照组：恒放行，judge_result 占位标记消融状态
                passed, j_result, feedback = (
                    True, {"layer": "ablation_off"}, ""
                )
            turn.judge_result = j_result

            if "scores" in j_result:
                metrics.judge_scores = j_result["scores"]
                metrics.judge_passed = passed

            yield sp.ev(
                sp.EVENT_JUDGE,
                layer=j_result.get("layer", ""),
                passed=passed,
                scores=j_result.get("scores"),
                retry=retry,
                reason=(feedback or j_result.get("reason") or "")[:200],
            )

            if passed:
                turn.final_answer = answer
                async for event in self._emit_verified(turn, answer):
                    yield event
                return

            logger.info(f"[重试 {retry + 1}] 校验未通过: {feedback[:100]}")
            if turn.live:
                turn.pending_replace = True
            if retry >= settings.max_retry_generation:
                # 仍不通过 → 标准拒答
                turn.final_answer = REFUSAL_TEXT
                async for event in self._emit_refusal(turn):
                    yield event
                return

    async def _expand_and_reretrieve(self, turn: _Turn) -> None:
        """首次重试前用扩展词再检索一次，只替换生成上下文。

        注：此处沿用既有行为——校验仍比对首轮 turn.reranked。已知偏差记入
        ADR-0007「已知问题」，本批次不擅自改动校验上下文（会移动评测基线）。
        """
        expanded_query = retriever.expand_query(turn.query)
        try:
            local2, pubmed2, _ = await retriever.retrieve(expanded_query, "both")
            if turn.options.use_reranker:
                reranked2 = await asyncio.to_thread(
                    reranker.rerank, turn.query, local2, pubmed2,
                    source_weights=turn.source_weights,
                )
            else:
                reranked2 = (local2 + pubmed2)[: settings.rerank_top_k]
            turn.context_text = reranker.build_context_text(reranked2)
        except Exception:
            pass  # 保持原上下文

    async def _emit_verified(
        self, turn: _Turn, answer: str
    ) -> AsyncIterator[Dict]:
        """校验通过的正文下发：已推过则整段替换，高风险则校验后回放。"""
        if not turn.emit_tokens:
            return
        if turn.pending_replace:
            # 屏幕上已有被判不合格的旧答案 → 整段替换
            yield sp.ev(sp.EVENT_REPLACE, text=answer, reason="judge_retry")
            turn.pending_replace = False
        elif not turn.live:
            # 高风险：校验通过后才回放（分片下发，不做人为延迟）
            for piece in sp.split_text(answer, settings.stream_replay_chunk_chars):
                _touch_first_token(turn.metrics)
                yield sp.ev(sp.EVENT_DELTA, text=piece)
                await asyncio.sleep(0)

    async def _emit_refusal(self, turn: _Turn) -> AsyncIterator[Dict]:
        """重试耗尽的拒答：屏上有旧答案用 replace，否则正常下发。"""
        if not turn.emit_tokens:
            return
        if turn.pending_replace:
            yield sp.ev(
                sp.EVENT_REPLACE, text=REFUSAL_TEXT, reason="judge_refused"
            )
            turn.pending_replace = False
            return
        for piece in sp.split_text(REFUSAL_TEXT, settings.stream_replay_chunk_chars):
            _touch_first_token(turn.metrics)
            yield sp.ev(sp.EVENT_DELTA, text=piece)
            await asyncio.sleep(0)

    # ==================== 节点 4：输出合规 ====================

    async def _check_compliance(self, turn: _Turn) -> AsyncIterator[Dict]:
        """拦截诊断 / 处方 / 剂量；命中则整段替换为标准拒答。"""
        yield sp.ev(sp.EVENT_STAGE, node=sp.NODE_COMPLIANCE, status="start")
        output_ok, violations = await asyncio.to_thread(
            safety_filter.check_output_compliance, turn.final_answer, turn.reranked
        )
        if output_ok:
            return

        for v in violations:
            logger.warning(f"[合规拦截] {v['type']}: {v['detail']}")
        turn.final_answer = safety_filter.get_refusal_response("out_of_scope")
        if turn.emit_tokens:
            yield sp.ev(
                sp.EVENT_REPLACE, text=turn.final_answer,
                reason="output_compliance",
            )

    # ==================== 收尾与出口 ====================

    def _finalize(self, turn: _Turn) -> Dict:
        """正常出口：统一结账并打完成日志，返回 final 事件。"""
        result = turn.finish(
            outcome.EXIT_OK,
            turn.final_answer,
            sources=turn.sources,
            judge_result=turn.judge_result,
        )
        logger.info(
            f"[工作流完成] latency={result['latency_ms']}ms, "
            f"retry={turn.metrics.retry_count}, "
            f"classification={turn.classification}, "
            f"judge={turn.metrics.judge_passed}, "
            f"stream={turn.metrics.stream_mode}"
        )
        return sp.ev(sp.EVENT_FINAL, **result)

    async def _exit(
        self, turn: _Turn, kind: str, answer: str, **extra
    ) -> AsyncIterator[Dict]:
        """提前出口：下发正文 → 记最终事件（由 stream() 统一 yield）。"""
        result = turn.finish(kind, answer, **extra)
        async for event in self._text_events(answer, turn.emit_tokens, turn.metrics):
            yield event
        turn.final_event = sp.ev(sp.EVENT_FINAL, **result)

    # ==================== 阻塞驱动 ====================

    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        options: Optional[RunOptions] = None,
    ) -> Dict:
        """阻塞问诊：消费 stream() 并只取 final。

        与流式共用全部节点，不维护第二份 RAG 链（ADR-0004）。
        返回固定 7 键（answer/sources/judge_result/latency_ms/session_id/
        outcome/message_id）。
        """
        final: Optional[Dict] = None
        async for event in self.stream(
            query, session_id, options, emit_tokens=False
        ):
            if event["event"] == sp.EVENT_FINAL:
                final = dict(event["data"])

        if final is None:
            # stream() 每个出口都以 final 收尾，这里只防御"未来改动漏发"。
            logger.error("流式问诊未产出 final 事件，按 error 出口兜底")
            final = outcome.degrade(
                session_id or "anonymous",
                outcome.EXIT_ERROR,
                FALLBACK_TEXT,
                reason="no_final_event",
            )
        return final

    # ==================== 内部工具 ====================

    @staticmethod
    async def _text_events(
        text: str, emit_tokens: bool, metrics: RequestMetrics
    ) -> AsyncIterator[Dict]:
        """把一段完整文本按协议分片下发（用于非生成类出口）。"""
        if not emit_tokens or not text:
            return
        for piece in sp.split_text(text, settings.stream_replay_chunk_chars):
            _touch_first_token(metrics)
            yield sp.ev(sp.EVENT_DELTA, text=piece)
            await asyncio.sleep(0)

    # ========== 同步封装 (FastAPI 调用) ==========

    def run_sync(self, query: str, session_id: Optional[str] = None) -> Dict:
        """同步运行工作流 (供脚本/评测调用)"""
        return asyncio.run(self.run(query, session_id))


# 全局单例
workflow = MedicalRAGWorkflow()
