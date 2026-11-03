# ============================================================
# 流式问诊协议（SSE 事件词汇 + 流式风险分级 + 帧格式化）
#
# 为什么单独成文件：
#   - 事件名是前后端契约，必须单一所有：workflow 只产出事件 dict，
#     路由只把 dict 搬运成 SSE 帧，两边都不再手写字符串；
#   - 「哪些问题必须先校验再回放」是业务策略，与 HTTP 传输无关，
#     放这里可以脱离 FastAPI 单测；
#   - 帧格式化（含换行转义）集中一处，避免 data 里出现裸换行把
#     一个事件劈成两个。
#
# 决策与取舍见 docs/adr/0007-streaming-protocol.md；
# 领域词条见 CONTEXT.md「流式问诊」。
# ============================================================

import json
from typing import Dict, List

# ---------------- 事件词汇 ----------------
#
# 一次流式问诊的事件序列（正常出口）：
#   meta → stage(safety) → stage(classify) → stage(retrieve) → stage(rerank)
#        → risk → sources → stage(generate) → delta* → judge → final
# 重生成时在 judge(不通过) 之后追加 stage(generate, retry) 与 replace；
# 高风险链路在 judge 之前没有 delta（先校验后回放）。

EVENT_META = "meta"          # 会话建立：session_id（客户端据此落 localStorage）
EVENT_RISK = "risk"          # 流式风险分级：level=high|low / mode=token|buffered
EVENT_STAGE = "stage"        # 节点进度：node / status=start|done|retry / ms
EVENT_SOURCES = "sources"    # 召回来源（含原文片段），生成前先给前端
EVENT_DELTA = "delta"        # 增量文本（追加）
EVENT_REPLACE = "replace"    # 整段替换（校验不过重生成 / 合规拦截 / 最终拒答）
EVENT_JUDGE = "judge"        # 校验结果：layer / passed / scores / retry
EVENT_FINAL = "final"        # 收尾：与 /chat 同形状的固定 6 键
EVENT_ERROR = "error"        # 传输层异常（工作流内部异常走 final 的 error 出口）

ALL_EVENTS = frozenset({
    EVENT_META, EVENT_RISK, EVENT_STAGE, EVENT_SOURCES,
    EVENT_DELTA, EVENT_REPLACE, EVENT_JUDGE, EVENT_FINAL, EVENT_ERROR,
})

# ---------------- 节点名 ----------------

NODE_SAFETY = "safety"
NODE_CLASSIFY = "classify"
NODE_RETRIEVE = "retrieve"
NODE_RERANK = "rerank"
NODE_GENERATE = "generate"
NODE_JUDGE = "judge"
NODE_COMPLIANCE = "compliance"

# ---------------- 流式风险分级 ----------------

RISK_HIGH = "high"
RISK_LOW = "low"

# 必须先校验后回放的问题类型：答错会被直接执行（吃药、停药、儿童孕妇）
# 或直接下结论（确诊）。刻意不收录「怎么办 / 什么病 / 严重吗」这类
# 高频泛问——它们仍会触发 LLM-Judge（见 classifier.should_trigger_high_judge），
# 但不该为此牺牲首字延迟；泛问的答案由 Prompt 约束 + 规则层兜底。
HIGH_RISK_TERMS = (
    # 剂量与用法
    "剂量", "用量", "吃几片", "几片", "几粒", "多少毫克", "毫克", "ml", "毫升",
    "服用", "吃药", "用药", "服药", "停药", "换药", "加量", "减量", "配伍",
    # 处方与风险
    "处方", "禁忌", "副作用", "不良反应", "过量", "中毒",
    # 特殊人群
    "孕妇", "怀孕", "哺乳", "婴儿", "新生儿", "儿童", "小孩",
    # 结论性判断（答错即误导）
    "确诊", "诊断", "是不是", "什么病", "还能活", "会死",
)


def assess_risk(
    query: str,
    classification: str = "",
    need_high_judge: bool = False,
) -> str:
    """判定本次问诊的流式策略：high=先校验后回放，low=边生成边推。

    刻意只看「问题本身的风险」，不看 need_high_judge：后者是"要不要跑
    LLM-Judge"的开关（含"怎么办"这类泛问），拿它做缓冲条件会让绝大多数
    真实问诊退化成非流式，与首字延迟目标冲突。

    classification 目前不参与判定，保留形参是为了将来按双源命中做保守升级
    （改动点唯一，不必再改调用方）。异常/空输入按 high 保守处理。
    """
    if not query:
        return RISK_HIGH
    if need_high_judge and any(t in query for t in ("确诊", "诊断")):
        return RISK_HIGH
    lowered = query.lower()
    if any(term in lowered for term in HIGH_RISK_TERMS):
        return RISK_HIGH
    return RISK_LOW


# ---------------- 事件构造与帧格式化 ----------------


def ev(kind: str, **payload) -> Dict:
    """构造一个事件 dict（workflow 的产出单位，与传输层解耦）。"""
    if kind not in ALL_EVENTS:
        raise ValueError(f"未知的流式事件: {kind!r}（可用: {sorted(ALL_EVENTS)}）")
    return {"event": kind, "data": payload}


def sse_frame(event: Dict) -> str:
    """把事件 dict 序列化为一帧 SSE 文本。

    data 必须是单行：json.dumps 会把正文里的换行转义成 \\n，
    因此不会出现"裸换行劈帧"。ensure_ascii=False 保证中文原样传输，
    前端无需二次解码。
    """
    kind = event["event"]
    data = json.dumps(event.get("data", {}), ensure_ascii=False)
    return f"event: {kind}\ndata: {data}\n\n"


def sse_comment(text: str = "ping") -> str:
    """SSE 注释帧（心跳）。用于跨代理长连接保活，客户端会忽略。"""
    return f": {text}\n\n"


def split_text(text: str, size: int) -> List[str]:
    """按固定字符数切片，供"先校验后回放"分片下发。

    size <= 0 时整段返回，避免调用方传 0 导致空切片死循环。
    """
    if not text:
        return []
    if size <= 0 or len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)]
