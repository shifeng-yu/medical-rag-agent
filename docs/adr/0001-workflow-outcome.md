# ADR-0001: 问诊结果出口（outcome）设计约定

- 状态：Accepted
- 日期：2026-09-04
- 关联词条：`CONTEXT.md` → 问诊回合 / 问诊结果出口 / 出门原因 / 写会话规则 / success 口径

## 背景

`src/core/workflow.py` 的 `run()` 原有 5 个手拼同构 dict 的 return 点（急症拦截、敏感拦截、降级、正常、异常），并发现以下实际问题：

1. **拦截出口漏统计**：急症/敏感两个拦截出口不调 `request_logger.log_request`，这类请求从监控与 /stats 中凭空消失。
2. **形状漂移**：`concurrency.py` 的超时/过载响应手拼 dict 且**缺 `session_id`**，`/chat` 路由读取 `result["session_id"]` 时 KeyError 崩成 500——用户实际收不到降级话术；`/chat/batch` 用 `SourceInfo(**s)` / `JudgeResult(**dict)` 无脑展开，结果 dict 只要多一个键就崩。
3. **知识散落**：写会话（`add_message`）、上报指标、sources 白名单映射（workflow 与 routes 各筛一遍）、"layer" 枚举值散落在 workflow / judge / concurrency 三处。
4. **历史隐患**：`RequestLogger._lock` 用不可重入的 `threading.Lock`，而 `_flush()` 持锁期间调用 `get_stats()` 再次加锁——buffer 达到 50 条或 shutdown 时真实 flush 即死锁（一直潜伏，本次补记账后暴露并已修复为 `RLock`）。

## 决策

1. **一个出口模块，唯一收银台**：新增 `src/core/outcome.py`。它定义 7 个「出门原因」常量（`ok / emergency_blocked / content_blocked / degraded / error / overload / timeout`），提供唯一的结账函数 `finish()`（写会话、填指标、上报、拼装返回）与 API 层降级入口 `degrade()`。workflow 的每个出口与 API 层的超时/过载都必须经它收尾，禁止在出口处手拼返回 dict。
2. **返回形状固定 5 键**：`answer / sources / judge_result / latency_ms / session_id`。sources 白名单（title/source/score/publish_time/department）由 `outcome.to_sources` 单一所有；`judge_result.layer` 若调用方未给，则默认等于出门原因。API 层用显式逐字段映射构建响应，不再 `**splat`。
   > **修订（ADR-0004）**：返回形状由 5 键扩展为 **6 键**——`outcome.finish/degrade` 返回值新增 `outcome` 键携带出门原因 kind（消融评测据此判断每条样本的出口，路由层按显式字段消费故无害）。本决策其余内容不变。
3. **success 口径收紧**：指标 `success` 仅当出门原因为 `ok` 时真；拦截/降级/失败一律为假。同时 `RequestMetrics` 新增 `outcome` 字段记录出门原因，保证拦截类请求既能被统计、又不会稀释成功率。**影响**：/stats 的 `success_rate` 含义变为"完整问答链路成功率"，与历史口径（降级也计成功）不同。
4. **写会话规则**：只有面向用户的正式答复（ok/拦截/degraded）写入会话历史；系统失败出口（error/overload/timeout）一律不写——失败的请求不污染多轮上下文。已知副作用：失败回合在会话中留下孤儿的用户消息，下一轮 `build_context` 可能顺带带上，可接受。
5. **分两步走**：本次保持 dict 形状（单一构建函数收敛行为），类型化结果对象（pydantic/dataclass）留作后续独立升级，不做半吊子改造。

## 后果

- 正面：拦截类请求进入统计；超时/过载响应补全 `session_id`（修 500）；出口不可能漏写会话/漏上报；返回形状单一来源，批量/单条路由行为一致。
- 代价：`graceful.overload_response()/timeout_response()` 改为返回文案字符串（`overload_message()/timeout_message()`），响应拼装责任移交 outcome；`success_rate` 口径变化需在对外口径说明中同步。
- 未来评审如需推翻本 ADR（例如"系统失败也要写会话"或"恢复宽松 success"），先指出本 ADR 再讨论，不要静默改回。
