# ADR-0007：流式问诊协议与分级流式

- 状态：已采纳（2026-11-03）
- 影响面：`src/core/workflow.py`、`src/core/stream_protocol.py`、`src/core/llm.py`、
  `src/core/generator.py`、`src/core/outcome.py`、`src/api/routes.py`

## 背景

产品要求「SSE 流式逐字输出」与「首字延迟 < 1.2s」，同时保留「LLM-Judge
交叉校验不过则重生成 ≤2 次，仍不过标准拒答」。这三条在时序上互相拉扯：

- 逐字输出要求 token 边生成边下发；
- LLM-Judge 只能对**完整答案**打分，发生在生成结束之后；
- 医疗场景又不允许把未校验的结论先给用户看。

原实现是一次性阻塞返回，且整个链路写在一个 347 行的 `run()` 里，无法在
不复制第二份 RAG 链的前提下插入事件流。

## 决策

### 1. `stream()` 是原语，`run()` 是它的消费者

`workflow.stream()` 产出事件 dict（事件词表见 `stream_protocol.py`），
`run()` 通过 `async for` 消费并只取 `final`。这样阻塞接口、批量接口、
benchmark、ablation 与在线流式共用同一串节点，满足 ADR-0004「只维护一条链」。

`emit_tokens=False` 只关闭 token 下发，不改变节点与判定；评测链路因此
不会因为引入流式而改变行为基线。

### 2. 分级流式（tiered streaming）

按**问题本身的风险**分两档：

| 档位 | 触发 | 行为 |
|---|---|---|
| `high` | 剂量/用法/处方/停药、儿童孕妇、确诊结论类词 | 先攒完 → 校验 → 通过后分片回放 |
| `low` | 其余（含「怎么办」「什么病」这类泛问） | 边生成边推 → 生成完校验 → 不过发 `replace` |

刻意**不用** `classifier.should_trigger_high_judge` 作为分档依据：它包含
「怎么办」「严重」等高频词，会把绝大多数真实问诊推入缓冲档，首字延迟
直接等于整段生成时间（14B 生成 500 字通常 8–15s），流式失去意义。
泛问的答案风险由 Prompt 约束与规则层兜底。

### 3. 重生成用 `replace` 而非追加

低风险链路在第一次生成时已经把可能不合格的答案推给前端。重生成时：

- 开启新一轮 token 流会**追加**到旧答案后面，屏幕出现两段拼接；
- 因此 `pending_replace=True` 时下一轮改为「攒完 → 校验 → 单帧 `replace`」。

合规拦截（诊断/处方/剂量）与最终拒答同样走 `replace`，前端只需实现
「append 或 set」两种动作。

### 4. sources 白名单扩三个溯源字段

`doc_id` / `chunk_id` / `snippet`（≤1000 字）。`chunk_id` 取正文 sha1 前 12 位，
**不改 Milvus schema、不重建索引**，同一块内容在 Docker Milvus / Lite /
将来 pgvector 下算出的 ID 一致，可直接作为反馈归因锚点。

### 5. 心跳用 `asyncio.wait` 而不是 `wait_for`

`wait_for` 超时会取消被等待的协程——用它做心跳等于每个心跳周期把工作流
掐断一次。改为持有 `__anext__` 的 Task，用 `asyncio.wait(timeout=…)` 探测：
未完成就发注释帧继续等，完成才取结果；超时才 `cancel()`。

## 后果

- 首字延迟出现两种口径：`stream_mode=token` 记模型首字，
  `stream_mode=buffered` 记校验通过后的首字。指标里带 `stream_mode` 区分，
  对外不再笼统宣称单一数字。
- 前端必须实现 `delta`（追加）与 `replace`（覆盖）两种动作，不能只做 append。
- `metrics` 新增 `stream_mode / first_token_ms / risk_level`，JSONL 结构向后兼容
  （新键追加，不删旧键）。

## 已知问题（本批次不修，避免移动评测基线）

1. **重试后的校验上下文未对齐**：`retry==1` 会用扩展 query 重新检索并生成，
   但 `judge.validate` 仍比对首轮 `reranked`。这会让新上下文里的实体被判
   「未溯源」而再次失败，属既有行为，改造时原样保留；修复需单列一次
   基线重跑。
2. **低风险链路没有逐句规则拦截**：规则层仍在整段生成后执行，理论上存在
   「违规片段已下发数百毫秒」的窗口。高风险档不存在该窗口。若要收紧，
   下一步是在低风险链路做句级 `rule_check` 与即时截断。
