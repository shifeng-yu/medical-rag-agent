# ADR-0004: 消融评测搭在主链路上

- 状态：Accepted
- 日期：2026-09-04
- 关联词条：`CONTEXT.md` → 消融评测；环节开关定义见 `src/core/workflow.py` 的 `RunOptions`

## 背景

消融实验（验证 judge / reranker / source_weight 各模块的独立贡献）需要一条能开关环节的 RAG 链。原实现 `scripts/ablation.py` 的 `run_single` **把整条链复制了一遍**（classify→检索→重排→拼上下文→生成→裁判，约 60 行），且抄得比主链路"少"：

- 没有急症/敏感入口拦截、医学术语标准化、会话上下文、检索重试、校验失败重生成/扩词重检、输出合规拦截；
- 裁判不通过被简化为打一个 `passed=False` 标记，不走主链路的"重试→仍失败→拒答"逻辑。

后果是**评测跑的和线上跑的不是同一条链**——将来改主链路逻辑（如重试规则、合规拦截词表），评测不会跟着变，归因结果悄悄失真。这类"第二份实现"是评测最阴的 bug 来源。

## 决策

1. **主链路吃环节开关**：`workflow.run(query, session_id=None, options=None)` 增加可选 `RunOptions`（dataclass，默认全开 == 无 options 时完全一致 == 在线行为零变化），支持关闭：
   - `use_judge`：跳过 `judge.validate`（规则层 + LLM-Judge），判定恒放行，`judge_result` 占位 `{"layer": "ablation_off"}`——不产生重试循环；
   - `use_reranker`：跳过重排，按原始检索顺序 `(local + pubmed)[: rerank_top_k]` 拼接；
   - `use_source_weight`：仅在重排开启时有意义，关闭后两库等权 (0.5, 0.5)；
   - `use_session`：关闭 = 评测模式——不建会话、不写历史、对话上下文为空，评测不污染会话存储。
2. **返回形状暴露出口**：`outcome.finish/degrade` 返回 dict 增加 `outcome` 键（携带出门原因 kind）。这样消融评测无需靠"猜固定文案"判断某条样本是不是被拦截/降级/报错。路由层按显式字段消费，新键天然无害（这是 B 项目去掉 `**splat` 后才变得安全的）。
3. **删第二份实现**：`ablation.py` 删除 `run_single`，`AblationConfig.to_options()` 映射到 `RunOptions`，`evaluate` 直接 `asyncio.run(workflow.run(query, session_id=None, options=cfg.to_options()))`。
4. **评测口径**：命中规则与 run_baseline 一致（标准答案出现在生成结果中）；**仅 ok 出口参与命中**，非 ok 出口（拦截/降级/error）自动未命中，并在 per-sample 的 `outcome` 字段与汇总的 `outcome_counts` 中如实记录，便于事后分析哪类样本被链路拦下。
5. **custom_chunk 仍是摄入侧维度**：关闭它 = 用通用分块重建索引再评测，发生在 ingest 阶段，不在在线链路；脚本只负责记录该维度开关状态（沿用原语义）。

## 后果

- 正面：RAG 链只维护一份，评测测的就是线上那条链；环节开关以参数形式进入主链路，将来任何"需要对比开关"的实验（不只是消融）都能复用；评测模式（use_session=False）顺带解决了批量评测污染会话存储的问题。
- 代价：`workflow.run` 的签名多了一个可选参数（默认 None，无感）；主链路多了几处 `if options.use_xxx` 分支（默认全开路径下是恒真判断，开销可忽略）。judge 关闭时 `judge_result.layer="ablation_off"` 是第三套 layer 词汇的哨兵值——它与裁判内部报告（rule/rule_only/judge/both）、出门原因（kind）都不混用，仅在消融场景出现，测试已锁定。
- 未来评审如需推翻（例如评测希望恢复"连合规拦截都不跑的对照"），先指出本 ADR 再讨论。
