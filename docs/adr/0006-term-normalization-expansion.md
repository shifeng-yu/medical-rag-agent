# ADR-0006: 术语标准化接线（扩词式接入检索）

- 状态：Accepted
- 日期：2026-09-08
- 关联词条：`CONTEXT.md` → 问诊问题；实现见 `src/core/workflow.py`（3b 段）与 `src/core/retrieval.py`（`merge_dual_results`）

## 背景

`normalize_query()`（`src/core/knowledge_grader.py`：30 组疾病别名 + 10 组药品俗名）长期是**死代码**：`workflow.py` 调用赋值 `normalized_query` 后从未使用，`classify()` 与 `retrieve()` 用的都是原始 query。对外口径长期按"映射表与归一函数已就绪、检索主路径用原 query、归一为预留能力"如实降档，但这是 clone 仓库即可定位的"宣称 ↔ 实现"落差。

接线前需决策的关键问题：**归一化词如何进入检索**——直接替换主 query 会因别名映射的语义风险伤召回（例："头晕→眩晕""感冒→上呼吸道感染"是学术化改写，若知识库用口语词则替换后反而 miss；且无 GPU 评测环境无法量化验证方向）。

## 决策

1. **扩词式接入，不直接替换**：主检索保持原始 query（保留口语语义，对 paraphrase 鲁棒）；仅当 `normalize_query(query) != query`（确实发生别名→标准术语替换）时，用归一化词**补检索一次**，两路结果合并后进入既有重排。多数 query 无别名 → 零额外开销；别名 query 多一次检索召回更全。
2. **合并去重语义**：`retrieval.merge_dual_results` 纯函数，双源（local_kb / pubmed）独立合并；去重键 `(source, doc_id, content)`——完全相同的分块不进重复上下文；同文档不同分块保留，交给重排粗排截断（`top_k×3`）收敛，不预设输入上限。
3. **失败容忍**：补充检索失败只告警不致命，主结果照常走链路（不因增强路径降级主能力）。
4. **测试锁定**：`tests/test_retrieval_expand.py` 锁合并/去重/顺序/边界与 normalize 触发条件，行为可 clone 验证。
5. **不宣称量化增益**：本模块无单独评测数字支撑，README/对外口径只讲行为与设计理由（原 query 不丢信息 + 标准词兜底双路召回），不报"提升 X 个点"。

## 后果

- 正面：normalize_query 从死代码变为真实进链路；口语说法与标准术语语料两头可命中；主链路行为对非别名 query 零变化（默认分支无额外延迟）；对外口径可从"预留能力"升级为"标准化已进入链路（扩词式）"并可指认代码与测试。
- 代价：别名 query 多一次向量检索（延迟与召回略增，噪声由重排过滤）；真实增益数字待 GPU 评测环境 A/B（`docs/benchmark.md` 回填时可加该维度）；合并逻辑在 workflow 层实现，保持适配器接口不变。
- 未来评审如需推翻（例如改为直接替换、或接入分类前归一），先指出本 ADR 再讨论。
