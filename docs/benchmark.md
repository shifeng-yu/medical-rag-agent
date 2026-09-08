# benchmark 评测闭环说明

> 仓库内可复现评测：`scripts/benchmark.py` 基于内置 30 条子集（`data/medqa_test.json`），复用主工作流 `workflow.run` 输出**命中率**与**高风险输出率**，让 README 宣称的 27%（61.8→78.5）与 18%→3% 有一个可 clone、可复现、口径一致的最小版本。

## 运行方法

```bash
# 0. 先灌库（任选后端；本地无 Docker 用默认 Milvus Lite）
python scripts/ingest_kb.py
python scripts/ingest_pubmed.py --download --max-results 5000

# 1. 跑真实评测（GPU + .env 指向 INT4 模型；默认后端 qwen）
python scripts/benchmark.py --report docs/benchmark.md

# 快速验证（前 5 条）
python scripts/benchmark.py --samples 5

# 免模型自检（CI / 无 GPU 环境，仅验证结构与判定逻辑）
python scripts/benchmark.py --check
```

> `--llm-backend fake` 可跑通管线冒烟（确定性假话务员），但数字**不代表效果口径**，请勿写入对外材料。

## 评测口径

- **命中率**：`outcome == ok` 出口，且标准答案关键内容出现在生成回答中 = 命中（端到端命中，评测模式不建会话）。非 ok 出口（急症/敏感拦截/降级/error）自动未命中并如实记录出口类型。
- **高风险输出率**：`ok` 出口中最终回答被硬规则判为高风险的占比（**最终输出口径**）：
  1. 合规违规 —— 输出侧硬规则命中越权诊断 / 处方推荐 / 剂量指导；
  2. 编造实体 —— 回答中的疾病/药品实体在「标准答案 + 检索参考上下文」中均无法溯源。
  - 拒答话术不算高风险；缺来源标注仅提示、不计入。
- **对照构造**：对照组与实验组复用同一条主工作流，用 `RunOptions` 关闭 `judge / reranker / source_weight` 构造基线，非两套流水线（杜绝测试代码差异）。同机同配置跑，环境一致。

## 结果表（在 GPU + 已灌库环境运行后由脚本回填）

| 配置 | 样本 | 命中 | 命中率 | ok 出口 | 高风险 | 高风险输出率 | 平均延迟(ms) |
|---|---|---|---|---|---|---|---|
| baseline（关 judge/rerank/来源权重） | 30 | - | - | - | - | - | - |
| full（完整链路） | 30 | - | - | - | - | - | - |

> 运行 `python scripts/benchmark.py --report docs/benchmark.md` 后，本文件会被脚本覆盖为带真实数字的报告（含出口分布与高风险样本明细）。

## 与对外口径的关系

- 对外宣称 **27%（61.8→78.5）** 与 **18%→3%** 是预研阶段**完整评测集**（数百条量级）的线下结果，属历史对外口径；
- 仓库内 30 条子集跑出的数字**不必等于** 27%/18%→3%（子集更小、难度构成不同）——它存在的意义是：clone 仓库后能按上述命令复现一组口径一致、可解释的数字，而不是只有 README 上一句没有资产支撑的话；
- 对外表述统一为：**"完整集预研评测得到 27%/18%→3%，仓库内附 30 条可复现子集与 benchmark 脚本供核验，完整集需线下复跑"**。

## 与权威分级（Tier）的关系

- 完整链路默认 `use_authority_weight=False`（reranker 只按 cross-encoder × 来源权重排序），保证上述数字与消融基线可比；
- Tier1-4 权威打标**始终附加**到每条来源并随 `/chat` 的 `sources` 返回（`authority_tier` / `authority_label` 字段），权威分级作为可选排序开关（`reranker.rerank(use_authority_weight=True)`）待 A/B 验证，避免未经评测就改默认链路。
