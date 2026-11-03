# 架构决策记录（ADR）

按时间顺序排列。**先读 0001–0004 能理解主链路，0005–0006 是知识库，0007–0009 是这一轮改造。**

| 编号 | 决策 | 一句话 |
|---|---|---|
| [0001](0001-workflow-outcome.md) | 问诊结果出口收拢 | 所有出口走一个「收银台」，写会话/记指标/拼返回只此一处 |
| [0002](0002-retrieval-seam.md) | 检索适配器接缝 | `get_retriever()` 唯一选择入口，调用方不写 if/else |
| [0003](0003-llm-service.md) | LLM 总机 | 所有模型调用经 `get_llm()`，采样参数按任务收敛 |
| [0004](0004-ablation-on-main-pipeline.md) | 消融搭主链路 | 用 `RunOptions` 开关做对照，不维护第二份 RAG 链 |
| [0005](0005-upload-delete-api.md) | 上传/删除 API | 同名替换语义、异步任务、host 网络与数据卷约定 |
| [0006](0006-term-normalization-expansion.md) | 术语标准化扩词式接入 | 别名替换后补检一次并合并，主 query 不丢口语原意 |
| [0007](0007-streaming-protocol.md) | 流式协议与分级流式 | 9 个 SSE 事件；高风险先校验后回放，低风险边推边校 |
| [0008](0008-storage-layering.md) | 存储分层与任务外置 | MySQL 真源、Redis 读穿透、`kb_task` 跨进程可见 |
| [0009](0009-third-vector-backend.md) | 第三个检索后端 | PostgreSQL+pgvector 真异构，证明"换引擎只改配置" |

## 怎么写新的 ADR

`docs/agents/domain.md` 里有约定。摘要：**记录"为什么"和"放弃了什么"**，
不要复述代码；写清楚影响面与已知问题；被推翻的决策不要删，标状态指向新编号。
