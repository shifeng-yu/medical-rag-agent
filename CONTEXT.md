# CONTEXT.md — 领域词汇表

medical-rag-agent 的领域词条。命名新模块、写测试名、描述重构时，使用本表术语；架构词汇（模块/接口/接缝/适配器/深度）另见 codebase-design 技能词汇表，决策记录见 `docs/adr/`。

## 问诊链路

- **问诊回合（consultation turn）**：一次完整的 user→assistant 请求-响应。从 `workflow.run(query, session_id, options)` 进入，无论走哪条路结束，都算一个回合。`options`（`RunOptions`）是环节开关，默认全开 == 在线行为，消融评测用它关环节（见「消融评测」）。
- **问诊问题（query）**：用户的输入。可能被急症/敏感拦截、标准化（`normalize_query`）、分类、作为检索与生成的输入。术语标准化为**扩词式接入**：当 `normalize_query` 结果与原文不同（发生别名→标准术语替换）时，检索用标准术语补检一次并去重合并（`retrieval.merge_dual_results`，键 source+doc_id+content），主检索仍用原始 query。决策见 ADR-0006。
- **双源检索（dual-source retrieval）**：一次问诊同时检索两个来源——本地医疗知识库（local_kb）与 PubMed 文献（pubmed）。两种后端（Docker Milvus 的 `retriever.py` 与嵌入式 Milvus Lite 的 `retriever_lite.py`）是同一接口下的两个适配器：connect / retrieve / insert / expand_query / encode_query / embedder。**选后端的唯一入口是 `src/core/retrieval.py` 的 `get_retriever()`**（按 `use_milvus_lite` import 期选择并缓存），任何调用方不得再写 if/else 分支；灌库脚本同样经由适配器的 `insert()`。分数统一 **IP 指标**（向量已归一化 → score≈余弦相似度，越大越相关、降序返回）。决策见 ADR-0002。
- **重排（rerank）**：粗排（向量相似度截断）+ 精排（BGE-M3 Cross-Encoder）+ 场景化来源权重，产出带 `rerank_score` 的有序候选。
- **幻觉校验（LLM-Judge）**：规则层（`rule_check`）+ 模型层（`llm_judge`）。`judge_result.layer` 取 `rule / rule_only / judge / both`，是裁判模块**自己的内部报告**，与问诊回合的「出门原因」是两套词汇，不混用。消融关闭校验时占位为 `ablation_off`（见「消融评测」）。

## 模型对话（LLM 总机）

- **LLM 总机（llm service）**：仓库里所有"与大模型对话"的唯一入口，实现在 `src/core/llm.py`。业务模块只声明意图：`get_llm().complete("answer", prompt)`，**不得**直接加载模型或各自定采样参数。
- **任务参数表（TASKS）**：5 个业务任务（answer / degraded / classify / judge / summary）的默认采样参数（max_tokens / temperature / do_sample）在此单一来源。当前全部显式贪婪解码（do_sample=False），temperature 不生效——历史参数 0.3/0.5 因从未传 do_sample=True 而一直空转，收口后如实归零（行为不变）。决策见 ADR-0003。
- **假话务员（FakeLLM）**：确定性应答、不加载模型的后端，由 `settings.llm_backend="fake"`（env `LLM_BACKEND=fake`）启用。分类→本地库、裁判→三维高分 JSON、其余→带来源标注的合规答复；测试可用 `responses` 定制。启用后整条链路不依赖 Qwen 即可跑通（CI/演示/无 GPU）。
- **双后端契约**：`QwenBackend`（真实，职责收编自已删除的 model_loader.py）与 `FakeLLM` 遵循同一 `generate(task, prompt, *, max_tokens, temperature, do_sample)` 约定，不强制继承，接口一致性由 `tests/test_llm.py` 契约测试锁死。切换后端只改配置，不动调用方。

## 收尾（问诊结果出口）

- **问诊结果出口（outcome）**：所有问诊回合的收尾处，唯一实现是 `src/core/outcome.py` 的收银台函数。它统一负责：写会话（按规则）、填指标（outcome/success）、上报 `log_request`、拼装返回形状。任何人不得绕过它在出口处手拼返回 dict。
- **出门原因（kind）**：一个问诊回合的唯一收尾方式，共 7 个值：`ok / emergency_blocked / content_blocked / degraded / error / overload / timeout`。前四个表示系统产出了面向用户的正式答复；后三个是系统失败。
- **写会话规则**：只有"面向用户的正式答复"（`ok / emergency_blocked / content_blocked / degraded`）才写入会话历史；系统失败出口（`error / overload / timeout`）一律不写。已知副作用：失败回合会在会话里留下孤儿的用户消息。
- **success 口径**：指标里 `success` 仅当出门原因是 `ok` 时为真。拦截/降级/失败都不算链路成功（要区分它们看 `outcome` 字段）。

## 消融评测

- **评测模式（eval mode）**：`RunOptions(use_session=False)` 下 `workflow.run` 不建会话、不写历史、对话上下文为空——供消融/批量评测使用，避免污染会话存储。
- **消融实验（ablation）**：`scripts/ablation.py` 通过 `AblationConfig → RunOptions` 关闭 judge / reranker / source_weight 环节后，**直接跑主工作流**评测，不再复制第二份 RAG 链。评测跑的与线上是同一条链。custom_chunk 维度在摄入侧（ingest），脚本只记录不开关。决策见 ADR-0004。

## 知识库文档

- **文档集合（collection）**：知识库按来源分成两个互斥的向量集合——本地医疗知识库与 PubMed 文献库。对外命名（API、检索结果的 source 标签、UI）一律 `local_kb` / `pubmed`。历史遗留叫法仅限代码内部且已收敛：物理集合名 `medical_knowledge_base`/`pubmed_literature`（settings）、元数据 `source_type` 写入值 `medical_kb`/`pubmed`（标记"这份文档从哪个库灌入"，与检索 source 标签不同义）、检索路由分类 `classification` 取 `local/pubmed/both`（见「问诊链路」双源检索）。**新增代码不得再引入新叫法**。
- **知识库文档（document）**：知识库的最小摄入单元，一个文件（`.md`/`.txt`/`.json`）。文档与分块是一对多：`ingest` 把一个文档切成多块写入集合。doc_id 取文件名去后缀（同集合内唯一）；同名文档再次上传 = 替换语义（先删旧 doc_id 向量再落盘重灌），不会累积重复。决策见 ADR-0005。
- **上传任务（ingest task）**：网页上传的异步处理单元。`upload` 接口受理后返回 task_id，后台任务执行"落盘 → 分块 → 向量化 → 写入集合"，状态机 `processing → done | error`；删除文档同理先删文件再删向量，任务模型记录文件数/分块数/失败数。任务状态为进程内存态（单实例内网部署可接受，多 worker 需外置）。
- **文档目录（kb dir）**：落盘目录 `data/medical_kb/<department>/`（kb）与 `data/pubmed/`（pubmed）即知识库的"源真"：`ingest_kb.py` 全量扫目录重灌、upload 写同一棵树。`department`（科室/主题标签）同时决定元数据与子目录名。网页上传只开放 `local_kb` 集合；pubmed 文献库只经 `ingest_pubmed.py` 离线管线灌入，避免普通文本被误标为文献。决策见 ADR-0005。
- **摄入侧配置项**：分块参数（max 800 / min 100 / overlap 50 token）与双源/重排/校验等在线环节正交，改分块不影响已入库向量（需重建索引生效），与 `ingest` 脚本及 `scripts/generate_samples.py` 同源。

## 质量与约束

- **返回形状**：`workflow.run` / API 层降级返回固定 6 键：`answer / sources / judge_result / latency_ms / session_id / outcome`。其中 `outcome` 键携带「出门原因」，供评测等下游判断本次回合的出口。sources 白名单字段（title/source/score/publish_time/department）只在 `outcome.to_sources` 一处定义。
- **合规**：本产品只做医学科普参考，不构成诊疗建议；急症与敏感内容在入口拦截，诊断/处方/剂量在输出拦截。
