# ADR-0002: 检索接缝归位（双源检索统一入口）

- 状态：Accepted
- 日期：2026-09-04
- 关联词条：`CONTEXT.md` → 双源检索；接口契约注释见 `src/core/retrieval.py`

## 背景

项目有两种双源检索后端，同一接口、不同实现：

- `src/core/retriever.py` → Docker Milvus（pymilvus，生产）
- `src/core/retriever_lite.py` → Milvus Lite（嵌入式，本地开发 / CI 无 Docker 环境）

发现的实际问题：

1. **选择逻辑复制 4 处**：`workflow.py:21-24`、`ablation.py:49-52`、`routes.py:65-69` 各自写 `if settings.use_milvus_lite:` 导入分支；`main.py` 启动预热更是**无视配置**，无条件 import Docker 版。
2. **适配器已漂移**：lite 独有 `insert()`、pymilvus 独有无人调用的 `degrade_search()`；打分指标一个 IP 一个 COSINE（score 不可比）；`both` 分支一个并行一个串行；向量字段名 `embedding` vs `vector`。
3. **灌库脚本绕开适配器**：`setup_milvus.py` / `ingest_kb.py` / `ingest_pubmed.py` 各自直连 pymilvus、自加载模型、自建 schema（字段硬编码 1024），与 lite 的 schema（用 `settings.embedding_dim`）字段名不一致——**lite 模式根本没有可用的灌库入口**。
4. 共享的 `expand_query` 实现被抄了两份。

## 决策

1. **唯一选择入口**：新增 `src/core/retrieval.py`，提供 `get_retriever()`（按 `use_milvus_lite` import 期选择、缓存单例）与共享 `expand_query()`。workflow / ablation / routes / main / 灌库脚本一律从该函数取适配器，禁止再写 if/else 分支。main 启动预热 bug 随之修复。
2. **接口补齐对齐**：pymilvus 适配器补 `insert(collection_name, chunks, batch_size=100)`；删除无人调用的 `degrade_search()`；`expand_query` 两边委托共享实现。两个适配器保留各自的存储细节（Docker 字段 `embedding` / lite 字段 `vector`），差异被接口挡住，外部只见统一 chunk 契约：`{"text", "metadata": {title, department, publish_time, source_type, doc_id}}`。
3. **打分口径统一为 IP**：向量入库前已 `normalize_embeddings=True`，IP 分数 ≈ 余弦相似度（越大越相关、降序返回），两适配器一致。生产 Docker collections 本就是 IP，零迁移；本地 lite 库按 COSINE 建的，重建一次。
4. **建表与灌库职责进适配器**：pymilvus 适配器 `connect()` 缺表则非破坏性自动创建（维度用 `settings.embedding_dim`，替换 setup 脚本硬编码的 1024）；`setup_milvus.py` 瘦身为显式管理命令（默认不动数据，`--force` 才删建）。两个灌库脚本删掉各自直连/自加载模型/自拼列的重复，改为 `get_retriever().insert()`——lite 模式从此可灌库。
5. **契约用测试锁定**：`tests/test_retrieval.py` 静态检查两适配器方法齐全、签名一致、`expand_query` 同源；并以临时库 + 假编码器做 insert→retrieve 真往返，实证 IP 打分方向。
6. **Windows flush 不尝试**：实证发现 milvus-lite 的 `flush()` 在 Windows 会因 rename bug 抛 `FileExistsError`，且**失败一次后数据变得查不到**（索引段损坏）；故 Windows 下直接跳过 flush（数据驻留内存可正常检索，进程重启后丢失——本地开发可接受），Linux/CI 正常 flush。

## 后果

- 正面：选择逻辑一处；main 启动不再无视配置；lite 模式可灌库、可在 CI 真跑检索；分数口径一致；新增 13 个契约测试锁住接口；顺带修复 Windows 下 lite 灌库后查不到数据的问题。
- 代价：本地旧 lite 库（COSINE）需重建一次（已移至系统临时目录备份 `%TEMP%/milvus_lite_backup_20260904_142343`，验证后用户自行决定清理）；`use_milvus_lite` 仍是 import 期决策，切换后端需重启（与现状一致，有意保留）。
- 未来评审如需推翻（例如引入第三个后端、或恢复不同打分口径），先指出本 ADR 再讨论。
