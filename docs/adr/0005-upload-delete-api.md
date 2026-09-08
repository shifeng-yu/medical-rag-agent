# ADR-0005: 知识库文档管理（网页上传 / 删除 / 替换语义）

- 状态：Accepted
- 日期：2026-09-04
- 关联词条：`CONTEXT.md` → 知识库文档；实现见 `src/api/documents.py`、`src/core/retriever.py` / `retriever_lite.py`

## 背景

系统要"实际部署使用"，但知识库只有离线脚本灌库（`scripts/ingest_kb.py` 全量扫目录、`ingest_pubmed.py` 拉文献），没有网页上传入口，也没有删除/更新能力。用户需要"网页传文档就能问答"，且知识库文档会迭代（写错要替换、过时要下架），只增不减三个月就脏。同时代码里有根深蒂固的命名混乱：同一个本地库概念并存 `local_kb`（检索 source 标签）/ `medical_kb`（source_type 写入值）/ `medical_knowledge_base`（物理集合名）/ `data/medical_kb`（目录）/ `kb`（本次新增 API 参数）五种叫法。

## 决策

1. **网页上传只开放本地库（`local_kb` 集合）**，pubmed 文献库仍只经 `ingest_pubmed.py` 离线管线灌入。理由：普通文本上传到 pubmed 集合会被检索端按 `source=="pubmed"` 误标成"[PubMed文献]"，污染文献库并误导来源标注（检索端根本不读 pmid/journal 字段，纯向量检索按 classification 路由）。
2. **命名收敛为 `local_kb` / `pubmed`**：本次新增的 API 参数不再叫 `kb`，文档集合对外统一 `collection=local_kb|pubmed`；`medical_kb` 等历史写法仅保留在既有 settings/元数据/目录名，不加新写法。见 CONTEXT.md 词条。
3. **上传 = 异步任务 + 落盘 + 重灌同一棵树**：`POST /documents/upload` 受理后返回 `task_id`，后台任务"落盘 `data/medical_kb/<department>/` → MedicalQAChunker 分块 → 写入 Milvus 集合"，与 `ingest_kb.py` 同一链路，不另起炉灶。
4. **同名文档 = 替换语义**：同一 `department` 下重传同名文件，先删旧 doc_id 的向量再落盘重灌（doc_id = 文件名去后缀，同集合内唯一），知识库不累积重复。这就避免了"upload 加时间戳前缀 → 脚本重灌 → 同内容新 doc_id 二次入库"的重复问题。
5. **删除 = 文件 + 向量一起清**：补 `retriever.delete_by_doc_id(col, doc_id)`（Docker 走 milvus `delete(expr=...)`，Lite 走自身 delete），新增 `DELETE /documents` 接口，删落盘文件并删对应向量。
6. **上传格式边界**：只收干净文本 `.md/.txt/.json`（json 若是 `{question,answer,...}` 问答结构则按条切，不做 PDF/docx/OCR——轻量私有化，不做重解析）。md/txt 单文件 ≤50MB、单批 ≤20 个。
7. **Docker 网络修复**：`medical-rag-api` / `baseline-rag` 与 etcd/minio/milvus 统一 `network_mode: host`，`MILVUS_HOST=localhost`——原 bridge 网络解析不到 host 网络里的 milvus 主机名。
8. **data 卷可写**：`./data:/app/data` 由 `:ro` 改 `rw`（upload 要写盘）；`./models` 保持 `:ro`，DEPLOYMENT 手册把"先放权重再 compose up + 启动校验目录非空"写成硬步骤。

## 后果

- 正面：知识库有完整生命周期（增/删/换），且与离线脚本共用目录树与链路；pubmed 文献库保持纯净；命名收敛为对外一套词。
- 代价：`retriever` 两个适配器各需补一个 `delete_by_doc_id`；上传任务状态是进程内存态（单 worker 内网部署够用，多 worker 需外置如 Redis）；已存在的旧命名（`medical_kb` 等）保留但不再扩散。
- 测试：documents API 测试（假编码器 + 临时 Lite 库）扩展到删除/替换路径；retriever delete 契约测试。docker-compose 改动验证需在有 Milvus 的环境做一次 compose up 冒烟。
