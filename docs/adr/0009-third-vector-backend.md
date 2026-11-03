# ADR-0009：第三个检索后端（PostgreSQL + pgvector）

- 状态：已采纳（2026-11-03）
- 影响面：`src/core/retriever_pg.py`（新增）、`src/core/retrieval.py`、
  `config/settings.py`、`docker-compose.yml`、`requirements.txt`

## 背景

目标口径是「换库无需改动代码、配置即可生效」。改造前，仓库里有两个适配器
（Docker Milvus 与 Milvus Lite），但它们**是同一个引擎的两种部署形态**：
只能证明"换部署方式不改代码"，证明不了"换引擎不改代码"。

但只要有人追问「你换过别的向量库吗」，这句话就露底。

## 决策

### 1. 落一个真异构后端

新增 `PgVectorRetriever`，与另两个适配器同契约（connect / retrieve / insert /
delete_by_doc_id / expand_query / encode_query / embedder）。选择逻辑仍然只在
`retrieval.get_retriever()` 一处，调用方零改动。

`VECTOR_BACKEND` 取值 `milvus | milvus_lite | pgvector`；**留空则沿用旧的
`USE_MILVUS_LITE`**，保证既有部署升级代码后行为不变（重要：不能让加功能
变成破坏性变更）。

### 2. 单表 + `collection` 列，而不是一集合一张表

Milvus 侧是物理集合（`medical_knowledge_base` / `pubmed_literature`），pg 侧用
一张 `kb_chunks` + `collection` 列区分。

理由：pgvector 的 HNSW 索引对「索引列 + 过滤列」组合友好，单表少一次建表/迁移；
跨库统计（哪类问题命中哪个库）一条 SQL 就够，不用 UNION 两张表。
代价是删除/重建要带 `collection` 条件——已在适配器内部封装。

### 3. 打分口径对齐 Milvus 的 IP

Milvus 侧向量已归一化，用 IP 指标，score≈余弦相似度。pg 侧用
`1 - (embedding <=> query)`（`<=>` 是余弦距离），同向同量纲，重排层的
来源权重与阈值不需要为新后端调参。

浮点误差可能让 `1-d` 略微越界，适配器夹回 `[0,1]`。

### 4. 连接池 + 表名白名单

- 用 `psycopg2.pool.ThreadedConnectionPool`（最小 1 / 最大 `PG_POOL_MAX`）：
  每次查询新建连接在问诊这种高频短查询下开销明显，且连接泄漏会拖垮库。
- 表名来自配置、无法参数化，用 `^[A-Za-z_][A-Za-z0-9_]*$` 校验后再拼 SQL。
- DSN 打日志前做脱敏（`://user:***@host`），避免口令进日志。

### 5. 依赖与部署

- `psycopg2-binary` 进 `requirements.txt`；**但只在选中 pgvector 时才 import**，
  没装驱动的环境仍可用 Milvus（适配器内部惰性 import）。
- compose 里 postgres 放在 `profiles: ["pgvector"]`，默认不启动：

  ```bash
  docker compose --profile pgvector up -d
  VECTOR_BACKEND=pgvector docker compose up -d medical-rag-api
  ```

## 后果

- 三个后端共用一套契约测试：成员集合、`retrieve/insert` 签名必须一致
  （`tests/test_retriever_pg.py` 与 `tests/test_retrieval.py` 双重约束）。
- 真库测试靠 `PG_TEST_DSN` 开启；CI 没有数据库时跳过而不是假失败。
- **切换后端必须重新灌库**：向量存在各自引擎里，不通用。适配器不做双写，
  也不做跨库迁移——那是另一个量级的需求，硬做会把"换库只改配置"变成
  "换库要停机迁移"。
- pg 侧需要 `CREATE EXTENSION vector` 权限；应用账号不是超级用户时，
  要先由 DBA 建好扩展（报错信息里已提示）。

## 已知问题

1. 没有做真实 BGE-M3 的端到端检索验证（本机无模型权重）：pg 往返测试用
   确定性假编码器，验的是 SQL、打分方向、删除语义；向量质量本身由
   BGE-M3 决定，与后端无关。
2. HNSW 参数用默认值（`m=16, ef_construction=64`）。数据量上万后若召回
   下降，需要按 `SET hnsw.ef_search` 调查询侧参数，目前未暴露配置项。
