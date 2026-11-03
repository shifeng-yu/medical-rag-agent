# ADR-0008：存储分层（MySQL 真源 + Redis 读穿透）与任务外置

- 状态：已采纳（2026-11-03）
- 影响面：`src/storage/*`（新增）、`src/session/manager.py`、`src/core/outcome.py`、
  `src/api/{sessions,feedback,documents}.py`、`src/tasks/*`（新增）、`docker-compose.yml`

## 背景

改造前，这个项目只有"能问答"，没有"能复盘"：

- 会话是进程内 `dict`（24h TTL），重启即丢，没有任何历史查询接口；
- 灌库任务状态是进程内 `_tasks`，多 worker 部署时 API 查不到 worker 的进度；
- 用户反馈无处可落，`judge_result` 里的分数只进 JSONL，无法和"哪条引用被吐槽"关联；
- 产品要求「会话历史与用户反馈落 MySQL 便于复盘调优」，代码里一条 SQL 都没有。

## 决策

### 1. MySQL 是真源，Redis 只做热缓存

写：先写 MySQL，成功后再追加 Redis（`RPUSH` + `LTRIM` + `EXPIRE`）。
读：先读 Redis，未命中回源 MySQL 并整段回填。

理由：会话是审计数据，丢一条就没法复盘；把 Redis 当真源再异步归档，
必然存在"最后一轮丢失"的窗口，而复盘恰恰最关心出问题的那几轮。

Redis 用 List 而不是 JSON blob：每条消息一次 `RPUSH` 即可，不需要
"读-改-写"，多进程写同一会话不会互相覆盖。

**冷缓存不追加**：`add_message` 会先 `EXISTS` 判断会话是否已在缓存中。
未命中时不写，否则会造出"只有最新几条"的半截历史，下次读缓存就丢了
早期对话——这比读不到更糟，因为它看起来是正常的。

### 2. 一套 SQLAlchemy 模型，两种后端

`DATABASE_URL` 决定后端：生产 `mysql+pymysql://...`，本地与 CI 默认
`sqlite:///volumes/app.db`。SQLite 不是"降级方案"而是"免运维的等价实现"：
CI 里跑的是真实 SQL（建表、唯一键、聚合查询），不是 mock。

若只用 MySQL、CI 不装数据库，测试就只能跳过存储层——那等于没测。
故选 ORM 吸收方言差异，而不是手写两套 SQL。

### 3. 七张表，按域拆开

| 表 | 作用 | 关键取舍 |
|---|---|---|
| `chat_session` | 会话头（标题/轮次/摘要） | 摘要与 `summary_upto` 同表，重算有依据 |
| `chat_message` | 消息（含出口/Judge/延迟） | 失败出口不写此表（沿用 ADR-0001 规则） |
| `chat_source` | 回答命中的来源 | 拆表而非 JSON：要能按 chunk 归因反馈 |
| `chat_feedback` | 用户反馈 | `UNIQUE(message_id)`：一条回答只留最新一次 |
| `chat_request` | 每回合指标 | 与 message 分开：拦截/超时也要进统计 |
| `kb_document` | 文档元数据快照 | 文件系统仍是源真，此表供后台展示分块数 |
| `kb_task` | 灌库/重建任务 | 落库后跨进程可见，替代进程内 `_tasks` |

`chat_request` 与 `chat_message` 不合并，是因为两者的写入门槛不同：
`error / overload / timeout` 按规则不写会话历史，但必须进成功率统计；
合并会让"成功率"只统计成功的那部分（分母缺失）。

### 4. 任务状态外置，执行器可切换

`TASK_MODE=inline|celery`：inline 用 `BackgroundTasks`（本地/测试），
celery 投递 Redis broker（生产 compose 的 `medical-rag-worker`）。
两条路径都调用 `src/tasks/runner.run_task`，不存在第二份实现。

worker 的嵌入默认跑 CPU（`DEVICE=cpu`）：单卡 12G 上 API 已占
Qwen(~8G)+BGE(~2G)，worker 再抢显存会 OOM。多卡机器可改回 cuda。

### 5. 返回形状 6 键 → 7 键

新增 `message_id`：前端提交反馈必须能定位到"哪一次回答"。没有它，
反馈只能绑到会话，无法回答"是检索错了还是生成错了"。

`_build_query_response` 用 `.get()` 取值，老客户端不受影响。

## 后果

- 会话 TTL 只作用于 Redis；MySQL 里的历史永久保留（复盘需要）。
  隐私清理走 `DELETE /api/v1/sessions/{id}`，级联删除消息/来源/反馈/指标。
- Redis 不可用时全链路降级为直读 MySQL，只告警一次；问诊不受影响。
- 单会话热缓存上限 100 条消息：超过后早期消息只在 MySQL 里，压缩摘要
  会基于缓存内的部分计算——超过 50 轮的会话摘要可能不完整（已知限制）。
- 存储失败不阻断问答：`outcome` 里所有落库操作都是 try/except + 告警，
  数据库抖动时用户仍能拿到已生成的答复（message_id 为空）。

## 已知问题

1. `chat_request` 与 JSONL 双写：JSONL 保留为原始日志，SQL 用于聚合；
   两处口径若不一致，以 SQL 为准（JSONL 不保证刷盘时机）。
2. 反馈只有 👍/👎 与原因标签，没有"引用级纠错"（用户标注某条来源不对）。
   该能力需要前端在溯源侧栏增加单条来源的操作，留到前端批次评估。
3. Celery worker 与 API 共用同一套数据卷，若同时重建同一文档，靠
   `doc_id` 幂等 + Milvus 删除后插入的顺序保证最终一致，但没有分布式锁。
   单 worker（compose 默认 `--concurrency=1`）下不构成问题。
