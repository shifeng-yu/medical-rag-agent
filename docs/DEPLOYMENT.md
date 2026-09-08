# DEPLOYMENT：Docker 化私有化部署手册

> 适用版本：medical-rag-agent（RAGFlow Agentic 图工作流蓝本自研 + Qwen-14B GPTQ INT4 + BGE-M3 + Milvus 单机）。
> 全链路本地推理，数据不出域、无公网 API 调用。编排决策见 `docs/adr/0005-upload-delete-api.md`（决策 7：host 网络；决策 8：data 卷可写、models 只读 + 权重预置硬步骤）。

## 1. 端口与网络模型

所有服务统一 **`network_mode: host`**——容器直接监听主机端口、经 `localhost` 互通（bridge 网络解析不到 host 网络里的 milvus 主机名，历史部署 bug，ADR-0005 决策 7 已修）。

| 端口 | 服务 | 用途 |
|---|---|---|
| 8000 | `medical-rag-api` | 问诊 API + 网页管理台（`http://<主机IP>:8000/`） |
| 8001 | `baseline-rag` | AB 对照实验服务（不部署可跳过） |
| 19530 / 9091 | `milvus-standalone` | 向量库 gRPC / 健康检查 |
| 9000 / 9001 | `minio` | Milvus 对象存储（内部） |
| 2379 | `etcd` | Milvus 元数据（内部） |

生产防火墙只需放行 **8000**（及内网管理面）；19530/9000/2379 建议仅内网可达。

## 2. 前置条件

| 项 | 要求 |
|---|---|
| Docker | 20.10+（compose v2，`docker compose version` 确认） |
| GPU | NVIDIA 驱动 + `nvidia-container-toolkit`（Qwen-14B INT4 推理约需 **8GB 显存**） |
| 内存 | ≥ 16GB（模型加载 + milvus/etcd/minio） |
| 磁盘 | ≥ 20GB（torch+auto-gptq 镜像构建约 10GB，另加数据卷） |
| 模型权重 | 两份，见步骤 3（一次性准备） |

## 3. 部署步骤

### 3.1 放置模型权重（**必须先做，启动前校验**）

`./models` 以只读挂载进容器，权重**不随镜像**构建（见 `.dockerignore`）。首次部署从 HuggingFace 下载（需公网一次；之后推理全本地）：

```bash
# models/Qwen-14B-Chat-GPTQ-Int4   ← Qwen/Qwen-14B-Chat-GPTQ-Int4
# models/bge-m3                    ← BAAI/bge-m3
huggingface-cli download Qwen/Qwen-14B-Chat-GPTQ-Int4 --local-dir models/Qwen-14B-Chat-GPTQ-Int4
huggingface-cli download BAAI/bge-m3            --local-dir models/bge-m3
```

**启动前校验（硬步骤，目录非空才允许继续）**：

```bash
test -d models/Qwen-14B-Chat-GPTQ-Int4 && test -n "$(ls -A models/Qwen-14B-Chat-GPTQ-Int4)" \
  && test -d models/bge-m3 && test -n "$(ls -A models/bge-m3)" \
  && echo "OK: 模型权重就位" || { echo "FAIL: models 目录为空或缺失，禁止启动"; exit 1; }
```

### 3.2 配置（可选）

```bash
cp .env.example .env   # 如需覆盖默认值
```

容器内关键变量已由 compose 写死，**无需**手动改：

- `USE_MILVUS_LITE=0`：容器走 pymilvus → Docker Milvus。若误设 true，服务会改用嵌入式 Milvus Lite 并把数据写在容器可写层，**容器重建即丢**。
- `MILVUS_HOST=localhost`：host 网络下直连本机 milvus-standalone。

### 3.3 构建与启动

```bash
docker compose build                 # 首次较慢（torch/auto-gptq），约 10 分钟+
docker compose up -d etcd minio milvus medical-rag-api   # 只起生产必需 4 个
# 完整含 AB 对照服务： docker compose up -d
```

### 3.4 健康与就绪校验

```bash
docker compose ps                     # 目标：4 个容器全部 (healthy)

# API liveness：HTTP 200 只代表进程活着
curl -s http://localhost:8000/api/v1/health
# 就绪判据：milvus_connected 必须为 true（milvus 未就绪时启动会自动降级但检索不可用）
# {"status":"ok","models_loaded":true,"milvus_connected":true,...}
```

若 `milvus_connected=false`：`docker compose ps` 看 milvus/etcd/minio 是否 healthy，等三者就绪后 `docker compose restart medical-rag-api`。

### 3.5 冒烟：上传 → 灌库 → 问答

```bash
# 1) 上传一篇知识库文档（.md/.txt/.json，department=科室标签）
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@头痛问答.md" -F "department=神经内科" -F "collection=local_kb"
#    → {"task_id":"...","status":"processing",...}

# 2) 轮询灌库任务直到 done
curl -s http://localhost:8000/api/v1/documents/tasks/<task_id>   # status=done, chunks>0

# 3) 问一句验证能命中刚上传的内容
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "苹果有什么好处？"}'
#    → answer 引用刚上传文档，sources[0].department == 神经内科

# 网页管理台（上传/文档列表/删除 + 聊天一体）：http://<主机IP>:8000/
```

删除/替换/格式边界等接口行为见 `src/api/documents.py` 头注释与 ADR-0005。

## 4. 数据与备份

| 主机目录 | 内容 | 说明 |
|---|---|---|
| `./data/` | 知识库文本：`medical_kb/<department>/`（网页上传落盘）、`pubmed/`（离线脚本灌入） | 只增语义，**备份重点** |
| `./volumes/` | milvus/etcd/minio 元数据与向量 | 向量库重建成本高，**备份重点** |
| `./models/` | 模型权重（只读） | 可重新下载，不必入备份 |
| `./logs/` | 应用日志 | 按需 |

```bash
# 建议 cron 每日备份（示例：保留 7 天）
tar czf backup_$(date +%F).tar.gz data volumes
```

> 灌库链路是幂等的（同名替换语义），恢复数据后执行 `scripts/ingest_kb.py` 可重灌 `data/` 下全部文档，作为向量库损坏时的兜底。

## 5. 运维与回滚

```bash
docker compose logs -f medical-rag-api     # 看日志
docker compose restart medical-rag-api      # 改 .env/挂载后重启
docker compose pull                         # 更新镜像（自定义 build 则 docker compose build）
docker compose down                         # 停服务（数据卷保留在 ./data ./volumes）
docker compose down -v                     # ⚠️ 清空 volumes/ 下的向量数据，慎用
```

代码更新后重部署：`git pull && docker compose build && docker compose up -d`。

## 6. 常见问题

| 症状 | 原因与处置 |
|---|---|
| api 容器反复重启，日志报模型加载失败 | `models/` 权重未放置/不完整 → 回到 3.1 校验步骤 |
| `health` 返回 200 但 `milvus_connected=false` | milvus 依赖未就绪 → `compose ps` 等待 healthy 后 restart api |
| 8000/19530 端口被占 | host 网络直占主机端口，先 `ss -lntp` 排查再启动 |
| 上传成功但检索为空 | 多发生在历史 Lite 库场景；Docker 模式确认 `USE_MILVUS_LITE=0` 生效（`docker compose exec medical-rag-api env \| grep MILVUS`） |
| 本机 Windows Docker 下 host 网络不可用 | 生产目标为 Linux 服务器；纯本机体验请走 README 的本地开发模式（Milvus Lite） |
| `docker compose build` 报 data 相关 COPY 失败 | 检查 `.dockerignore` 白名单 `!data/common_diseases_drugs.txt` 是否被误删 |

## 7. 与 RAGFlow 的关系

本项目**不自带** RAGFlow 引擎容器：编排引擎是按 RAGFlow 的 Agentic 图工作流（图编排 / 分支重试 / 降级出口）**自研落地**，Milvus/Qwen/BGE 为底层组件。若需引入 RAGFlow 原生的 DeepDoc 文档解析（OCR/版面/表格 → Q&A 分块模板），属后续增强项，不在本手册范围内。
