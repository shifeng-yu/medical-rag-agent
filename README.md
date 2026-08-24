# 全科医疗问诊 RAG 智能助手

基于 RAGFlow + Qwen-14B + BGE-M3 + Milvus 的私有化全链路医疗问诊 Agent。

[![GitHub](https://img.shields.io/badge/GitHub-仓库地址-181717?logo=github)](https://github.com/yyyyyyyysf/medical-rag-agent)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-orange.svg)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目概述

针对医疗数据合规要求与传统 RAG 检索不准的痛点，开发的全链路本地化医疗问诊智能助手。全程数据不出域，无公网 API 调用。

### 技术栈

`Python` `RAGFlow` `Qwen-14B` `BGE-M3` `LlamaIndex` `Milvus` `FastAPI` `Docker`

### 核心能力

| 模块 | 说明 |
|------|------|
| 私有化全链路部署 | RAGFlow Docker + Milvus 本地部署，GPTQ INT4 量化（28G→8G） |
| 双源检索增强 | 医疗知识库 + PubMed 离线文献，双 Collection 并行检索 |
| Q&A 语义分块 | 阈值定制(512→800)、医疗边界正则、超长 QA 二级拆分 |
| LLM-Judge 幻觉防控 | 规则硬拦截 + 模型三维打分，幻觉率 18%→3% |
| 动态上下文压缩 | >4轮自动摘要，token 压缩 >70%，准确率仅降 0.8% |
| 场景化来源权重 | 常见病优先本地库，前沿问题优先文献，自适应检索决策 |

### 性能指标

| 指标 | 数值 |
|------|------|
| 检索准确率 | 78.5%（Baseline 61.8%，提升 27%） |
| 幻觉率 | 3%（无校验 18%） |
| 平均响应 | 1.2s |
| 调用成功率 | 98% |

## 项目结构

```
├── config/                  # 全局配置
│   └── settings.py          # 模型路径、向量库连接、阈值参数
├── src/
│   ├── main.py              # FastAPI 应用入口
│   ├── api/                 # API 路由 + 请求/响应模型
│   ├── core/                # 核心业务逻辑
│   │   ├── workflow.py      # 主工作流编排（RAGFlow DAG 复现）
│   │   ├── classifier.py    # 两级问题分类
│   │   ├── retriever.py     # Milvus 双源并行检索
│   │   ├── reranker.py      # BGE-M3 两阶段重排序 + 来源权重
│   │   ├── generator.py     # 答案生成（CPU/GPU 自适应）
│   │   ├── model_loader.py  # GPTQ INT4 量化模型加载
│   │   └── judge.py         # LLM-Judge 双层幻觉校验
│   ├── chunking/            # 医疗 Q&A 自定义分块
│   ├── session/             # 会话缓存 + 动态上下文压缩
│   ├── monitoring/          # 请求日志 + 运行指标
│   └── utils/               # 工具函数
├── scripts/                 # 数据生成 + 评估脚本
│   ├── setup_milvus.py      # Milvus Collection 初始化
│   ├── ingest_kb.py         # 本地知识库导入
│   ├── ingest_pubmed.py     # PubMed 离线文献下载
│   ├── generate_local_kb.py # 生成本地 Q&A 示例数据
│   ├── generate_pubmed_library.py
│   ├── run_baseline.py      # Baseline RAG 对照（LlamaIndex）
│   ├── ablation.py          # 消融实验框架（模块可开关，归因验证）
│   └── generate_samples.py  # 生成知识库样本
├── ragflow_plugins/         # RAGFlow 可插拔分块插件
├── prompts/                 # Prompt 模板
├── data/                    # 关键词库 + 样本数据
│   ├── common_diseases_drugs.txt
│   ├── medqa_test.json      # 评测测试集（30条内置样例，可扩展至完整 MedQA 2134条）
│   └── samples/             # PubMed (9科×200条) + 本地KB (4科×200条)
├── tests/                   # 单元测试 + 幻觉率评估框架
├── docker-compose.yml       # 容器编排
└── requirements.txt
```

## 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU（推荐 RTX 3060 12G 或以上）
- Docker & Docker Compose
- Milvus 2.4+

### 安装

```bash
# 0. 克隆仓库
git clone https://github.com/yyyyyyyysf/medical-rag-agent.git
cd medical-rag-agent

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入模型路径

# 3. 启动 Milvus
docker compose up -d

# 4. 初始化向量库
python scripts/setup_milvus.py

# 5. 导入知识库
python scripts/ingest_kb.py
python scripts/ingest_pubmed.py --download --max-results 5000

# 6. 启动服务
python -m src.main
```

### API 使用

```bash
# 单次问诊
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "我最近经常头痛，是什么问题？"}'

# 多轮对话（带 session_id）
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "那需要注意什么？", "session_id": "上次返回的 session_id"}'

# 查看统计
curl http://localhost:8000/api/v1/stats
```

### 运行测试

```bash
pytest tests/ -v

# A/B 对比测试（Baseline vs 优化方案）
python scripts/run_baseline.py              # 启动 Baseline 服务(8001)
python -m src.main                            # 启动优化方案(8000)
python scripts/run_baseline.py --eval data/medqa_test.json --api http://localhost:8000
```

## 工作流

```
用户提问 → 问题分类(关键词+LLM) → 并行检索(本地KB+PubMed)
        → BGE-M3 重排序(粗排+精排+来源权重)
        → LLM 生成(Qwen/DeepSeek) → 规则校验+LLM-Judge打分
        → 通过则返回 / 不通过则重试(max 2次)
```

## 消融实验设计

检索准确率 27% 的提升（61.8% → 78.5%）是多模块叠加的结果，为验证各模块的独立贡献，提供消融实验框架：

| 模块 | 关闭方式 | 验证的问题 |
|------|---------|-----------|
| LLM-Judge 幻觉校验 | `--disable judge` | 校验层对幻觉率/准确率的影响 |
| BGE-M3 两阶段重排序 | `--disable reranker` | 重排序对检索精度的贡献 |
| 场景化来源权重 | `--disable source_weight` | 双库加权策略 vs 等权拼接 |
| 医疗定制分块 | `--disable custom_chunk` | 定制分块 vs 通用 512 分块（需配合 ingest 重建索引） |

```bash
python scripts/ablation.py                          # 完整配置
python scripts/ablation.py --disable judge          # 关掉幻觉校验
python scripts/ablation.py --disable judge reranker # 关掉多个模块
python scripts/ablation.py --samples 50             # 只跑前50条快速验证
```

> 注：custom_chunk 发生在数据摄入阶段（ingest），关闭它需用通用分块参数重新建库后再评测；本框架负责其余在线模块的开关与统一评测口径。

## 测试集说明

`data/medqa_test.json` 提供 **30 条内置评测样例**（本地知识库 18 条 + PubMed 文献 12 条），答案均可在 `data/samples/` 知识库中溯源，用于快速验证评测管线与消融框架（如 `python scripts/ablation.py --samples 5`）。

完整指标（检索准确率 78.5% vs 61.8%）基于公开 **MedQA 中文医学问答测试集（2134 条）** 评测，该数据集体积较大，未随仓库分发，可替换 `data/medqa_test.json` 为完整版后复现：

```bash
python scripts/run_baseline.py --eval data/medqa_test.json --api http://localhost:8000
```

## RAGFlow 插件

`ragflow_plugins/medical_qa_chunker.py` 可注册为 RAGFlow 自定义分块解析器，提供医疗场景专属分块策略。

## 医疗合规保障（7 维度）

本项目针对医疗 RAG 高风险场景实现了完整的合规保障体系：

| 维度 | 实现 |
|------|------|
| 知识库与数据层 | 术语标准化(别名映射)、语义分块(阈值800/100)、元数据完善(科室/来源/时间)、权威分级(Tier1-4) |
| RAG 核心引擎 | 向量检索+关键词召回、置信度阈值触发拒答、每项回答强制溯源标注 |
| 智能体能力 | 意图识别(科普/用药/就医)、多轮会话(独立缓存24h TTL)、上下文压缩(>4轮触发) |
| 合规与安全 | 输入敏感词过滤、输出合规拦截(诊断/处方/剂量)、急症强制引导120、免责声明 |
| 工程化 | Docker 一键部署、环境变量解耦、全局异常兜底、完整日志监控 |
| 业务可用性 | 场景边界标注、医学术语通俗化、测试用例体系(幻觉率/准确率评估) |
| 开源规范 | 标准 Python 项目结构、完整 README、代码语义化命名、无废弃文件 |

### 安全模块

- `src/core/safety.py` — 急症检测(10种)、敏感内容过滤、输出合规校验(诊断/处方/剂量拦截)
- `src/core/knowledge_grader.py` — 权威分级(Tier1-4)、医学术语标准化、科室映射

### 免责声明

本产品仅提供医学科普参考，不构成任何诊疗建议。身体不适请及时前往正规医疗机构就诊。
