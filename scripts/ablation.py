# ============================================================
# 消融实验框架 (Ablation Study Framework)
# 用途：验证 RAG 管线中每个优化模块对最终效果的独立贡献
#
# 【为什么需要它】
#   项目的检索准确率提升 27%（61.8% → 78.5%）是多模块叠加的结果：
#   定制分块 + 双源检索 + 两阶段重排 + 场景化来源权重 + LLM-Judge。
#   评审或自查被问到"哪个模块贡献最大"时，需要有归因证据。
#   本脚本即为此设计：每个模块均可独立开关，跑同一测试集对比。
#
# 【设计维度】
#   A. judge          —— 幻觉校验（规则层 + 三维打分）
#   B. reranker       —— BGE-M3 两阶段重排序
#   C. source_weight  —— 场景化来源权重（常见病优先本地/前沿优先文献）
#   D. custom_chunk   —— 医疗定制分块（在数据摄入侧控制，见下方说明）
#
# 【运行方式】（需 GPU + 模型 + MedQA 测试集环境）
#   python scripts/ablation.py                          # 完整配置
#   python scripts/ablation.py --disable judge          # 关掉幻觉校验
#   python scripts/ablation.py --disable judge reranker # 关掉多个模块
#   python scripts/ablation.py --samples 50             # 只跑前50条（快速验证）
#   python scripts/ablation.py --test-file data/medqa_test.json --out ablation_result.json
#
# 【注意事项】
#   1. custom_chunk 发生在数据摄入阶段（ingest），不在在线推理链路。
#      关闭它 = 用通用分块（chunk_max_tokens=512）重新建库索引后再评测，
#      本脚本只负责记录该维度，实际由 scripts/ingest_*.py 配合重建。
#   2. 输出为 JSON + 控制台表格，建议多组配置跑完后人工汇总成对比表。
# ============================================================

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 确保项目根目录在 path 中（与 run_baseline.py 一致）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from config.settings import settings
from src.core.classifier import classifier
from src.core.reranker import reranker
from src.core.generator import generator
from src.core.judge import judge

# CPU/GPU 自适应检索器（与 workflow.py 同一套选择逻辑）
if settings.use_milvus_lite:
    from src.core.retriever_lite import lite_retriever as retriever
else:
    from src.core.retriever import retriever


# ============================================================
# 消融配置
# ============================================================

class AblationConfig:
    """单组消融配置：默认全部开启，按需关闭"""

    def __init__(
        self,
        name: str,
        use_judge: bool = True,
        use_reranker: bool = True,
        use_source_weight: bool = True,
        use_custom_chunk: bool = True,
    ):
        self.name = name
        self.use_judge = use_judge
        self.use_reranker = use_reranker
        self.use_source_weight = use_source_weight
        self.use_custom_chunk = use_custom_chunk

    def describe(self) -> str:
        """生成配置的人类可读描述"""
        parts = []
        parts.append(f"judge={'on' if self.use_judge else 'OFF'}")
        parts.append(f"reranker={'on' if self.use_reranker else 'OFF'}")
        parts.append(f"source_weight={'on' if self.use_source_weight else 'OFF'}")
        parts.append(f"custom_chunk={'on' if self.use_custom_chunk else 'OFF'}")
        return ", ".join(parts)


def build_configs(disabled: List[str]) -> List[AblationConfig]:
    """根据 --disable 参数构建本次要跑的配置组"""
    disabled = set(disabled)

    def _off(key: str) -> bool:
        return key in disabled

    return [
        AblationConfig(
            name="full",
            use_judge=not _off("judge"),
            use_reranker=not _off("reranker"),
            use_source_weight=not _off("source_weight"),
            use_custom_chunk=not _off("custom_chunk"),
        ),
    ]


# ============================================================
# 单条样本跑一遍 RAG 管线（模块可开关）
# ============================================================

async def run_single(
    query: str,
    cfg: AblationConfig,
) -> Dict:
    """
    对单条 query 执行消融版 RAG 管线。
    与 src/core/workflow.py 的完整链路保持一致，仅按配置跳过指定模块。
    """
    t0 = time.time()

    # 1. 问题分类（L1 关键词规则 + L2 大模型兜底）—— 全配置共用
    classification = classifier.classify(query)

    # 2. 双源并行检索（本地知识库 + PubMed）—— 全配置共用
    local_results, pubmed_results, _ = await retriever.retrieve(query, classification)

    # 3. 重排序（可消融）
    if cfg.use_reranker:
        if cfg.use_source_weight:
            # 场景化来源权重：常见病优先本地指南、前沿问题优先文献
            source_weights = classifier.get_source_weight(query, classification)
        else:
            # 无来源权重：退化为两库等权
            source_weights = (0.5, 0.5)
        reranked = reranker.rerank(
            query, local_results, pubmed_results,
            source_weights=source_weights,
        )
    else:
        # 无重排：直接拼接两库结果（按原始检索顺序）
        reranked = (local_results + pubmed_results)[: settings.rerank_top_k]

    # 4. 拼装上下文
    context_text = reranker.build_context_text(reranked)

    # 5. 答案生成（Qwen-14B GPTQ INT4）
    answer, _ = generator.generate(query=query, context_text=context_text)

    # 6. 幻觉校验（可消融）
    judge_result = {}
    if cfg.use_judge:
        passed, judge_result, _ = judge.validate(
            query=query,
            answer=answer,
            retrieved_contexts=reranked,
            trigger_high_judge=classifier.should_trigger_high_judge(query),
        )
        if not passed:
            # 校验未通过：按完整链路逻辑重试一次（简化：直接标记未通过）
            judge_result["passed"] = False
    else:
        judge_result["passed"] = True  # 无校验 = 默认放行（这是消融对照组）

    return {
        "query": query,
        "answer": answer,
        "judge_result": judge_result,
        "latency_ms": (time.time() - t0) * 1000,
    }


# ============================================================
# 测试集评测
# ============================================================

def evaluate(
    test_data: List[Dict],
    cfg: AblationConfig,
    max_samples: Optional[int] = None,
) -> Dict:
    """在 MedQA 测试集上评测单组配置"""
    import asyncio

    samples = test_data[:max_samples] if max_samples else test_data
    correct = 0
    latencies = []
    details = []

    for i, item in enumerate(samples):
        query = item.get("question", "")
        expected = item.get("answer", "")

        try:
            result = asyncio.run(run_single(query, cfg))
            latencies.append(result["latency_ms"])

            # 评测口径：标准答案出现在生成结果中即命中（与 run_baseline.py 一致）
            hit = bool(expected and expected in result["answer"])
            correct += 1 if hit else 0
            details.append({"query": query, "expected": expected, "hit": hit})

        except Exception as e:
            logger.error(f"样本失败 [{i}]: {e}")

        if (i + 1) % 50 == 0:
            logger.info(f"[{cfg.name}] 进度 {i + 1}/{len(samples)}, "
                        f"当前准确率 {correct / (i + 1):.2%}")

    total = len(samples)
    return {
        "config": cfg.name,
        "description": cfg.describe(),
        "total_samples": total,
        "top1_accuracy": round(correct / total, 4) if total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "correct": correct,
    }


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="消融实验框架：验证各优化模块的独立贡献",
    )
    parser.add_argument(
        "--disable", nargs="*", default=[],
        choices=["judge", "reranker", "source_weight", "custom_chunk"],
        help="要关闭的模块（可多个）",
    )
    parser.add_argument(
        "--test-file", type=str, default="data/medqa_test.json",
        help="MedQA 测试集路径",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="只评测前 N 条（快速验证用）",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="结果输出到 JSON 文件",
    )
    args = parser.parse_args()

    # 1. 加载测试集
    test_path = Path(args.test_file)
    if not test_path.exists():
        logger.warning(
            f"测试集不存在: {test_path}。\n"
            "请先准备 MedQA 测试集（data/medqa_test.json），"
            "或使用 --test-file 指定路径。\n"
            "【提示】当前为框架设计验证，可先用 --samples 5 配小样本测试集跑通流程。"
        )
        return

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"加载测试集: {test_path} ({len(test_data)} 条)")

    # 2. 构建配置组并评测
    configs = build_configs(args.disable)
    results = []
    for cfg in configs:
        logger.info(f"===== 评测配置 [{cfg.name}]: {cfg.describe()} =====")
        result = evaluate(test_data, cfg, max_samples=args.samples)
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # 3. 汇总对比表
    print("\n========== 消融实验汇总 ==========")
    print(f"{'配置':<24}{'准确率':<10}{'样本数':<8}{'平均延迟ms'}")
    print("-" * 56)
    for r in results:
        print(f"{r['config']:<24}{r['top1_accuracy']:<10.2%}"
              f"{r['total_samples']:<8}{r['avg_latency_ms']:<12.2f}")

    # 4. 可选：输出到文件
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
