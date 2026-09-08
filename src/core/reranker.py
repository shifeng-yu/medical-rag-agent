# ============================================================
# BGE-M3 两阶段重排序 + 场景化来源权重 + 可选权威权重
# ============================================================

import re
import time
from typing import List, Dict, Tuple
import numpy as np
from loguru import logger
from config.settings import settings
from src.core.knowledge_grader import knowledge_grader


class SourceAwareReranker:
    """
    来源感知重排序器 
    两阶段:
    - 粗排: 按 Milvus 向量相似度快速排序
    - 精排: BGE-M3 Cross-Encoder 逐对打分
    - 来源权重: 根据分类结果对分数加权
    - 权威权重(可选): 按 chunk 的知识权威分级(Tier1-4)做小幅偏置，
      默认关闭(use_authority_weight=False)保持与线上/消融基线行为一致；
      开启后进入排序的可消融设计项，供权威分级效果 A/B 验证。
      权威打标(authority_tier/label/confidence)始终附加到结果项上，
      随 sources 返回客户端，实现"来源分级可追溯"。
    """

    def __init__(self):
        self._cross_encoder = None
        logger.info("重排序器已初始化")

    @property
    def model(self):
        """懒加载 BGE-M3 Cross-Encoder 模式 """
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("加载 BGE-M3 Cross-Encoder 用于精排")
                self._cross_encoder = CrossEncoder(
                    settings.bge_model_path,
                    device=settings.device,
                )
            except Exception as e:
                logger.error(f"Cross-Encoder 加载失败: {e}")
                raise
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        local_results: List[Dict],
        pubmed_results: List[Dict],
        source_weights: Tuple[float, float] = (0.5, 0.5),
        top_k: int = None,
        use_authority_weight: bool = False,
    ) -> List[Dict]:
        """
        两阶段重排序 
        参数:
            source_weights: (local_weight, pubmed_weight) 来源权重
            use_authority_weight: 是否叠加知识权威分级权重(Tier1-4)。
                默认 False = 与既有行为/消融基线一致；True = 进入排序偏置。
                无论开关如何，每个返回项都会附加 authority_tier /
                authority_label / authority_confidence 打标，随 sources 透出。
        返回: 排序后的合并结果
        """
        if top_k is None:
            top_k = settings.rerank_top_k

        start = time.perf_counter()

        # 合并双源结果
        all_results = local_results + pubmed_results
        if not all_results:
            return []

        # ---- Phase 1: 粗排 ( 按向量相似度快速排序) ----
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        coarse_candidates = all_results[: min(len(all_results), top_k * 3)]

        # ---- Phase 2: 精排 ( BGE-M3 Cross-Encoder 逐对打分) ----
        local_weight, pubmed_weight = source_weights
        scored = []
        for item in coarse_candidates:
            # Cross-Encoder 相关性打分 
            cross_score = self._cross_score(query, item["content"])

            # 权威分级打标（与来源权重正交；始终附加，供 sources 展示/后续策略使用）
            authority = self._authority_for(item)
            item["authority_tier"] = authority["tier"]
            item["authority_label"] = authority["label"]
            item["authority_confidence"] = authority["confidence"]

            # 来源权重加权 ( 常见病优先本地指南, 前沿优先文献)
            source = item.get("source", "")
            if source == "local_kb":
                weighted_score = cross_score * local_weight
            elif source == "pubmed":
                weighted_score = cross_score * pubmed_weight
            else:
                weighted_score = cross_score * 0.5

            # 权威权重（可选消融项）：Tier4(0.60)≈0.90x、Tier1(0.95)≈1.08x，
            # 只做小幅偏置——高相关性文档不被低档来源压过，低档同相关文档让位。
            if use_authority_weight:
                weighted_score *= (0.6 + 0.5 * authority["confidence"])

            item["rerank_score"] = weighted_score
            item["cross_score"] = cross_score
            scored.append(item)

        # 按加权分数排序
        scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # 截断到 top_k
        final = scored[:top_k]
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"重排序完成: {len(all_results)} → 粗排{len(coarse_candidates)} "
            f"→ 精排{len(final)}, 耗时={latency_ms:.0f}ms, "
            f"authority_weight={'on' if use_authority_weight else 'off'}"
        )
        return final

    # ---------------- 权威分级打标 ----------------

    @staticmethod
    def _authority_for(item: Dict) -> Dict:
        """为单个检索结果项判定知识权威分级。

        探测文本优先级：item 已有的来源型字段（含中文的 title / source_type /
        source）→ 从正文头部抓"来源/依据/指南/共识/说明书"等标记后的来源名
        （知识库语料形如 `**来源指南:** 中国冠心病康复指南`）→ 正文头部前 40 字兜底。
        判定交给 knowledge_grader.grade_source（AUTHORITY_TIERS 匹配）。
        """
        probe = ""
        for key in ("title", "source_type", "source"):
            value = str(item.get(key, "") or "").strip()
            if value and re.search(r"[\u4e00-\u9fff]", value):
                probe = value
                break
        if not probe:
            head = (item.get("content", "") or "")[:200]
            m = re.search(
                r"(?:来源|依据|参考|指南|共识|说明书|规范)[：:\s]*"
                r"([^。，,\n\s*]{2,30})",
                head,
            )
            probe = m.group(1) if m else head[:40]
        return knowledge_grader.grade_source(probe)

    def _cross_score(self, query: str, document: str) -> float:
        """BGE-M3 Cross-Encoder 逐对打分 ( 相关性打分)"""
        try:
            scores = self.model.predict([(query, document)])
            # 归一化到 0-1
            score = float(scores[0]) if scores is not None else 0.5
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Cross-Encoder打分失败: {e}")
            return 0.5  # 降级返回默认值

    def build_context_text(self, reranked_results: List[Dict]) -> str:
        """
        构建检索上下文文本 
        格式: 标注来源, 优先排序后的内容
        """
        if not reranked_results:
            return ""

        parts = []
        for i, item in enumerate(reranked_results):
            source_label = (
                "[本地权威医疗知识库]"
                if item.get("source") == "local_kb"
                else "[PubMed文献]"
            )
            title = item.get("title", "")
            content = item.get("content", "")
            pub_time = item.get("publish_time", "")

            header = f"{source_label} (相关度: {item.get('rerank_score', 0):.2f})"
            if title:
                header += f" {title}"
            if pub_time:
                header += f" ({pub_time})"

            parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)


# 全局单例
reranker = SourceAwareReranker()
