# ============================================================
# BGE-M3 两阶段重排序 + 场景化来源权重
# 来源：项目需求
# ============================================================

import time
from typing import List, Dict, Tuple
import numpy as np
from loguru import logger
from config.settings import settings


class SourceAwareReranker:
    """
    来源感知重排序器 (项目需求)
    两阶段:
    - 粗排: 按 Milvus 向量相似度快速排序
    - 精排: BGE-M3 Cross-Encoder 逐对打分
    - 来源权重: 根据分类结果对分数加权
    """

    def __init__(self):
        self._cross_encoder = None
        logger.info("重排序器已初始化")

    @property
    def model(self):
        """懒加载 BGE-M3 Cross-Encoder 模式 (项目需求)"""
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
    ) -> List[Dict]:
        """
        两阶段重排序 (项目需求)
        参数:
            source_weights: (local_weight, pubmed_weight) 来源权重
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
            # Cross-Encoder 相关性打分 (项目需求)
            cross_score = self._cross_score(query, item["content"])

            # 来源权重加权 ( 常见病优先本地指南, 前沿优先文献)
            source = item.get("source", "")
            if source == "local_kb":
                weighted_score = cross_score * local_weight
            elif source == "pubmed":
                weighted_score = cross_score * pubmed_weight
            else:
                weighted_score = cross_score * 0.5

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
            f"→ 精排{len(final)}, 耗时={latency_ms:.0f}ms"
        )
        return final

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
        构建检索上下文文本 (项目需求)
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
