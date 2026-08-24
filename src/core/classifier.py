# ============================================================
# 两级问题分类模块
# 来源：规则预判断 + 大模型兜底两级分类
# - L1: 关键词规则快速匹配 (毫秒级)
# - L2: Qwen 二分类兜底 (<100ms)
# ============================================================

from typing import Tuple, Optional
from loguru import logger
from config.settings import settings
from src.utils.helpers import match_keywords, load_keywords, load_prompt_template


class QueryClassifier:
    """
    两级问题分类器 (项目需求)
    判断用户问题应路由到:
    - "local": 本地医疗知识库 (常见病/常规问诊)
    - "pubmed": PubMed 离线文献库 (罕见病/前沿/时效性)
    - "both": 并行双库检索
    """

    def __init__(self):
        # 加载关键词库 ( L1规则匹配)
        self.diseases, self.drugs = load_keywords(
            settings.classification_keywords_path
        )
        logger.info(
            f"问题分类器已初始化: {len(self.diseases)} 疾病词, "
            f"{len(self.drugs)} 药品词"
        )
        self._llm = None  # 延迟加载

    @property
    def llm_pipeline(self):
        """懒加载 Qwen 做分类 ( <100ms)"""
        if self._llm is None:
            # 使用轻量调用，不需要完整模型加载
            from src.core.generator import classify_with_llm
            self._llm = classify_with_llm
        return self._llm

    def classify(self, query: str) -> str:
        """
        两级分类主入口 (项目需求)
        返回: "local" | "pubmed" | "both"
        """
        # L1: 关键词规则快速匹配 ( 毫秒级)
        has_keyword, has_temporal = match_keywords(query, self.diseases, self.drugs)

        # 规则判断 (项目需求)
        if has_keyword and not has_temporal:
            # 命中常见病关键词 + 无时效性表述 → 本地知识库
            logger.debug(f"[分类-L1] 关键词命中, 路由到本地库 | query={query[:60]}")
            return "local"

        if not has_keyword and has_temporal:
            # 无常见病关键词 + 有时效性表述 → 文献库优先
            logger.debug(f"[分类-L1] 时效性表述, 路由到文献库 | query={query[:60]}")
            return "pubmed"

        # L2: 大模型轻量分类兜底 ( <100ms)
        logger.debug(f"[分类-L2] 关键词未明确匹配, 大模型兜底 | query={query[:60]}")
        return self._llm_classify(query)

    def _llm_classify(self, query: str) -> str:
        """
        L2: Qwen 二分类 (项目需求)
        Prompt: 极简二分类，只输出 "本地库" 或 "文献库"
        来源: 「Prompt 很简单，只要求输出"本地库"或"文献库"，全程不到100ms」
        """
        prompt_template = load_prompt_template("classification")
        if not prompt_template:
            prompt_template = (
                "你是一个医疗问诊问题分类器。根据用户的问题，判断应该查询哪个知识库。\n\n"
                "规则：\n"
                "- 如果问题是关于常见疾病、常规症状、基础用药、日常保健，输出「本地库」\n"
                "- 如果问题是关于罕见病、最新研究、前沿治疗、新药信息、时效性医学问题，输出「文献库」\n"
                "- 只输出「本地库」或「文献库」，不要输出任何其他内容\n\n"
                "用户问题：{query}\n"
                "分类结果："
            )

        prompt = prompt_template.replace("{query}", query)

        try:
            result = self.llm_pipeline(prompt, max_tokens=10).strip()
            if "文献" in result:
                return "pubmed"
            else:
                return "local"
        except Exception as e:
            logger.warning(f"LLM分类失败，默认路由到本地库: {e}")
            return "local"  # 安全降级

    def should_trigger_high_judge(self, query: str) -> bool:
        """
        判断是否需要触发高等级校验 (项目需求)
        涉及诊断、用药的核心医疗问题 → 触发 LLM-Judge
        普通咨询 → 可跳过校验提升速度
        来源: 「自主判断是否触发高等级校验」
        """
        # 诊断/用药高风险关键词
        high_risk_keywords = [
            "诊断", "确诊", "是不是", "什么病", "怎么回事",
            "用药", "吃药", "服药", "处方", "剂量", "用量",
            "治疗", "怎么办", "严重", "危险", "会死",
            "癌症", "肿瘤", "心脏病", "中风", "艾滋",
            "副作用", "禁忌", "过敏", "孕妇",
        ]
        return any(kw in query for kw in high_risk_keywords)

    def get_source_weight(self, query: str, classification: str) -> Tuple[float, float]:
        """
        获取双源检索权重 (项目需求/ 场景化来源权重)
        返回: (local_weight, pubmed_weight)
        """
        if classification == "local":
            return 0.7, 0.3   # 常见病优先本地库 (项目需求)
        elif classification == "pubmed":
            return 0.3, 0.7   # 前沿/罕见病优先文献 (项目需求)
        else:
            return 0.5, 0.5   # 默认均衡


# 全局单例
classifier = QueryClassifier()
