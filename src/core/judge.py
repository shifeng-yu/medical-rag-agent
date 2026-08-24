# ============================================================
# LLM-Judge 幻觉校验模块 (双层)
# 第一层: 规则校验 (医疗实体可溯源/违规内容/来源标注)
# 第二层: 模型交叉校验 (三维打分: 事实一致性/逻辑合理性/回答有用性)
# 流程: 规则校验 → 模型校验 → 均通过才输出
# ============================================================

import re
import json
from typing import Dict, Tuple, List
from loguru import logger
from config.settings import settings
from src.utils.helpers import (
    extract_medical_entities,
    check_source_attribution,
    load_prompt_template,
    safe_json_parse,
)


class HallucinationJudge:
    """
    幻觉防控两层校验机制
    - 规则层: 硬红线拦截 (编造实体/越权诊断/处方推荐)
    - 模型层: 三维打分 (事实一致性/逻辑合理性/回答有用性), 均>6分通过
    """

    def __init__(self):
        self.score_threshold = settings.judge_score_threshold  # 6
        logger.info(
            f"LLM-Judge 已初始化 (分数阈值={self.score_threshold})"
        )

    # ==================== 第一层: 规则校验  ====================

    def rule_check(
        self, answer: str, retrieved_contexts: List[Dict]
    ) -> Tuple[bool, str, Dict]:
        """
        规则层硬红线检查 
        三条规则:
        1. 医疗实体可溯源: 所有药品名/疾病名必须在检索上下文中能找到
        2. 违规内容拦截: 不能有确诊结论/处方推荐
        3. 来源标注: 必须标注内容来源
        返回: (passed, fail_reason, details)
        """
        details = {}

        # 检查1: 医疗实体可溯源 
        entities = extract_medical_entities(answer)
        context_text = " ".join(c.get("content", "") for c in retrieved_contexts)

        unverified_entities = []
        for entity_type in ["diseases", "drugs"]:
            for entity in entities.get(entity_type, []):
                if entity not in context_text:
                    unverified_entities.append(entity)
                    logger.warning(
                        f"[规则校验] 发现未溯源实体: {entity} (类型={entity_type})"
                    )

        details["unverified_entities"] = unverified_entities
        if unverified_entities:
            return False, (
                f"回答中存在无法在参考资料中溯源的信息: {', '.join(unverified_entities)}"
            ), details

        # 检查2: 违规内容拦截 ( 不得确诊/开处方)
        violation = self._check_violations(answer)
        details["violations"] = violation
        if violation:
            return False, f"回答包含违规医疗内容: {'; '.join(violation)}", details

        # 检查3: 来源标注 ( 保证可追溯)
        has_source = check_source_attribution(answer)
        details["has_source_attribution"] = has_source
        if not has_source:
            return False, "回答未标注内容来源", details

        logger.info("[规则校验] 通过: 实体可溯源, 无违规内容, 已标注来源")
        return True, "", details

    def _check_violations(self, answer: str) -> List[str]:
        """
        检查违规医疗内容 
        - 直接下确诊结论
        - 推荐处方药
        - 其他医疗合规红线
        来源: 「检查回答里有没有违规内容，比如直接下确诊结论、推荐处方药」
        """
        violations = []

        # 越权诊断模式 
        diagnosis_patterns = [
            re.compile(r"你(?:得了|患有?|得的是?|确诊为|肯定是?|绝对是?)\s*\S{2,10}(?:病|症|炎|癌)"),
            re.compile(r"(?:明确|肯定|绝对|一定)(?:是|患有?)\s*\S{2,10}(?:病|症|炎)"),
            re.compile(r"诊断(?:结果|结论)?[：:是为]\s*[一-鿿]{2,10}(?:病|症|炎|癌)"),
            re.compile(r"这(?:就)?是\s*\S{2,10}(?:病|症|炎|癌)"),
        ]
        for pat in diagnosis_patterns:
            if pat.search(answer):
                violations.append("越权给出确定诊断结论")
                break

        # 处方推荐模式 
        prescription_patterns = [
            re.compile(r"(?:建议|推荐)(?:服用?|使用|购买)\s*\S{2,8}(?:片|胶囊|颗粒|口服液|注射液|丸|散|膏|丹|剂)"),
            re.compile(r"开(?:了?)?处方[：:]*[一-鿿a-zA-Z0-9]"),
            re.compile(r"(?:每次|每日|一天)\s*\d+\s*(?:片|粒|次|mg|g)"),
            re.compile(r"你可以(?:买|购买|吃|服用)\s*\S{2,8}(?:片|胶囊|颗粒|口服液|注射液)"),
        ]
        for pat in prescription_patterns:
            if pat.search(answer):
                violations.append("推荐处方类药物")
                break

        return violations

    # ==================== 第二层: 模型交叉校验  ====================

    def llm_judge(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[Dict],
    ) -> Tuple[bool, Dict[str, float], str]:
        """
        LLM-Judge 三维打分 
        维度:
        1. 事实一致性 (factual_consistency): 回答是否基于检索上下文，有无编造
        2. 逻辑合理性 (logical_coherence): 回答逻辑是否通顺，有无矛盾
        3. 回答有用性 (answer_helpfulness): 是否能解决用户问题
        每项 0-10 分，均 >6 分才通过
        来源: 「从三个维度做0-10分打分」「均高于6分通过」
        """
        context_text = "\n".join(
            f"[{c.get('source', '')}] {c.get('content', '')}"
            for c in retrieved_contexts
        )

        judge_prompt = load_prompt_template("judge")
        if not judge_prompt:
            judge_prompt = (
                "你是一个医疗问诊回答质量的评审专家。请从以下三个维度，"
                "对助手的回答进行 0-10 分的评分：\n\n"
                "1. 事实一致性 (factual_consistency): "
                "回答是否完全基于参考资料，有没有编造不存在的信息？\n"
                "2. 逻辑合理性 (logical_coherence): "
                "回答的逻辑是否通顺，前后有没有矛盾？\n"
                "3. 回答有用性 (answer_helpfulness): "
                "回答是否能切实解决用户的问题？\n\n"
                "## 评分要求\n"
                "- 每个维度 0-10 分\n"
                "- 如果发现回答编造了参考资料中不存在的信息，事实一致性打 0 分\n"
                "- 如果回答存在医疗合规问题（越权诊断、推荐处方），逻辑合理性打 0 分\n"
                "- 必须以 JSON 格式输出：\n"
                '{{"factual_consistency": 分数, "logical_coherence": 分数, '
                '"answer_helpfulness": 分数, "feedback": "简要反馈"}}\n\n'
                "## 用户问题\n{query}\n\n"
                "## 参考资料\n{context_text}\n\n"
                "## 助手回答\n{answer}\n\n"
                "## 评审结果 (JSON)"
            )

        prompt = judge_prompt.replace("{query}", query)
        prompt = prompt.replace("{context_text}", context_text[:3000])  # 截断控制长度
        prompt = prompt.replace("{answer}", answer)

        try:
            from src.core.generator import generate_text

            raw_output = generate_text(prompt, max_tokens=256, temperature=0.0)
            result = safe_json_parse(raw_output, default={
                "factual_consistency": 0,
                "logical_coherence": 0,
                "answer_helpfulness": 0,
                "feedback": "解析失败",
            })

            scores = {
                "factual_consistency": float(result.get("factual_consistency", 0)),
                "logical_coherence": float(result.get("logical_coherence", 0)),
                "answer_helpfulness": float(result.get("answer_helpfulness", 0)),
            }
            feedback = result.get("feedback", "")

            # 判断是否全部通过 ( 均>6分)
            passed = all(s >= self.score_threshold for s in scores.values())

            logger.info(
                f"[LLM-Judge] 打分: "
                f"事实一致性={scores['factual_consistency']}, "
                f"逻辑合理性={scores['logical_coherence']}, "
                f"有用性={scores['answer_helpfulness']} | "
                f"结果={'通过' if passed else '不通过'}"
            )
            return passed, scores, feedback

        except Exception as e:
            logger.error(f"LLM-Judge 执行失败: {e}")
            # 校验失败时保守处理：通过（避免阻塞），但记录为低置信度
            return True, {
                "factual_consistency": 5,
                "logical_coherence": 5,
                "answer_helpfulness": 5,
            }, f"校验异常: {e}"

    # ==================== 完整校验流程  ====================

    def validate(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[Dict],
        trigger_high_judge: bool = True,
    ) -> Tuple[bool, Dict, str]:
        """
        完整两层校验流程 
        返回: (最终通过, 校验详情, 反馈信息)
        """
        # 第一层: 规则校验 
        rule_passed, rule_reason, rule_details = self.rule_check(
            answer, retrieved_contexts
        )
        if not rule_passed:
            return False, {
                "layer": "rule",
                "reason": rule_reason,
                "details": rule_details,
            }, rule_reason

        # 第二层: 模型交叉校验 
        # 涉及核心医疗问题才触发LLM-Judge，普通咨询可跳过
        if not trigger_high_judge:
            logger.info("[LLM-Judge] 跳过模型校验（非高风险问题）")
            return True, {"layer": "rule_only", "details": rule_details}, ""

        judge_passed, scores, feedback = self.llm_judge(
            query, answer, retrieved_contexts
        )

        if not judge_passed:
            return False, {
                "layer": "judge",
                "scores": scores,
                "feedback": feedback,
                "details": rule_details,
            }, feedback

        return True, {
            "layer": "both",
            "scores": scores,
            "details": rule_details,
        }, ""


# 全局单例
judge = HallucinationJudge()
