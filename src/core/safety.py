"""
Medical content safety filter, emergency detection, and compliance enforcement.

Covers:
  - Emergency keyword detection (chest pain, bleeding, etc.) -> force redirect to 120
  - Content safety: input-side sensitive word filtering
  - Output-side compliance: block diagnosis, prescription, dosage guidance
  - Standardized refusal templates for out-of-scope / emergency queries
"""

import re
from typing import Tuple, Optional

# ============================================================
# Emergency patterns - force redirect, no answer allowed
# 设计原则: 单一咨询词(胸闷/心慌)允许普通问诊; 关键词 + 严重修饰词(剧烈/突然/无法/晕/倒)
# 或独立致命医学术语(心绞痛/心梗/脑卒中等)才触发急症拦截
# ============================================================
EMERGENCY_PATTERNS = [
    # 急性胸痛/胸闷 - 任何急性发作都应警惕, 移除原有修饰词限制
    (re.compile(r"(突然|剧烈|持续|不停|一直).{0,8}(胸痛|胸闷|胸口疼|胸口痛)"), "急性胸痛"),
    (re.compile(r"(胸痛|胸闷|胸口疼|胸口痛).{0,8}(剧烈|严重|压榨|撕裂|出汗|想吐|要死)"), "急性胸痛"),
    # 独立心脏主诉(无咨询词即视为高风险, 避免"我心脏疼"被误判为普通问诊)
    (re.compile(r"(心脏疼|心口疼|心窝疼|心前区疼|心脏痛|心口痛|心窝痛|心前区痛)(?![得了很已])([!。?！啊呀吧呢]|$)"), "急性心脏主诉"),
    # 心脏症状 - 口语化/同义表达 + 严重修饰
    (re.compile(r"(心脏|心口|心窝|心前区).{0,4}(疼|痛|绞痛).{0,3}(厉害|剧烈|严重|突然|一直|不停|要死|受不了)"), "急性心脏症状"),
    (re.compile(r"(心脏|心口|心窝).{0,4}(疼|痛|绞痛|难受).{0,8}(喘不上气|出汗|要死|快死了)"), "急性心脏症状"),
    # 严重心血管事件医学术语
    (re.compile(r"(心绞痛|心梗|心肌梗死|急性冠脉综合征|主动脉夹层)"), "急性心血管事件"),
    (re.compile(r"(心脏骤停|没有心跳|脉搏.*没有|心跳.*停了)"), "心脏骤停"),
    # 大出血
    (re.compile(r"(大出血|大量吐血|大量咯血|便血不止|血崩|动脉.*出血)"), "大出血"),
    # 意识障碍
    (re.compile(r"(意识不清|昏迷|晕倒|昏倒|不省人事|叫不醒|抽搐|癫痫发作|羊癫疯)"), "意识障碍"),
    # 急性呼吸困难
    (re.compile(r"(呼吸困难|喘不上气|窒息|无法呼吸|呼吸停止|憋气.*要死)"), "急性呼吸困难"),
    # 急性中毒
    (re.compile(r"(急性中毒|误食.*毒|吃了.*药.*昏迷|农药.*中毒|煤气.*中毒)"), "急性中毒"),
    # 严重外伤
    (re.compile(r"(触电|雷击|溺水|严重烧伤|大面积烫伤|车祸.*重伤|高空坠落)"), "意外伤害"),
    # 急性腹痛/急腹症
    (re.compile(r"(剧烈|突然).{0,4}(腹痛|肚子疼|胃痛).{0,8}(大汗|昏迷|便血|呕吐不止)"), "急腹症"),
    # 急性脑卒中
    (re.compile(r"(中风|脑卒中|脑出血|脑梗).{0,8}(突然|发作|症状)"), "脑卒中"),
    (re.compile(r"(半边.*不能动|嘴角歪斜|说话不清|突然.*不会说话)"), "脑卒中预警"),
    # 急性头痛(脑出血/动脉瘤预警)
    (re.compile(r"(突然|剧烈).{0,4}(头痛|头炸裂|脑袋要炸)"), "急性头痛"),
    # 过敏性休克
    (re.compile(r"(过敏.*休克|过敏.*呼吸困难|过敏.*喉头水肿|过敏.*血压.*低)"), "过敏性休克"),
    # 自杀倾向
    (re.compile(r"(自杀|不想活了|怎么死|安眠药.*自杀|割腕|跳楼)"), "自杀倾向"),
]

EMERGENCY_RESPONSE = (
    "【紧急提示】根据您的描述，您的情况可能属于医疗急症，需要立即就医！\n\n"
    "请立即拨打 120 急救电话或前往最近的医院急诊科就诊。\n"
    "在等待救援期间，请保持镇静，不要随意移动患者（如疑似脊柱损伤），"
    "如有条件可进行基本生命支持（如 CPR）。\n\n"
    "本系统无法提供急救指导，请务必立即寻求专业医疗救助！"
)

# ============================================================
# Input-side sensitive content filter
# ============================================================
SENSITIVE_INPUT_PATTERNS = [
    (re.compile(r"(自杀|自残|割腕|跳楼|上吊|怎么死最)"), "自残/自杀相关内容"),
    (re.compile(r"(买.*毒品|吸毒|冰毒|海洛因|摇头丸|K粉)"), "非法药物咨询"),
    (re.compile(r"(制造.*毒药|制毒|毒死别人|怎么下毒)"), "非法/危害他人行为"),
    (re.compile(r"(堕胎.*药|打胎.*药|流产.*自己|米非司酮.*买)"), "非法堕胎药物咨询"),
]

SENSITIVE_RESPONSE = (
    "您的问题涉及本系统无法提供帮助的内容。如果您正处于心理危机中，"
    "请拨打全国心理援助热线：400-161-9995 或 010-82951332。"
)

# ============================================================
# Output-side compliance check
# ============================================================
DIAGNOSIS_PATTERNS = [
    re.compile(r"你(得了|患有|得的是|确诊为|肯定是|绝对是|一定(是|患))"),
    re.compile(r"诊断(结果|结论)?[：:是为]\s*"),
    re.compile(r"这(就)?是典型的?\s*(病|症|炎|癌)"),
]

PRESCRIPTION_PATTERNS = [
    re.compile(r"(建议|推荐|可以)(服用?|使用|购买)\s*[一-鿿]{2,8}(片|胶囊|颗粒|口服液|注射液)"),
    re.compile(r"(每次|每日|每天|一天)\s*\d+\s*(片|粒|次|mg|g|毫升)"),
    re.compile(r"处方[：:]*[一-鿿\d]+"),
    re.compile(r"你可以(买|购买|吃|服用)\s*[一-鿿]{2,8}"),
]

DOSAGE_PATTERNS = [
    re.compile(r"(用法|用量|剂量)[：:是为]*\s*[一-鿿\d]+"),
    re.compile(r"(每次|每日|每天)\s*\d+\s*(片|粒|毫克|mg|克|g)"),
]

# ============================================================
# Boundary refusal templates
# ============================================================
LOW_CONFIDENCE_RESPONSE = (
    "抱歉，根据当前可用的医疗参考资料，我无法对您的问题给出准确可靠的回答。\n\n"
    "建议您：\n"
    "1. 咨询执业医师获取专业诊断意见\n"
    "2. 前往正规医疗机构进行相关检查\n"
    "3. 提供更详细的症状描述以便更精准的检索\n\n"
    "本系统仅提供医学科普参考，不构成诊疗建议。"
)

OUT_OF_SCOPE_RESPONSE = (
    "您的问题超出了本系统的服务范围。本系统仅能提供基于权威医学资料的科普参考信息，"
    "不能提供诊断结论、处方方案或个性化治疗建议。\n\n"
    "如有健康问题，请咨询专业医生或前往正规医疗机构就诊。"
)


class MedicalSafetyFilter:
    """Medical content safety and compliance enforcement layer."""

    def check_emergency(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if query describes an emergency requiring immediate medical attention.
        Returns: (is_emergency, matched_pattern, response_message)
        """
        for pattern, label in EMERGENCY_PATTERNS:
            if pattern.search(query):
                return True, label, EMERGENCY_RESPONSE
        return False, None, None

    def check_input_safety(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if input contains sensitive/illegal content.
        Returns: (is_blocked, reason, response_message)
        """
        for pattern, label in SENSITIVE_INPUT_PATTERNS:
            if pattern.search(query):
                return True, label, SENSITIVE_RESPONSE
        return False, None, None

    def check_output_compliance(self, answer: str, retrieved_contexts: list = None) -> Tuple[bool, list]:
        """
        Check if model output contains prohibited content.
        Returns: (is_compliant, violations_list)
        """
        violations = []

        # Check for definitive diagnosis
        for pat in DIAGNOSIS_PATTERNS:
            matches = pat.findall(answer)
            if matches:
                violations.append({
                    "type": "unauthorized_diagnosis",
                    "detail": f"疑似越权诊断: {str(matches[:2])}",
                    "severity": "high",
                })
                break

        # Check for prescription recommendations
        for pat in PRESCRIPTION_PATTERNS:
            matches = pat.findall(answer)
            if matches:
                violations.append({
                    "type": "prescription_guidance",
                    "detail": f"疑似处方推荐: {str(matches[:2])}",
                    "severity": "high",
                })
                break

        # Check for dosage guidance
        for pat in DOSAGE_PATTERNS:
            matches = pat.findall(answer)
            if matches:
                violations.append({
                    "type": "dosage_guidance",
                    "detail": f"疑似剂量指导: {str(matches[:2])}",
                    "severity": "high",
                })
                break

        # Check source attribution
        if retrieved_contexts and len(retrieved_contexts) > 0:
            has_source = any(
                marker in answer
                for marker in ["来源", "参考", "依据", "根据", "指南", "知识库", "文献"]
            )
            if not has_source:
                violations.append({
                    "type": "missing_source_attribution",
                    "detail": "回答未标注内容来源",
                    "severity": "medium",
                })

        return len(violations) == 0, violations

    def get_refusal_response(self, reason: str = "low_confidence") -> str:
        """Return standardized refusal response based on reason."""
        responses = {
            "low_confidence": LOW_CONFIDENCE_RESPONSE,
            "out_of_scope": OUT_OF_SCOPE_RESPONSE,
            "emergency": EMERGENCY_RESPONSE,
        }
        return responses.get(reason, OUT_OF_SCOPE_RESPONSE)


# Global singleton
safety_filter = MedicalSafetyFilter()
