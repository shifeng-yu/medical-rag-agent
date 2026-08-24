# ============================================================
# 工具函数模块
# ============================================================

import re
import json
from pathlib import Path
from typing import List, Tuple, Set, Optional
from loguru import logger


# ---- Token 计数 ( 用 Qwen tokenizer 计算) ----
# 全局延迟加载 tokenizer 避免启动时过重
_tokenizer = None


def _get_tokenizer():
    """懒加载 Qwen tokenizer """
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            from config.settings import settings

            _tokenizer = AutoTokenizer.from_pretrained(
                settings.qwen_model_path, trust_remote_code=True
            )
            logger.info("Tokenizer 加载完成")
        except Exception as e:
            logger.warning(f"Tokenizer 加载失败，回退到字符估算: {e}")
            _tokenizer = "fallback"
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    计算文本 token 数 ( 用于触发上下文压缩)
    优先使用 Qwen tokenizer，失败时回退到字符/1.5 估算
    """
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    if tokenizer == "fallback":
        return len(text) // 1.5  # 中英文混合估算
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text) // 1.5


# ---- 医疗实体提取 ( 规则校验-检查药品名/疾病名) ----

# 医疗问诊常见实体正则模式 ( 医疗专属边界识别)
MEDICAL_PATTERNS = {
    "disease": re.compile(
        r"(?:疾病|病症|确诊|患有|得了|感染了?)([一-鿿]{2,10}(?:症|病|炎|瘤|癌|疹|疮|癣|肿|伤|血|痰|咳|烧|痛|痹|风|劳|郁|厥|脱|泄|秘|淋|浊|癃|闭|厥))"
    ),
    "drug": re.compile(
        r"(?:服用?|口服|注射|外敷|输液|配药|处方|开药|用药?)[：:\s]*([一-鿿]{2,8}(?:片|胶囊|颗粒|口服液|注射液|丸|散|膏|丹|剂|素|林|松|芬|坦|替尼|单抗|唑|汀|韦|米特|昔布))"
    ),
    "dosage": re.compile(r"(?:剂量|用量|用法)[：:\s]*([^，。\n]{2,30})"),
    "diagnosis": re.compile(r"(?:诊断结果|确诊为|诊断为|结论是)[：:\s]*([^，。\n]{2,50})"),
    "symptom": re.compile(
        r"(?:主诉|现病史|症状|表现|感觉|觉得|不舒服)[：:\s]*([^，。\n]{2,50})"
    ),
}


def extract_medical_entities(text: str) -> dict:
    """
    从文本中提取医疗实体 ( 规则校验层)
    返回: {"diseases": [...], "drugs": [...], "symptoms": [...]}
    """
    entities = {"diseases": [], "drugs": [], "symptoms": [], "diagnoses": []}

    # 提取疾病（匹配 pattern 中的捕获组）
    for m in MEDICAL_PATTERNS["disease"].finditer(text):
        entities["diseases"].append(m.group(1))
    for m in MEDICAL_PATTERNS["drug"].finditer(text):
        entities["drugs"].append(m.group(1))
    for m in MEDICAL_PATTERNS["symptom"].finditer(text):
        entities["symptoms"].append(m.group(1))
    for m in MEDICAL_PATTERNS["diagnosis"].finditer(text):
        entities["diagnoses"].append(m.group(1))

    return entities


# ---- 来源标注检查 ( 规则校验-追溯性检查) ----


def check_source_attribution(answer: str) -> bool:
    """
    检查回答是否标注了内容来源 ( 保证可追溯)
    来源: 「检查回答有没有标注内容的来源，保证可追溯」
    """
    source_markers = [
        r"\[来源[：:\s]*",
        r"\[参考[：:\s]*",
        r"来源[：:\s]*",
        r"参考[：:\s]*",
        r"依据[：:\s]*",
        r"根据[：:\s]*.*?(?:指南|文献|知识库)",
        r"PubMed",
        r"PMID",
    ]
    return any(re.search(pattern, answer, re.IGNORECASE) for pattern in source_markers)


# ---- 关键词匹��� ( L1快速规则匹配) ----

# 时效性表述模式 ( 判断是否涉及前沿/时效性)
TEMPORAL_PATTERNS = re.compile(
    r"最新|最近|今年|近[年月日]|新研究|新进展|前沿|突破|最新版|202[0-9]|新药|临床试验"
)


def load_keywords(file_path: str) -> Tuple[Set[str], Set[str]]:
    """
    加载常见病/常用药关键词 ( L1规则匹配)
    返回: (diseases_set, drugs_set)
    """
    diseases = set()
    drugs = set()
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"关键词文件不存在: {file_path}，使用内置默认词库")
        return _default_keywords()

    with open(path, "r", encoding="utf-8") as f:
        section = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if "疾病" in line:
                    section = "disease"
                elif "药品" in line or "药物" in line:
                    section = "drug"
                continue
            if section == "disease":
                diseases.add(line)
            elif section == "drug":
                drugs.add(line)
    return diseases, drugs


def _default_keywords() -> Tuple[Set[str], Set[str]]:
    """内置默认常见病/常用药关键词 ( 知识库覆盖的常见病/常用药标准名称)"""
    diseases = {
        "感冒", "发烧", "咳嗽", "头痛", "高血压", "糖尿病", "冠心病", "哮喘", "肺炎",
        "胃炎", "胃溃疡", "肝炎", "肾炎", "关节炎", "骨质疏松", "贫血", "甲亢",
        "甲减", "痛风", "湿疹", "皮炎", "鼻炎", "咽炎", "扁桃体炎", "结膜炎",
        "颈椎病", "腰椎间盘突出", "肩周炎", "前列腺炎", "月经不调", "阴道炎",
        "小儿发热", "小儿腹泻", "手足口病", "水痘", "带状疱疹", "过敏性鼻炎",
        "支气管炎", "肺气肿", "心绞痛", "心肌梗死", "脑梗塞", "脑出血", "脂肪肝",
        "肝硬化", "胆结石", "肾结石", "尿路感染", "膀胱炎", "痔疮", "肛裂",
        "静脉曲张", "深静脉血栓", "白癜风", "银屑病", "痤疮", "荨麻疹",
        "失眠", "焦虑症", "抑郁症", "强迫症", "偏头痛", "癫痫", "帕金森",
        "老年痴呆", "白内障", "青光眼", "中耳炎", "牙周炎", "口腔溃疡",
    }
    drugs = {
        "阿莫西林", "头孢", "阿奇霉素", "左氧氟沙星", "青霉素", "布洛芬",
        "对乙酰氨基酚", "阿司匹林", "氨氯地平", "硝苯地平", "氯沙坦", "缬沙坦",
        "二甲双胍", "格列美脲", "胰岛素", "奥美拉唑", "雷尼替丁", "蒙脱石散",
        "氯雷他定", "西替利嗪", "沙丁胺醇", "布地奈德", "孟鲁司特",
        "阿托伐他汀", "瑞舒伐他汀", "氯吡格雷", "华法林", "硝酸甘油",
        "复方甘草片", "川贝枇杷膏", "板蓝根", "连花清瘟", "藿香正气",
        "感冒灵", "感康", "快克", "泰诺", "芬必得", "扶他林", "开瑞坦",
        "思密达", "达喜", "吗丁啉", "金嗓子", "西瓜霜",
    }
    return diseases, drugs


def match_keywords(query: str, diseases: Set[str], drugs: Set[str]) -> Tuple[bool, bool]:
    """
    关键词快速匹配 ( 毫秒级匹配)
    返回: (has_disease_kw, has_temporal_kw)
    - has_disease_kw: 是否命中常见病/常用药关键词
    - has_temporal_kw: 是否包含时效性表述
    """
    has_disease = any(d in query for d in diseases)
    has_drug = any(d in query for d in drugs)
    has_temporal = bool(TEMPORAL_PATTERNS.search(query))
    return (has_disease or has_drug), has_temporal


# ---- Prompt 模板加载 ----


def load_prompt_template(name: str) -> str:
    """加载 prompt 模板文件"""
    from config.settings import settings

    path = Path(settings.prompts_dir) / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning(f"Prompt 模板不存在: {path}")
    return ""


# ---- JSON 安全解析 ----


def safe_json_parse(text: str, default=None):
    """安全解析 JSON，用于解析 LLM 返回的结构化输出"""
    try:
        # 尝试提取 JSON 块
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return default if default is not None else {}
