"""
Medical knowledge authority grading and metadata management.

Covers:
  - Knowledge authority tiering (Clinical Guideline > Drug Label > Textbook > Popular Science)
  - Medical terminology normalization (disease/drug alias mapping)
  - Source tracking with confidence labels
  - Department-based metadata filtering support
"""

from typing import Dict, List, Optional

# ============================================================
# Authority tier definitions
# ============================================================
AUTHORITY_TIERS = {
    "tier_1": {
        "label": "临床指南",
        # "指南"作为通用兜底 marker：诊治指南/防治指南/康复指南/用药指南等
        # 均属 Tier1 临床指南；特定机构 marker 优先列前面保持可读性。
        "sources": ["国家卫健委", "中华医学会", "中国临床指南", "NMPA", "FDA", "WHO", "指南"],
        "confidence": 0.95,
        "priority": 1,
    },
    "tier_2": {
        "label": "药品说明书",
        "sources": ["NMPA药品说明书", "FDA Label", "中国药典"],
        "confidence": 0.90,
        "priority": 2,
    },
    "tier_3": {
        "label": "权威教材",
        "sources": ["全国高等学校教材", "临床诊疗指南", "专家共识"],
        "confidence": 0.80,
        "priority": 3,
    },
    "tier_4": {
        "label": "合规科普",
        "sources": ["三甲医院官方科普", "CDC", "卫健委官方科普"],
        "confidence": 0.65,
        "priority": 4,
    },
}

# ============================================================
# Medical terminology normalization (alias -> standard name)
# ============================================================
DISEASE_ALIAS_MAP = {
    "心梗": "心肌梗死",
    "心衰": "心力衰竭",
    "脑梗": "脑梗死",
    "脑出血": "脑出血",
    "中风": "脑卒中",
    "感冒": "上呼吸道感染",
    "发烧": "发热",
    "拉肚子": "腹泻",
    "肚子疼": "腹痛",
    "胃疼": "上腹痛",
    "胸痛": "胸痛",
    "头疼": "头痛",
    "头晕": "眩晕",
    "嗓子疼": "咽痛",
    "打喷嚏流鼻涕": "上呼吸道卡他症状",
    "过敏性鼻炎": "变应性鼻炎",
    "甲状腺功能亢进": "甲亢",
    "糖尿病": "糖尿病",
    "高血压": "高血压",
    "冠心病": "冠心病",
    "脂肪肝": "非酒精性脂肪性肝病",
    "胆结石": "胆囊结石",
    "肾结石": "肾结石",
    "前列腺增生": "良性前列腺增生",
    "老慢支": "慢性支气管炎",
    "肺气肿": "肺气肿",
    "肝炎": "病毒性肝炎",
    "肝硬化": "肝硬化",
    "尿毒症": "终末期肾病",
    "牛皮癣": "银屑病",
}

DRUG_ALIAS_MAP = {
    "阿司匹林": "阿司匹林",
    "扑热息痛": "对乙酰氨基酚",
    "消炎痛": "吲哚美辛",
    "安定": "地西泮",
    "降压药": "抗高血压药",
    "降糖药": "口服降糖药",
    "抗生素": "抗菌药物",
    "止疼药": "镇痛药",
    "胃药": "胃黏膜保护剂/抑酸药",
    "感冒药": "复方感冒制剂",
}

# ============================================================
# Department/specialty mapping
# ============================================================
DEPARTMENT_MAP = {
    "呼吸内科": ["感冒", "咳嗽", "肺炎", "哮喘", "COPD", "结核"],
    "心血管内科": ["高血压", "冠心病", "心梗", "心衰", "心律失常"],
    "神经内科": ["头痛", "脑梗", "癫痫", "帕金森", "阿尔茨海默"],
    "消化内科": ["胃炎", "胃溃疡", "腹泻", "便秘", "肝炎"],
    "内分泌科": ["糖尿病", "甲亢", "甲减", "痛风", "肥胖"],
    "骨科": ["骨折", "关节炎", "颈椎病", "腰椎间盘突出"],
    "皮肤科": ["湿疹", "荨麻疹", "银屑病", "痤疮", "皮炎"],
    "精神科": ["抑郁", "焦虑", "失眠", "双相障碍"],
    "肾内科": ["肾炎", "肾病综合征", "肾结石", "肾功能不全"],
    "感染科": ["发热", "乙肝", "结核", "HIV", "流感"],
}


class KnowledgeGrader:
    """Medical knowledge authority grading and metadata management."""

    @staticmethod
    def grade_source(source_text: str) -> Dict:
        """Grade knowledge authority based on source text markers."""
        for tier_key, tier_info in AUTHORITY_TIERS.items():
            for source_marker in tier_info["sources"]:
                if source_marker in source_text:
                    return {
                        "tier": tier_key,
                        "label": tier_info["label"],
                        "confidence": tier_info["confidence"],
                        "priority": tier_info["priority"],
                    }
        return {
            "tier": "tier_4",
            "label": "合规科普",
            "confidence": 0.60,
            "priority": 4,
        }

    @staticmethod
    def normalize_disease(query: str) -> str:
        """Normalize disease aliases to standard medical terms."""
        result = query
        for alias, standard in DISEASE_ALIAS_MAP.items():
            if alias in result:
                result = result.replace(alias, standard)
        return result

    @staticmethod
    def normalize_drug(query: str) -> str:
        """Normalize drug aliases to standard names."""
        result = query
        for alias, standard in DRUG_ALIAS_MAP.items():
            if alias in result:
                result = result.replace(alias, standard)
        return result

    @staticmethod
    def normalize_query(query: str) -> str:
        """Full query normalization for medical terminology."""
        return KnowledgeGrader.normalize_drug(
            KnowledgeGrader.normalize_disease(query)
        )

    @staticmethod
    def get_department(keyword: str) -> Optional[str]:
        """Map a medical keyword to the relevant clinical department."""
        for dept, keywords in DEPARTMENT_MAP.items():
            if any(kw in keyword for kw in keywords):
                return dept
        return None

    @staticmethod
    def build_metadata(source_text: str, content: str, publish_time: str = "") -> Dict:
        """Build enriched metadata for a knowledge chunk."""
        grade = KnowledgeGrader.grade_source(source_text)
        return {
            "authority_tier": grade["tier"],
            "authority_label": grade["label"],
            "confidence": grade["confidence"],
            "publish_time": publish_time,
            "source_type": source_text[:32],
        }


# Global singleton
knowledge_grader = KnowledgeGrader()
