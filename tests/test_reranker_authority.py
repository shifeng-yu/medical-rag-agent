# ============================================================
# reranker 权威分级：打标 + 可选权威权重开关
# 覆盖（tier 接入重排）:
#   1. 无论开关，返回项都附加 authority_tier/label/confidence 打标
#   2. use_authority_weight=False 时 rerank_score 与旧行为一致（只乘来源权重）
#   3. use_authority_weight=True 时同相关性下 Tier1 指南排在无分级来源之前
#   4. 探测文本从正文"来源指南:"标记抓取（真实知识库语料形态）
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.core.reranker import SourceAwareReranker


def _make_item(title: str = "", content: str = "", score: float = 0.9,
               source: str = "local_kb") -> dict:
    return {
        "title": title,
        "source": source,
        "score": score,
        "content": content,
        "department": "心内科",
        "publish_time": "2024",
    }


GUIDELINE_CONTENT = (
    "# 心血管问诊\n\n"
    "**疾病类别:** 冠心病 | **来源指南:** 中国冠心病康复指南\n\n"
    "稳定期冠心病患者适合中等强度有氧运动，建议每周累计150分钟。"
)


class _FakeReranker(SourceAwareReranker):
    """用固定 cross_score 绕过模型，聚焦排序逻辑本身"""

    def _cross_score(self, query: str, document: str) -> float:
        return 0.8


class TestAuthorityTagging:
    def test_tag_attached_even_when_weight_off(self):
        r = _FakeReranker()
        item = _make_item(content=GUIDELINE_CONTENT)
        out = r.rerank("冠心病人能运动吗", [item], [], top_k=3,
                       use_authority_weight=False)
        assert len(out) == 1
        # 正文含"中国冠心病康复指南" → Tier1 临床指南
        assert out[0]["authority_tier"] == "tier_1"
        assert out[0]["authority_label"] == "临床指南"
        assert out[0]["authority_confidence"] == pytest.approx(0.95)
        # 权重关闭 → 只乘来源权重(等权 0.5)与旧行为一致
        assert out[0]["rerank_score"] == pytest.approx(0.8 * 0.5)

    def test_tag_from_title_chinese(self):
        r = _FakeReranker()
        item = _make_item(title="中国2型糖尿病防治指南", content="血糖管理内容")
        out = r.rerank("血糖控制目标", [item], [], top_k=3)
        assert out[0]["authority_tier"] == "tier_1"

    def test_unknown_source_falls_back_to_low_tier(self):
        r = _FakeReranker()
        item = _make_item(title="english_doc", content="普通科普短文，无来源标记")
        out = r.rerank("随便问问", [item], [], top_k=3)
        assert out[0]["authority_tier"] == "tier_4"


class TestAuthorityWeight:
    def test_weight_boosts_guideline_over_plain_when_equal_cross(self):
        r = _FakeReranker()
        guideline = _make_item(content=GUIDELINE_CONTENT, score=0.9)
        # 普通语料必须避开"来源/指南/共识"等分级触发词，确保判到 Tier4 兜底
        plain_content = "这是一段普通说明文字，没有任何标注信息，仅作对照。"
        plain = _make_item(content=plain_content, score=0.9)

        # 关闭权重：等权(0.5,0.5) × 相同 cross 分 → 平局，保持输入序(guideline 在前)
        off = r.rerank("冠心病人能运动吗", [guideline, plain], [],
                       top_k=3, use_authority_weight=False)
        assert off[0]["rerank_score"] == off[1]["rerank_score"]

        # 开启权重：guideline(Tier1≈1.08x) 高于 plain(Tier4≈0.90x)
        on = r.rerank("冠心病人能运动吗",
                      [_make_item(content=GUIDELINE_CONTENT, score=0.9),
                       _make_item(content=plain_content, score=0.9)],
                      [], top_k=3, use_authority_weight=True)
        assert on[0]["rerank_score"] > on[1]["rerank_score"]
        assert on[0]["authority_tier"] == "tier_1"
        assert on[1]["authority_tier"] == "tier_4"
        # 偏置是小幅的：低档来源不因权威权重重排翻盘强相关结果
        assert on[1]["rerank_score"] > 0.5 * on[0]["rerank_score"]
