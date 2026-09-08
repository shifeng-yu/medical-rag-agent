# ============================================================
# 术语标准化扩词（merge_dual_results）单元测试
# 覆盖 (normalize_query 接线 · 扩词式接入):
#   1. 完全重复分块去重 (source+doc_id+content 同键)
#   2. 同文档不同分块保留（交重排收敛，不误杀）
#   3. 双源独立合并、顺序保持（主结果在前、补充追加在后）
#   4. 空补充 / 空主结果边界
#   5. normalize_query 触发条件（别名确实发生替换才扩词）
# ============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.retrieval import merge_dual_results
from src.core.knowledge_grader import knowledge_grader


def _chunk(source, doc_id, content, score=0.5):
    return {
        "source": source,
        "doc_id": doc_id,
        "content": content,
        "score": score,
    }


class TestMergeDualResults:
    def test_dedupe_identical_chunk_from_extra(self):
        main_local = [_chunk("local_kb", "cardiology", "内容A", 0.9)]
        extra_local = [_chunk("local_kb", "cardiology", "内容A", 0.7)]  # 同块
        local, pubmed = merge_dual_results(main_local, [], extra_local, [])
        assert len(local) == 1  # 完全重复块被丢弃
        assert local[0]["score"] == 0.9  # 保留主结果

    def test_same_doc_different_chunk_kept(self):
        main_local = [_chunk("local_kb", "cardiology", "内容A", 0.9)]
        extra_local = [_chunk("local_kb", "cardiology", "内容B", 0.8)]
        local, _ = merge_dual_results(main_local, [], extra_local, [])
        assert len(local) == 2  # 同文档不同分块都保留，交重排粗排收敛

    def test_dual_source_merged_independently_and_ordered(self):
        main_local = [_chunk("local_kb", "a", "L1", 0.9)]
        main_pubmed = [_chunk("pubmed", "p1", "P1", 0.9)]
        extra_local = [_chunk("local_kb", "b", "L2", 0.8)]
        extra_pubmed = [_chunk("pubmed", "p2", "P2", 0.8)]
        local, pubmed = merge_dual_results(
            main_local, main_pubmed, extra_local, extra_pubmed
        )
        assert [r["doc_id"] for r in local] == ["a", "b"]  # 主在前、补在后
        assert [r["doc_id"] for r in pubmed] == ["p1", "p2"]

    def test_empty_extra_returns_main_untouched(self):
        main_local = [_chunk("local_kb", "a", "L1", 0.9)]
        local, pubmed = merge_dual_results(main_local, [], [], [])
        assert local == main_local
        assert pubmed == []

    def test_empty_main_with_extra(self):
        local, pubmed = merge_dual_results(
            [], [], [_chunk("local_kb", "b", "L2", 0.8)], []
        )
        assert len(local) == 1
        assert local[0]["doc_id"] == "b"

    def test_cross_source_not_deduped(self):
        # 同 doc_id 但不同 source 属于不同集合，不去重
        main_local = [_chunk("local_kb", "x", "同文", 0.9)]
        extra_pubmed = [_chunk("pubmed", "x", "同文", 0.8)]
        local, pubmed = merge_dual_results(main_local, [], [], extra_pubmed)
        assert len(local) == 1
        assert len(pubmed) == 1

    def test_does_not_mutate_inputs(self):
        main_local = [_chunk("local_kb", "a", "L1", 0.9)]
        extra_local = [_chunk("local_kb", "b", "L2", 0.8)]
        main_snapshot = list(main_local)
        extra_snapshot = list(extra_local)
        merge_dual_results(main_local, [], extra_local, [])
        assert main_local == main_snapshot
        assert extra_local == extra_snapshot


class TestNormalizeExpansionTrigger:
    def test_alias_query_triggers_normalization(self):
        q = "我心梗犯了应该怎么办"
        assert knowledge_grader.normalize_query(q) != q  # 心梗 → 心肌梗死

    def test_plain_query_unchanged_no_extra_search(self):
        q = "高血压患者日常饮食要注意什么"
        assert knowledge_grader.normalize_query(q) == q  # 无别名 → 零额外开销
