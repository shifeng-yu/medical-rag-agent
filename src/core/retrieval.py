# ============================================================
# 检索知识库门面（唯一选择逻辑）
#
# 项目里有两套"双源检索"适配器，同一接口、不同后端：
#   - src/core/retriever.py      → Docker Milvus（pymilvus，生产）
#   - src/core/retriever_lite.py → Milvus Lite（嵌入式，本地开发/CI）
#
# 历史问题：选后端的 if/else 复制在 workflow/ablation/routes/main 四处，
# main 启动还无视配置；两套适配器已漂移（打分指标、both 并行度、insert 缺失）。
#
# 本模块是唯一收口点：
#   - get_retriever()：按 settings.use_milvus_lite 选择并缓存单例（import 期选择）
#   - expand_query()：两个适配器共用的查询扩展实现，只此一份
#
# 决策与接口契约见 docs/adr/0002-retrieval-seam.md；领域词条见 CONTEXT.md。
# ============================================================

from loguru import logger
from config.settings import settings


# ---------------- 适配器选择（import 期决定，缓存单例） ----------------

_retriever = None


def get_retriever():
    """返回按配置选定的双源检索适配器。

    选择逻辑只存在于此处：workflow / ablation / routes / main / scripts
    一律从本函数取适配器，禁止再写 if use_milvus_lite 分支。
    """
    global _retriever
    if _retriever is None:
        if settings.use_milvus_lite:
            from src.core.retriever_lite import lite_retriever
            _retriever = lite_retriever
            logger.info("检索后端: Milvus Lite（嵌入式）")
        else:
            from src.core.retriever import retriever
            _retriever = retriever
            logger.info("检索后端: Docker Milvus (pymilvus)")
    return _retriever


# ---------------- 适配器契约（两个适配器共同遵守） ----------------
#
# 接口: connect / retrieve / insert / expand_query / encode_query / embedder
# 结果: retrieve 返回 (local_results, pubmed_results, latency_ms)，
#       每条结果 dict 含 content/score/source/department/publish_time/title/doc_id
# 分数: 统一 IP 指标（向量已归一化 → score≈余弦相似度），越大越相关、降序返回
# 写入: insert(collection_name, chunks, batch_size)，chunk 为
#       {"text": str, "metadata": {title, department, publish_time, source_type, doc_id}}

# ---------------- 查询扩展（共享实现） ----------------

_STOPWORDS = {"的", "了", "是", "我", "你", "吗", "呢", "啊", "吧", "什么", "怎么"}


def expand_query(query: str) -> str:
    """检索失败重试时的查询扩展：去掉停用词、保留核心实体。"""
    words = [w for w in query if w not in _STOPWORDS]
    return "".join(words) if words else query


def merge_dual_results(
    main_local: list,
    main_pubmed: list,
    extra_local: list,
    extra_pubmed: list,
) -> tuple:
    """合并主检索与补充检索（术语标准化扩词式接入）结果。

    语义：主检索的 query 保持原样（不丢口语原意）；当术语标准化
    （normalize_query）产生不同查询词时，用标准化词再检一次并把结果并入。
    - 去重键 (source, doc_id, content)：完全相同的分块不重复进上下文；
      同一文档的不同分块保留，交给重排粗排截断收敛。
    - 双源各自独立合并，保持 score 降序原序（主结果在前、补充追加在后）。
    """
    def _merged(main_items: list, extra_items: list) -> list:
        seen = {
            (r.get("source"), r.get("doc_id"), r.get("content"))
            for r in main_items
        }
        merged = list(main_items)
        for r in extra_items:
            key = (r.get("source"), r.get("doc_id"), r.get("content"))
            if key not in seen:
                merged.append(r)
                seen.add(key)
        return merged

    return (
        _merged(main_local, extra_local),
        _merged(main_pubmed, extra_pubmed),
    )
