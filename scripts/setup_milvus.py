# ============================================================
# 检索库强制重建管理命令
#
# 用法（显式 --force，默认不删除任何数据）：
#   python scripts/setup_milvus.py            # 缺表则建，已有表不动（非破坏）
#   python scripts/setup_milvus.py --force    # 删除并重建两个 collection
#
# 建表 schema 与索引逻辑已收敛进适配器（src/core/retriever.py 的
# ensure_collections / _create_collection），本脚本只负责调用，
# 不再各自维护一套字段定义。此命令面向 Docker Milvus；
# Milvus Lite 模式 connect 时自动建库，如需重建请删除 milvus_lite 目录。
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from config.settings import settings


def ensure_collections(force: bool = False):
    """确保检索库 schema 存在（--force 时删除重建）"""
    from src.core.retrieval import get_retriever

    if settings.use_milvus_lite:
        logger.error(
            "setup_milvus 面向 Docker Milvus。当前为 Milvus Lite 模式："
            "connect 时自动建库，如需重建请删除 milvus_lite 目录后重启服务。"
        )
        sys.exit(1)

    retriever = get_retriever()
    retriever.connect()
    retriever.ensure_collections(force=force)
    logger.info(
        f"检索库就绪 (force={force}): "
        f"{settings.milvus_collection_kb} / {settings.milvus_collection_pubmed}"
    )


if __name__ == "__main__":
    force = "--force" in sys.argv[1:]
    ensure_collections(force=force)
