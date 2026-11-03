#!/usr/bin/env python
# ============================================================
# 数据库初始化脚本
#
#   python scripts/init_db.py            # 建表（幂等）
#   python scripts/init_db.py --ddl      # 只打印建表 DDL（用于评审/迁移参考）
#   python scripts/init_db.py --drop     # 先删表再建（仅本地开发，生产禁用）
#
# 生产 compose 里 API 启动时会自动建表；本脚本用于"先建库再放流量"的
# 运维流程，以及给 DBA 出具 MySQL DDL。
# ============================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.schema import CreateTable

from config.settings import settings
from src.storage.db import get_engine, init_schema
from src.storage.models import Base


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="初始化 medical-rag 数据库表结构")
    parser.add_argument(
        "--ddl", action="store_true", help="只打印建表 DDL，不执行"
    )
    parser.add_argument(
        "--drop", action="store_true", help="先 drop_all 再 create_all（仅本地）"
    )
    return parser


def main(argv=None) -> int:
    args = get_parser().parse_args(argv)
    engine = get_engine()

    if args.ddl:
        dialect = engine.dialect
        for table in Base.metadata.sorted_tables:
            print(str(CreateTable(table).compile(dialect=dialect)).strip() + ";\n")
        return 0

    if args.drop:
        Base.metadata.drop_all(engine)
        print("已删除全部表（drop_all）")

    init_schema()
    tables = ", ".join(Base.metadata.tables)
    print(f"OK 数据库就绪: {settings.database_url}")
    print(f"   tables: {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
