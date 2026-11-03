"""测试环境的可选依赖检查。

背景：仓库根目录有个 `milvus_lite/` 数据目录。未安装真正的 milvus-lite 包时，
Python 会把这个同名目录当成命名空间包，`pytest.importorskip("milvus_lite")`
会"成功"，随后 `from milvus_lite import MilvusLite` 才报错——本地看起来像
8 个莫名其妙的失败。这里同时检查顶层符号，缺包就干净跳过。
"""

import pytest


def require_milvus_lite():
    module = pytest.importorskip("milvus_lite")
    if not hasattr(module, "MilvusLite"):
        pytest.skip(
            "milvus-lite 未安装（根目录 milvus_lite/ 是本地数据目录，不是 Python 包）"
        )
    return module
