"""核心业务包。

刻意不在 `__init__` 里 eager import workflow：`src.session.manager` 只依赖
`src.core.llm`，但 `import src.core.llm` 会先执行本文件的 `__init__`；若这里
再 import workflow，就会把 outcome → session.manager 整条链拉起来，形成
循环 import（症状是 "partially initialized module"）。

`from src.core import MedicalRAGWorkflow` 仍可用——走 PEP 562 惰性属性。
"""

__all__ = ["MedicalRAGWorkflow"]


def __getattr__(name: str):
    if name == "MedicalRAGWorkflow":
        from src.core.workflow import MedicalRAGWorkflow

        return MedicalRAGWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
