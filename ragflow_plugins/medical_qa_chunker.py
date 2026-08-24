# ============================================================
# RAGFlow 自定义分块插件
# 将本模块注册到 RAGFlow 的分块解析器扩展点
# ============================================================

"""
RAGFlow 自定义分块插件注册指南 

RAGFlow 的文档解析模块支持可插拔扩展，通过编写自定义分块解析函数并注册实现。

使用方法:
1. 将此文件放置在 RAGFlow 部署目录的 `rag/app/chunking/` 下
2. 在 RAGFlow 配置中启用自定义分块策略
3. 在知识库设置中选择「Medical QA Chunker」作为分块方法

或在独立部署模式下，直接调用 MedicalQAChunker 类处理文档。
"""

import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.medical_qa_chunker import MedicalQAChunker


# ---- RAGFlow 插件入口 ----

class RAGFlowMedicalQAChunker:
    """
    RAGFlow 可插拔自定义分块解析器 
    符合 RAGFlow 分块解析器扩展接口规范
    """

    name = "Medical QA Chunker"
    description = "医疗Q&A专用分块 - 阈值定制 + 医疗边界识别 + 超长QA二级拆分"

    def __init__(self, **kwargs):
        max_tokens = kwargs.get("max_tokens", 800)      # 512→800
        min_tokens = kwargs.get("min_tokens", 100)       # 最小阈值
        self.chunker = MedicalQAChunker(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )

    def chunk(self, text: str, **kwargs) -> list:
        """
        分块主入口 - RAGFlow 标准接口
        返回: [{"text": "...", "tokens": int, "metadata": {...}}, ...]
        """
        source_metadata = kwargs.get("metadata", {})
        chunks = self.chunker.chunk(text, source_metadata=source_metadata)
        return [
            {
                "text": c.text,
                "tokens": c.tokens,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

    def batch_chunk(self, documents: list, **kwargs) -> list:
        """
        批量分块 - RAGFlow 标准接口
        documents: [{"content": "...", "metadata": {...}}, ...]
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(
                text=doc.get("content", ""),
                source_metadata=doc.get("metadata", {}),
            )
            all_chunks.extend([
                {"text": c.text, "tokens": c.tokens, "metadata": c.metadata}
                for c in chunks
            ])
        return all_chunks


# ---- 独立使用示例 ----

if __name__ == "__main__":
    # 测试分块效果
    chunker = MedicalQAChunker()

    sample_text = (
        "患者：我头疼一周了，怎么办？\n"
        "医生：头疼的性质是什么？胀痛还是刺痛？\n"
        "患者：主要是胀痛，早上起床的时候最明显。\n"
        "医生：根据您的描述，可能是紧张性头痛。\n"
        "诊断：紧张性头痛\n"
        "用药指导：建议规律作息，避免长时间看手机电脑。\n"
        "如果疼痛严重，可以短暂使用布洛芬缓解，但不建议长期服用。\n"
        "注意事项：如果出现呕吐、视力模糊等症状，请立即就医。"
    )

    chunks = chunker.chunk(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} (tokens={chunk.tokens}) ---")
        print(chunk.text[:200])
        if chunk.source_prefix:
            print(f"[来源问题: {chunk.source_prefix}]")
