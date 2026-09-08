# ============================================================
# 医疗Q&A自定义分块模块
# 三大优化：
#   1. Token阈值调整: 512→800, min→100 (适配长问诊)
#   2. 医疗边界正则: 患者/医生/主诉/现病史等专属前缀
#   3. 超长QA二级拆分: 按语义模块拆分 + 带原问题前缀
# ============================================================

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from loguru import logger


# ---- 医疗专属边界识别规则 ----
# 原生Q&A分块靠 "?/问:/答:" 通用标识 → 补充医疗场景专属正则

MEDICAL_BOUNDARY_PATTERNS = [
    # 角色标识 
    re.compile(r"(?:患者|病人|用户|家属)[：:]\s*"),
    re.compile(r"(?:医生|医师|大夫|专家)[：:]\s*"),
    # 病历结构关键词 
    re.compile(r"(?:主诉|现病史|既往史|家族史|个人史)[：:]\s*"),
    re.compile(r"(?:诊断|鉴别诊断|初步诊断|最终诊断)[：:]\s*"),
    re.compile(r"(?:用药|处方|医嘱|治疗方案|用药指导)[：:]\s*"),
    re.compile(r"(?:检查|检验|辅助检查|体格检查)[：:]\s*"),
    re.compile(r"(?:注意事项|随访建议|健康指导|康复建议)[：:]\s*"),
    # 问答标识
    re.compile(r"(?:问题|提问|询问|咨询)[：:]\s*"),
    re.compile(r"(?:回答|答复|建议|意见)[：:]\s*"),
    # 专科前缀
    re.compile(r"(?:症状|体征|表现|感觉)[：:]\s*"),
    re.compile(r"(?:病因|病理|发病机制)[：:]\s*"),
]

# ---- 语义模块拆分正则 (二级拆分用) ----
SEMANTIC_MODULE_PATTERNS = {
    "症状描述": re.compile(r"(?:症状|表现|感觉|主诉|不舒服)[：:是]"),
    "诊断结论": re.compile(r"(?:诊断|判断|考虑|可能性|结论)[：:是为]"),
    "用药指导": re.compile(r"(?:用药|药物|服药|药品|用药方案|处方)[：:是]"),
    "注意事项": re.compile(r"(?:注意|避免|禁忌|不宜|建议|随访)[：:是]"),
    "检查建议": re.compile(r"(?:检查|检验|检测|筛查|影像)[：:是]"),
}


@dataclass
class ChunkInfo:
    """分块结果信息"""
    text: str
    tokens: int
    source_prefix: str = ""  # 超长QA二级拆分时的原问题前缀
    metadata: Dict = field(default_factory=dict)


class MedicalQAChunker:
    """
    医疗Q&A专用分块器 
    复现 RAGFlow 自定义分块插件逻辑
    """

    def __init__(
        self,
        max_tokens: int = 800,      # 从512调整到800
        min_tokens: int = 100,       # 最小token阈值
        overlap_tokens: int = 50,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = None
        logger.info(
            f"MedicalQAChunker 已初始化: max={max_tokens}, min={min_tokens}"
        )

    @property
    def tokenizer(self):
        """懒加载 BGE-M3 tokenizer ( 基于BGE-M3 tokenizer适配阈值)"""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                from config.settings import settings
                self._tokenizer = AutoTokenizer.from_pretrained(
                    settings.bge_model_path
                )
            except Exception:
                logger.warning("BGE-M3 tokenizer加载失败，使用字符估算")
                self._tokenizer = "fallback"
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """基于 BGE-M3 tokenizer 计数"""
        if self._tokenizer == "fallback" or self._tokenizer is None:
            return len(text) // 1.5
        return len(self._tokenizer.encode(text))

    # ========== 核心分块方法 ==========

    def chunk(self, text: str, source_metadata: Optional[Dict] = None) -> List[ChunkInfo]:
        """
        医疗QA文档分块主入口
        流程: 整体文本 → QA边界切分 → token检查 → 超长二级拆分
        """
        if source_metadata is None:
            source_metadata = {}

        # Step 1: 按QA边界切分
        qa_pairs = self._split_by_qa_boundary(text)

        # Step 2: 对每个QA对做token检查与二级拆分
        chunks = []
        for qa_text, qa_meta in qa_pairs:
            token_count = self.count_tokens(qa_text)

            if token_count > self.max_tokens:
                # 超长QA → 二级语义模块拆分，带原问题前缀
                sub_chunks = self._semantic_split(
                    qa_text,
                    source_prefix=qa_meta.get("question", ""),
                    source_metadata=source_metadata,
                )
                chunks.extend(sub_chunks)
                logger.debug(f"二级拆分: {len(sub_chunks)} 个子块 (原token={token_count})")
            else:
                chunks.append(ChunkInfo(
                    text=qa_text,
                    tokens=token_count,
                    metadata={**source_metadata, **qa_meta},
                ))

        # Step 3: 合并过短的块 ( 避免短问答被过度拆分)
        chunks = self._merge_short_chunks(chunks)

        logger.info(f"分块完成: 输入长度={len(text)}, 输出块数={len(chunks)}")
        return chunks

    def _split_by_qa_boundary(self, text: str) -> List[tuple]:
        """
         医疗QA边界识别
        优先级: 医疗专属前缀 > 通用QA标识 > 自然段
        """
        # 构建完整边界正则 (医疗专属 + 通用)
        all_patterns = MEDICAL_BOUNDARY_PATTERNS + [
            re.compile(r"\n(?:Q[：:?？]|\d+[.、]?\s*)"),
            re.compile(r"\n(?:A[：:]?|答[：:]?)"),
        ]

        # 查找所有边界位置
        boundaries = [0]
        for pattern in all_patterns:
            for m in pattern.finditer(text):
                pos = m.start()
                if pos not in boundaries:
                    boundaries.append(pos)
        boundaries = sorted(set(boundaries))
        boundaries.append(len(text))

        # 按边界切分
        pairs = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = text[start:end].strip()
            if len(segment) < 5:  # 跳过过短片段
                continue

            # 提取问题前缀 ( 超长拆分时携带)
            question = self._extract_question_prefix(segment)

            pairs.append((segment, {"question": question}))

        return pairs

    def _semantic_split(
        self,
        text: str,
        source_prefix: str = "",
        source_metadata: Optional[Dict] = None,
    ) -> List[ChunkInfo]:
        """
         超长QA二级拆分
        按语义模块(症状/诊断/用药/注意事项/检查)拆分
        每个子块携带原问题前缀与来源元数据（title/department/doc_id 等）
        """
        if source_metadata is None:
            source_metadata = {}

        # 找语义模块边界
        boundaries = [0]
        for name, pattern in SEMANTIC_MODULE_PATTERNS.items():
            for m in pattern.finditer(text):
                boundaries.append(m.start())
        boundaries = sorted(set(boundaries))
        boundaries.append(len(text))

        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = text[start:end].strip()
            if not segment:
                continue

            token_count = self.count_tokens(segment)

            # 带上原问题前缀 ( 保证语义完整)
            if source_prefix and not segment.startswith(source_prefix):
                full_text = f"[问题：{source_prefix}]\n{segment}"
            else:
                full_text = segment

            # 如果带前缀后仍然超长，按token硬截断 ( 避免无限拆分)
            if self.count_tokens(full_text) > self.max_tokens * 1.2:
                # 按 token 做滑动窗口截断
                sub_segments = self._token_window_split(full_text, source_prefix)
                for sub_text in sub_segments:
                    chunks.append(ChunkInfo(
                        text=sub_text,
                        tokens=self.count_tokens(sub_text),
                        source_prefix=source_prefix,
                        metadata=dict(source_metadata),
                    ))
            else:
                chunks.append(ChunkInfo(
                    text=full_text,
                    tokens=token_count,
                    source_prefix=source_prefix,
                    metadata=dict(source_metadata),
                ))

        return chunks if chunks else [
            ChunkInfo(
                text=text,
                tokens=self.count_tokens(text),
                metadata=dict(source_metadata),
            )
        ]

    def _token_window_split(self, text: str, prefix: str = "") -> List[str]:
        """按 token 滑动窗口分割长文本"""
        words = list(text)
        step = int(self.max_tokens * 1.5)
        window = int(self.max_tokens * 1.5)
        segments = []
        for i in range(0, len(words), step - int(self.overlap_tokens * 1.5)):
            chunk = "".join(words[i: i + window])
            if prefix and not chunk.startswith(prefix):
                chunk = f"[{prefix}]\n{chunk}"
            segments.append(chunk)
        return segments

    def _merge_short_chunks(self, chunks: List[ChunkInfo]) -> List[ChunkInfo]:
        """
        合并过短的块 ( 避免短问答被过度拆分)
        小于 min_tokens 的块尝试与前一个合并
        """
        if not chunks:
            return chunks

        merged = []
        buffer = chunks[0]

        for chunk in chunks[1:]:
            if buffer.tokens < self.min_tokens:
                # 合并：短块追加到前一个
                buffer.text += "\n" + chunk.text
                buffer.tokens += chunk.tokens
                buffer.metadata.update(chunk.metadata)
            else:
                merged.append(buffer)
                buffer = chunk

        merged.append(buffer)
        return merged

    def _extract_question_prefix(self, text: str) -> str:
        """从QA块中提取问题前缀 ( 超长拆分时携带)"""
        # 尝试匹配 "患者：xxx" 或 "问题：xxx" 等模式
        q_pattern = re.compile(
            r"(?:(?:患者|病人|用户|问题|提问|询问)\s*[：:]\s*)([^。？\n]{10,80}[？?])"
        )
        match = q_pattern.search(text)
        if match:
            return match.group(1)
        # 取前80字符作为问题摘要
        return text[:80].replace("\n", " ")

    # ========== 批量处理方法 (RAGFlow插件注册用) ==========

    def batch_chunk(
        self, documents: List[Dict]
    ) -> List[Dict]:
        """
        批量分块接口 - 适配 RAGFlow 文档解析插件格式
        documents: [{"content": "...", "metadata": {...}}, ...]
        返回: [{"text": "...", "tokens": int, "metadata": {...}}, ...]
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(
                text=doc.get("content", ""),
                source_metadata=doc.get("metadata", {}),
            )
            for c in chunks:
                all_chunks.append({
                    "text": c.text,
                    "tokens": c.tokens,
                    "metadata": c.metadata,
                })
        return all_chunks


# 默认分块器实例
default_chunker = MedicalQAChunker()
