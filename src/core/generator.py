# ============================================================
# 答案生成模块（CPU/GPU自适应）
# 来源：项目需求
# GPU: GPTQ INT4 量化 (项目需求)
# CPU: 标准transformers加载 (当前机器适配)
# ============================================================

import time
from loguru import logger
from config.settings import settings
from src.utils.helpers import load_prompt_template
from src.core.model_loader import generate_text as _generate_text


def generate_text(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """统一文本生成接口 (供 classifier / judge / summary / generator 共用)"""
    return _generate_text(prompt, max_tokens, temperature, do_sample)


def classify_with_llm(prompt: str, max_tokens: int = 10) -> str:
    """L2 分类专用轻量调用 ( <100ms)"""
    return generate_text(prompt, max_tokens=max_tokens, temperature=0.0)


class AnswerGenerator:
    """医疗问诊回答生成器 (项目需求)"""

    def generate(
        self,
        query: str,
        context_text: str,
        conversation_context: str = "",
    ):
        prompt_template = load_prompt_template("generation")
        if not prompt_template:
            prompt_template = (
                "你是一个专业的全科医疗问诊助手。请基于下面提供的参考资料，"
                "回答用户的问题。\n\n"
                "## 要求\n"
                "1. 优先使用最新的医学文献内容\n"
                "2. 回答中需要标注信息来源（来自本地知识库还是文献）\n"
                "3. 如果参考资料中没有相关内容，请明确说明\n"
                "4. 不要给出确定性的诊断结论，建议用户就医\n"
                "5. 不要推荐处方药物\n"
                "6. 回答要专业、清晰、易懂\n\n"
                "## 对话历史\n{conversation_context}\n\n"
                "## 参考资料\n{context_text}\n\n"
                "## 用户问题\n{query}\n\n"
                "## 回答"
            )

        prompt = prompt_template.replace("{query}", query)
        prompt = prompt.replace("{context_text}", context_text)
        prompt = prompt.replace("{conversation_context}", conversation_context or "无")

        start = time.perf_counter()
        answer = generate_text(prompt, max_tokens=1024, temperature=0.3)
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(f"回答生成完成: {len(answer)}字, {latency_ms:.0f}ms")
        return answer, latency_ms

    def generate_degraded(self, query: str) -> str:
        prompt_template = load_prompt_template("degradation")
        if not prompt_template:
            prompt_template = (
                "你是一个全科医疗问诊助手。由于当前检索服务暂时不可用，"
                "以下回答仅基于通用医学知识，仅供参考。\n\n"
                "用户问题：{query}\n\n回答："
            )
        prompt = prompt_template.replace("{query}", query)
        answer = generate_text(prompt, max_tokens=512, temperature=0.5)
        return (
            "⚠️ 当前检索工具暂时不可用，以下回答基于通用知识，仅供参考：\n\n"
            + answer
        )


generator = AnswerGenerator()
