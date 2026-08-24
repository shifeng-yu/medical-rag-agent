# ============================================================
# CPU/GPU 统一模型加载器
# GPU模式: GPTQ INT4 (项目需求)
# CPU模式: 标准transformers (当前机器适配)
# 架构代码不变，仅加载方式不同
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from config.settings import settings


# 全局模型缓存
_tokenizer = None
_model = None


def load_qwen():
    """加载 Qwen 模型（自动选择CPU/GPU模式）"""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    model_path = settings.qwen_model_path
    logger.info(f"加载 Qwen 模型: {model_path}")
    logger.info(f"设备: {settings.device}")

    _tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    if settings.use_gptq:
        # GPU模式: GPTQ INT4 (项目需求)
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # CPU模式: 标准加载，使用fp32
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    _model.eval()

    params = sum(p.numel() for p in _model.parameters()) / 1e9
    logger.info(f"模型加载完成: {params:.1f}B 参数, 设备={settings.device}")
    return _tokenizer, _model


def generate_text(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """统一文本生成接口"""
    tokenizer, model = load_qwen()

    inputs = tokenizer(prompt, return_tensors="pt")
    # CPU模式不传device
    if settings.device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()
