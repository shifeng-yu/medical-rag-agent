# ============================================================
# LLM 对话总机（模型对话收口）
#
# 仓库里所有"与大模型对话"的动作都经由此模块：
#   - 业务模块只声明意图：get_llm().complete("answer", prompt)
#   - 采样参数（max_tokens / temperature / do_sample）按任务收敛在
#     TASKS 参数表，一处管理，不再散落各调用点
#   - 后端可替换：真实后端 QwenBackend（职责收编自已删除的 model_loader.py）
#     与假后端 FakeLLM（确定性应答、不加载模型，供测试/CI/演示）
#   - 后端选择：settings.llm_backend（qwen | fake），import 期决定，
#     与检索门卫（retrieval.get_retriever）同一风格
#
# 背景与决策见 docs/adr/0003-llm-service.md；领域词条见 CONTEXT.md。
# ============================================================

from typing import Dict, Optional
from loguru import logger
from config.settings import settings


# ==================== 任务参数表 ====================
#
# 每个业务任务默认采样参数的唯一来源。
# 注意 transformers 语义：do_sample=False（贪婪解码）时 temperature 不生效。
# 历史参数 0.3 / 0.5 因调用方从未传 do_sample=True 而一直空转；收口后统一
# 显式贪婪解码（行为不变），温度字段如实记为 0.0。若要采样多样性，
# 将来把对应任务 do_sample 改为 True 并填温度即可，无需改动任何调用方。
TASKS: Dict[str, Dict] = {
    "classify": {"max_tokens": 10,   "temperature": 0.0, "do_sample": False},  # L2 分类兜底
    "judge":    {"max_tokens": 256,  "temperature": 0.0, "do_sample": False},  # 幻觉三维打分
    "summary":  {"max_tokens": 256,  "temperature": 0.0, "do_sample": False},  # 会话摘要
    "answer":   {"max_tokens": 1024, "temperature": 0.0, "do_sample": False},  # 主回答
    "degraded": {"max_tokens": 512,  "temperature": 0.0, "do_sample": False},  # 降级回答
}


# ==================== 后端 ====================
#
# 后端约定（不强制继承，接口一致性由 tests/test_llm.py 契约测试锁死）：
#   generate(task: str, prompt: str, *, max_tokens: int,
#            temperature: float, do_sample: bool) -> str


class QwenBackend:
    """真实后端：Qwen 模型（GPU: GPTQ INT4 / CPU: 标准 transformers）。

    逻辑收编自已删除的 model_loader.py（load_qwen + generate_text），
    该历史文件已不存在，加载职责以本类为唯一实现。
    模型懒加载：首次 generate 才 import torch/transformers 并装载。
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        """懒加载 tokenizer + model（自动 CPU/GPU 模式），进程内只装一次。"""
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_path = settings.qwen_model_path
        logger.info(f"加载 Qwen 模型: {model_path}")
        logger.info(f"设备: {settings.device}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if settings.use_gptq:
            # GPU 模式: GPTQ INT4
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU 模式: 标准加载，使用 fp32
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info(f"模型加载完成: {params:.1f}B 参数, 设备={settings.device}")
        self._tokenizer, self._model = tokenizer, model

    def generate(
        self,
        task: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> str:
        import torch

        self._ensure_loaded()
        tokenizer, model = self._tokenizer, self._model

        inputs = tokenizer(prompt, return_tensors="pt")
        # CPU 模式不传 device
        if settings.device == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        # do_sample=False：贪婪解码，不传 temperature（否则会空转/误导）

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated.strip()


class FakeLLM:
    """假后端：确定性应答，不加载任何模型。

    用途：settings.llm_backend="fake" 时，整条问诊链路不依赖 Qwen 即可
    跑通（CI / 演示 / 无 GPU 环境）。按任务返回合理默认：
      - classify → "本地库"（L2 分类兜底走本地）
      - judge    → 三维高分 JSON（校验放行）
      - summary  → 简短摘要
      - 其余     → 带来源标注的合规医疗答复（可通过规则校验）
    可用 responses 定制（键=任务名，值=固定输出），供测试精确控制。

    演示增强: answer 任务会根据 prompt 抽取 query / 检索来源 / 科室
    生成差异化话术，让用户在无 GPU 演示中也能感受到"系统在结合
    知识库回答"；同时严格规避未溯源实体/违规结论/缺来源三大硬伤，
    保证 judge.rule_check 通过。
    """

    DEFAULT_ANSWER = (
        "根据本地医疗知识库的资料，针对您的描述建议尽快咨询专业医生，"
        "以获得准确的诊疗建议。[来源：本地知识库]"
    )

    # 停用词：抽 query 关键词时跳过
    _STOPWORDS = frozenset(
        "你我他她它的是了吗呢啊呀哈哦噢嗯了把被给和与及或也"
        "都还又再才就这那这个那个这些那些什么怎么怎样如何"
        "我我们你您他她它他们请帮我能不能会不会请帮我能"
        "请问咨询问询看"
        "日常怎么如何怎样怎么办怎么回事"
        "最新今天今年目前近期"
    )

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self._responses = responses or {}

    def _differential_answer(self, prompt: str) -> str:
        """从 prompt 抽 query / 来源 / 科室, 组装差异化但合规的演示话术.

        硬约束(必须满足, 否则 judge.rule_check 失败):
          1. 不能引入具体疾病名/药品名(会被判"未溯源")
          2. 不能出现"你得了X病""建议服用X片"等违规模式
          3. 必须含"来源"或"知识库"等字样
        """
        import re as _re
        # 1) 抽 query: "## 用户问题\nxxx\n## " 段
        q_match = _re.search(
            r"##\s*用户问题\s*\n(.+?)(?:\n##|\Z)", prompt, _re.S
        )
        query = (q_match.group(1).strip() if q_match else "").strip()
        # 截短 query (避免过长的多轮问题)
        query = query.split("\n")[0][:40]

        # 2) 抽来源类型 & 数量
        # 优先用"参考资料"段里的"N 条"数字, fallback 到关键词出现次数
        n_src = 0
        ctx_match = _re.search(r"##\s*参考资料\s*\n(.+?)(?:\n##|\Z)", prompt, _re.S)
        ctx_section = ctx_match.group(1) if ctx_match else ""
        cnt_match = _re.search(r"(\d+)\s*条", ctx_section)
        if cnt_match:
            n_src = int(cnt_match.group(1))
        has_pubmed = bool(_re.search(r"pubmed|PubMed|文献", ctx_section))
        has_local = bool(_re.search(r"local_kb|本地知识库|medical_kb", ctx_section))
        if has_pubmed and has_local:
            src_label = "本地知识库 + PubMed 文献"
        elif has_pubmed:
            src_label = "PubMed 离线文献"
        else:
            src_label = "本地医疗知识库"
        if n_src == 0:
            n_src = 5  # 兜底

        # 3) 抽 query 关键词(2-3个有信息量的实词)
        # 用相邻名词优先匹配 + 单独字组合, 避免"室性心动过/速"切碎
        words = _re.findall(r"[\u4e00-\u9fa5A-Za-z]{2,8}", query)
        keywords = [w for w in words if w not in self._STOPWORDS][:4]
        # 提取不到时用整句
        if not keywords and query:
            keywords = [query[:8]]

        # 4) 组装话术: 引用 query 关键词 + 来源统计 + 通用科普建议
        kw_part = "、".join(keywords) if keywords else "您咨询的问题"
        # 根据关键词特征粗分类, 选择不同提示骨架(只动表述骨架, 不引入具体实体)
        if any(k in query for k in ("保养", "预防", "管理", "怎么", "如何", "注意")):
            body = (
                f"1. 日常应注意规律作息与均衡饮食，戒烟限酒并适度运动；\n"
                f"2. 定期复查相关指标，出现异常及时就医；\n"
                f"3. 遵医嘱规范用药，切勿自行调整方案。"
            )
        elif any(k in query for k in ("症状", "原因", "是什么", "怎么回事", "为啥")):
            body = (
                f"1. 检索资料显示相关症状可能与多种因素相关，需要结合个人情况评估；\n"
                f"2. 建议先观察伴随特征（持续时间、严重程度、诱因），记录后供医生参考；\n"
                f"3. 如出现明显加重或新发症状，请尽快就医明确诊断。"
            )
        elif any(k in query for k in ("治疗", "用药", "怎么办", "怎么治", "药")):
            body = (
                f"1. 治疗方案需依据具体诊断与个人情况由医生决定；\n"
                f"2. 切勿自行购买或调整处方类药物；\n"
                f"3. 建议携带既往检查报告到正规医院就诊，由医生制定个体化方案。"
            )
        else:
            body = (
                f"1. 检索到相关医学资料，建议结合个人情况参考；\n"
                f"2. 日常管理以规律生活方式为基础；\n"
                f"3. 具体诊疗请咨询专业医生。"
            )

        return (
            f"根据{src_label}的检索结果（命中 {n_src} 条参考资料），"
            f"针对您关于「{kw_part}」的咨询，提供以下科普参考：\n\n"
            f"{body}\n\n"
            f"本回答仅作医学科普参考，不构成诊疗建议。"
            f"如需进一步评估，请咨询专业医生。\n"
            f"[来源：{src_label}]"
        )

    def generate(
        self,
        task: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> str:
        if task in self._responses:
            return self._responses[task]
        if task == "classify":
            return "本地库"
        if task == "judge":
            return (
                '{"factual_consistency": 9, "logical_coherence": 9, '
                '"answer_helpfulness": 9, "feedback": "FakeLLM 校验通过"}'
            )
        if task == "summary":
            return "用户咨询了症状与应对建议，已建议及时就医。"
        if task == "answer":
            return self._differential_answer(prompt)
        if task == "degraded":
            # 检索不可用时的降级答复, 保持简短
            return (
                "⚠️ 检索工具暂时不可用，以下为基于通用医学知识的科普参考：\n\n"
                "建议您前往正规医疗机构就诊，由专业医生进行评估与指导。"
                "本系统无法替代面诊。[来源：通用医学知识]"
            )
        return self.DEFAULT_ANSWER


# ==================== 总机 ====================


class LLMService:
    """总机：任务参数表 + 后端分发。业务模块只调 complete(task, prompt)。"""

    def __init__(self, backend):
        self._backend = backend

    @property
    def backend(self):
        return self._backend

    def complete(self, task: str, prompt: str, **overrides) -> str:
        if task not in TASKS:
            raise ValueError(
                f"未知 LLM 任务: {task!r}（可用: {sorted(TASKS)}）"
            )
        spec = dict(TASKS[task])
        for key, value in overrides.items():
            if key not in spec:
                raise TypeError(f"任务 {task!r} 不支持覆盖参数 {key!r}")
            spec[key] = value
        return self._backend.generate(
            task, prompt,
            max_tokens=spec["max_tokens"],
            temperature=spec["temperature"],
            do_sample=spec["do_sample"],
        )


_llm_instance: Optional[LLMService] = None


def get_llm() -> LLMService:
    """按 settings.llm_backend 选择后端并构造总机（import 期决定，缓存单例）。"""
    global _llm_instance
    if _llm_instance is None:
        backend_name = (settings.llm_backend or "qwen").lower()
        if backend_name == "fake":
            backend = FakeLLM()
            logger.info("LLM 总机使用假后端 FakeLLM（不加载模型）")
        elif backend_name == "qwen":
            backend = QwenBackend()
            logger.info("LLM 总机使用真实后端 QwenBackend")
        else:
            raise ValueError(
                f"未知 LLM 后端: {settings.llm_backend!r}（可用: qwen, fake）"
            )
        _llm_instance = LLMService(backend)
    return _llm_instance
