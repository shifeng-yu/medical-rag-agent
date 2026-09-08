# ADR-0003: 模型对话收口（LLM 总机）

- 状态：Accepted
- 日期：2026-09-04
- 关联词条：`CONTEXT.md` → 模型对话（LLM 总机）；接口契约注释见 `src/core/llm.py`

## 背景

仓库里有 5 个业务模块要跟大模型对话：答案生成（`generator.generate`）、降级回答（`generate_degraded`）、L2 问题分类（`classifier._llm_classify`）、幻觉裁判（`judge.llm_judge`）、会话摘要（`session.manager._generate_summary`）。发现的实际问题：

1. **同名 `generate_text` 两个入口**：`model_loader.py` 是真实实现，`generator.py` 是转发壳（注释自称"供共用"，实际大家各自调）；judge / session 为了躲导入环还在函数体内延迟 import。
2. **采样参数散落 5 处**：max_tokens 10/256/1024/512、temperature 0.0/0.1/0.3/0.5 由每个调用点自己拍脑袋传。
3. **temperature 一直在空转（潜伏 bug）**：`generate_text` 默认 `do_sample=False`，transformers 中 temperature 只在 do_sample=True 时生效——generation 的 0.3、degraded 的 0.5 **从未真正生效**，实际一直是贪婪解码。传参像有效，其实无效，误导后人。
4. **tokenizer 自加载 3 处**：model_loader（生成）、helpers（计数）、chunker（分块），各写一份 `AutoTokenizer.from_pretrained`（transformers 进程缓存保证实际只装一次，重复的是代码不是内存）。
5. **内置兜底 prompt 散 5 处**（模板文件缺失时的字符串各模块各存一份）。
6. **测试只能 monkeypatch**：要"不烧 GPU 跑链路"只能逐节点打补丁，没有统一的假后端接缝。

## 决策

1. **总机唯一入口**：新增 `src/core/llm.py`，业务模块只声明意图 `get_llm().complete("answer", prompt)`，禁止再直接 import transformers / 自定采样参数。
2. **任务参数表（TASKS）**：5 个任务（classify / judge / summary / answer / degraded）的 max_tokens / temperature / do_sample 在此单一来源。`complete(task, prompt, **overrides)` 支持按次覆盖（白名单字段，未知覆盖直接 TypeError，fail fast）。
3. **采样诚实化（行为不变）**：所有任务显式 `do_sample=False`（贪婪解码），温度如实记 0.0。历史 0.3/0.5 因从未生效而"收编时归零"——**线上输出不变**，只是删掉了空转参数。将来要采样多样性，改 TASKS 对应任务 do_sample=True 即可，无需动调用方。
4. **后端可替换 + 契约测试**：`QwenBackend`（真实，收编 model_loader 的 load_qwen/generate_text，懒加载至首次调用）与 `FakeLLM`（确定性应答、不加载模型）遵循同一 `generate(task, prompt, *, max_tokens, temperature, do_sample)` 约定；**不强制继承**，接口一致性由 `tests/test_llm.py` 锁定（签名逐参比对 + FakeLLM 行为 + complete 分发断言）。原 `model_loader.py` 删除。
5. **后端选择走配置**：`settings.llm_backend`（env `LLM_BACKEND`，`qwen` 默认 / `fake`），import 期决定并缓存单例——与检索门卫 `get_retriever()` 同一风格。`LLM_BACKEND=fake` 后整条问诊链路不依赖 Qwen 即可跑通（CI / 演示 / 无 GPU 环境）。
6. **tokenizer 收口边界**：生成侧 tokenizer 随 QwenBackend 收编；计数侧（helpers / chunker）保留各自 fallback 语义——它们必须在模型缺失时仍能工作（核心路径不可被模型缺失阻塞），且位于基础层，不宜反向依赖 core。三处实际共享 transformers 进程级缓存，内存仍只装一次。

## 后果

- 正面：采样参数一处管；temperature 空转 bug 从根上消除（代码不再撒谎）；假话务员解锁"不烧 GPU 跑全链路"；删掉双入口与函数内延迟 import（无环可顶层导入）；新增 20 个契约/行为测试。
- 代价：`LLM_BACKEND` 是 import 期决策，切换需重启（与检索后端一致，有意保留）；计数侧 tokenizer 仍三份代码（理由见决策 6）。
- 未来评审如需推翻（例如接入 OpenAI 兼容 API 作后端、或开启采样），先指出本 ADR 再讨论。
