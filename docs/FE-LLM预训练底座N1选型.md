# FE-LLM 预训练底座 N1 选型

更新日期：2026-06-13

## 结论

N1 首个底座选择：

```text
Qwen/Qwen2.5-0.5B
```

这是当前最适合 N1 的首发底座：参数量低于 1B、base 版、Apache 2.0、Causal LM、32K 上下文、多语种覆盖中英，并且训练阶段是 pretraining，适合做“冻结底座 + FE 机制层”的干净对照。

## 为什么不首选 Qwen3-0.6B

`Qwen/Qwen3-0.6B` 也满足参数量、许可和多语种能力要求，而且更新、更强。但 N1 的目标不是追最强小模型，而是判定 FE-LLM 机制层是否带来增益。Qwen3-0.6B 的公开说明包含 pretraining & post-training，并引入更强的推理/思考行为，这会把“底座自带行为”混入 action selection 与生成判断。

因此：

- Qwen2.5-0.5B 作为 N1 首发；
- Qwen3-0.6B 作为 N1 通过后的增强复测候选；
- 若 Qwen2.5-0.5B 的 B 组不优于 A/C，不允许直接换 Qwen3 把问题掩盖掉。

## 候选判断

### 首选：Qwen/Qwen2.5-0.5B

- 参数：约 0.49B，低于 N1 的 ≤1B 约束；
- 类型：base causal language model；
- 训练阶段：pretraining；
- 许可：Apache 2.0；
- 上下文：32,768 tokens；
- 语言：多语种，覆盖中文和英文；
- 选择理由：能力足够、口径干净、资源可控，适合做机制归因。

### 备选：Qwen/Qwen3-0.6B

- 参数：约 0.6B；
- 许可：Apache 2.0；
- 上下文：32,768 tokens；
- 语言：多语种范围更大；
- 暂不首选原因：post-training 与思考行为可能污染“FE 机制是否有效”的 N1 判定。

### 暂缓：SmolLM2-360M

- 优点：Apache 2.0、非常轻量；
- 暂缓原因：官方定位主要是英文能力，不适合作为 zh→en 翻译判定的首发底座。

### 暂缓：TinyLlama v1.1 Chinese

- 优点：Apache 2.0、有中文变体；
- 暂缓原因：约 1.1B，超过 N1 的 ≤1B 约束；上下文约 2K，且不是最干净的中英翻译首发选择。

## N1 默认实验配置

```text
backbone_name = "Qwen/Qwen2.5-0.5B"
freeze_backbone = true
max_source_length = 256
max_target_length = 128
top_k_candidates = 8
hybrid_alpha_start = 1.0
```

资源预估：

- BF16/FP16 权重约 1GB 量级；
- P1 只训练 adapter/head，但仍要保留 hidden states、logits 和优化器状态；
- 首轮建议短序列和小 batch 起步，优先确保 A/B/C 对照链路跑通。

## 后续验收

1. A 组：冻结 Qwen2.5-0.5B 原生 greedy/top-k/top-p；
2. B 组：冻结 Qwen2.5-0.5B + `IntentAdapter` + `EnergyHead` + hybrid decode；
3. C 组：冻结 Qwen2.5-0.5B + 随机/打乱 intent 负对照；
4. 报告必须同时给出 word-F1、char-F1、输出多样性、分歧率、残余/覆盖能量轨迹；
5. 只有 B 显著优于 A/C 时，才能主张 FE-LLM 机制层有效。

## 信息来源

- Qwen2.5 官方模型说明与 Hugging Face model card：参数量、base/pretraining、Apache 2.0、32K 上下文、多语种；
- Qwen3 技术报告与 Hugging Face model card：0.6B、Apache 2.0、32K 上下文、pretraining & post-training；
- SmolLM2 model card：Apache 2.0、主要英文；
- TinyLlama 技术报告与 model card：Apache 2.0、1.1B、2K 上下文、中文变体。
