# FE-LLM 文档

本目录是 FE-LLM 的论文、架构说明与配图。当前主线已经从早期“能量坍缩生成器”收敛为：

> **可溯源主动推理语言模型原型**：prompt 作为 observation 进入系统，触发 prediction error 与 surprise；模型先选择能降低 expected free energy 的 action，再把 action 实现为语言输出，并记录完整 inference trace。

## 当前论文

- **[FE-LLM论文.md](FE-LLM论文.md)** — 完整中文论文《FE-LLM：一种可溯源的主动推理语言模型原型》，涵盖：
  - 核心思想：不直接续写 prompt，而是 observation -> belief update -> action selection -> language realization。
  - 数学形式：prediction error、precision-weighted surprise、expected free energy。
  - 架构：`Observation`、`BeliefState`、`PredictionError`、`CandidateAction`、`ExpectedFreeEnergyScore`、`InferenceTrace`。
  - 算法：`ActiveInferenceController.respond(...)` 的完整推理流程。
  - 训练：8k 对话语料作为 `ANSWER` 样本，teacher 并发生成非 `ANSWER` 主动推理样本，小型 `PolicySelector` 使用 class-weighted loss。
  - 验收：问候、模糊请求、时间矛盾、实时信息、安全拒答、记忆候选六类场景。

## 图表

| 图 | 文件 | 说明 |
|----|------|------|
| 图 1 | [active_inference_architecture.svg](figures/active_inference_architecture.svg) | 生成模型、Markov blanket 与 v1 计算图 |
| 图 2 | [active_inference_algorithm.svg](figures/active_inference_algorithm.svg) | expected free energy 策略评分分解 |
| 图 3 | [active_inference_training.svg](figures/active_inference_training.svg) | 数据分布、policy selector 输入与训练目标 |
| 图 4 | [intent_slot_architecture.svg](figures/intent_slot_architecture.svg) | 单向量意图瓶颈（翻译实验实证）与意图序列化（象层级）方案对比 |

## 架构演进文档

- **[FE-LLM意图序列化架构草案.md](FE-LLM意图序列化架构草案.md)** — 由 opus-100 翻译泛化实验（word-F1 0.07）触发的架构升级草案：意图从单向量升级为 `global_intent + intent_slots + slot_salience`（道易方案"卦/爻/精度"的工程化转译），含与核心思想的逐条一致性审查与 M1-M4 判定实验路线。

## 一句话总结

FE-LLM v1 的研究重点不是和通用 LLM 比语言流畅性，而是证明一件更基础的事：语言系统可以先显式感到惊奇、更新 belief、选择 action，再生成文本，并把这条路径完整记录下来。
