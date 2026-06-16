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

- **[FE-LLM阶段总结.md](FE-LLM阶段总结.md)** — 【收口·先读这份】当前阶段句号：愿景/已验证成果（带证据）/belief 在真实数据(CrossWOZ)的价值地图/底座线与分层预测编码世界模型的封存定论/蓝图差距/方法论沉淀/复现。125 回归测试。
- **[FE-LLM核心引擎构想.md](FE-LLM核心引擎构想.md)** — 【下一步研究方向】方案一 CAPCW（内容寻址预测编码工作空间）：借鉴 Transformer 真正力量（内容寻址路由），把自由能/预测误差从失败的"固定纵向分层"搬到"内容寻址横向 slot 工作空间"+ 可溯源 + 可生长，保持六要素；含理论连接（attention 即推理 / slot-attention）、风险与组合泛化最小验证方案。
- **[FE-LLM从0自建v2架构设计.md](FE-LLM从0自建v2架构设计.md)** — 【主线·部分修订】完全从 0、不依赖任何预训练底座的认知架构蓝图。注：其核心引擎"分层预测编码（z_1..z_L 纵向）"已经组合泛化裁决判为本规模过度设计、封存（见阶段总结）；新核心引擎方向见上面的核心引擎构想。
- **[FE-LLM蓝图完成度.md](FE-LLM蓝图完成度.md)** — 【完成度】道易循环 / 规范架构 11 模块 / 六要素 → 已实现模块 → 验证证据 的逐条对齐；含诚实边界与复现入口。125 回归测试守护，全部 eval 可复跑。
- **[FE-LLM术语表.md](FE-LLM术语表.md)** — 【术语】白话解释 controller / headroom / taxonomy / belief / surprise / 自由能 / 槽位 / 能量解码 等所有名词，附在 FE-LLM 里指什么、在哪个文件。看不懂名词先看这份。
- **[端到端 Demo 实录](reports/fe_llm_demo_transcript.md)** — FE-LLM 核心闭环的 9 轮会话演示：信息不足→追问、风险→拒答、记住上下文→直接回答、稳定偏好→成长读回，每轮含显式 surprise 通道与 belief 槽位（可溯源）。运行 `python -m fe_llm.active_inference.experiments.fe_llm_demo --run`。
- **[端到端 Demo 网页可视化](reports/fe_llm_demo.html)** — 同一 9 轮会话的自包含 HTML（浏览器直接打开）：动作色标徽章、surprise 各通道条形、belief 槽位 chips、召回与回答。生成 `python -m fe_llm.active_inference.experiments.fe_llm_demo_web --run`。交互 CLI：`python -m fe_llm.active_inference.experiments.fe_llm_cli`。**交互网页**（零依赖，浏览器实时对话）：`python -m fe_llm.active_inference.experiments.fe_llm_web_server` 后打开 http://127.0.0.1:8000 。
- **[FE-LLM意图序列化架构草案.md](FE-LLM意图序列化架构草案.md)** — 由 opus-100 翻译泛化实验（word-F1 0.07）触发的架构升级草案：意图从单向量升级为 `global_intent + intent_slots + slot_salience`（道易方案"卦/爻/精度"的工程化转译），含与核心思想的逐条一致性审查与 M1-M4 判定实验路线。**M2 判定已完成（2026-06-11）：FAIL**（word-F1 0.083，仅 1.13x 基线），瓶颈锁定为小模型从零训练的容量而非意图结构，按预案转线。
- **[FE-LLM预训练底座路线草案.md](FE-LLM预训练底座路线草案.md)** — M2 阴性结果触发的转线设计：预训练底座（容量）+ FE-LLM 机制层（结构化意图、能量解码、主动推理、可溯源）。三级接入路线 P1（冻结底座+外挂意图/能量头）→ P2（LoRA 意图条件注入）→ P3（蒸馏回自有 PER 架构），N1-N4 判定里程碑，强制同底座无机制对照组防止贡献归属混淆。
- **[FE-LLM预训练底座N1执行规格.md](FE-LLM预训练底座N1执行规格.md)** — P1 首轮执行规格：冻结底座、外挂 `IntentAdapter` 与 `EnergyHead`、hybrid 打分、A/B/C 对照组与 N1 判定口径。
- **[FE-LLM预训练底座N1选型.md](FE-LLM预训练底座N1选型.md)** — N1 首发底座选型：选择 `Qwen/Qwen2.5-0.5B`，并记录 Qwen3、SmolLM2、TinyLlama 的暂缓理由。
- **[FE-LLM预训练底座N1阶段小结.md](FE-LLM预训练底座N1阶段小结.md)** — 汇总 P1/P1.5/P2 最小外挂路线的实验矩阵、阶段判定与下一步 P2b 建议。
- **[FE-LLM预训练底座N1下一阶段决策.md](FE-LLM预训练底座N1下一阶段决策.md)** — 在进入更重 LoRA/hook 前，先诊断 `IntentAdapter` 表达是否本身足够分离。
- **[FE-LLM预训练底座P1.5执行方案.md](FE-LLM预训练底座P1.5执行方案.md)** — P1 rerank 阴性/部分阳性后进入 P1.5：`IntentLogitsAdapter` 让 structured intent 直接生成候选 logit bias，再用同一 A/B/C 口径复测。
- **[FE-LLM预训练底座P2执行方案.md](FE-LLM预训练底座P2执行方案.md)** — P1/P1.5 正式评估未过后进入 P2：`IntentResidualAdapter` 让 intent 进入 hidden state 更新，再经底座 `lm_head` 读出 logits。

## 一句话总结

FE-LLM v1 的研究重点不是和通用 LLM 比语言流畅性，而是证明一件更基础的事：语言系统可以先显式感到惊奇、更新 belief、选择 action，再生成文本，并把这条路径完整记录下来。
