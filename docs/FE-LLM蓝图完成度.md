# FE-LLM 蓝图完成度

更新日期：2026-06-16

> 把道易主动推理愿景 / 规范架构方案 / 六要素，逐条对齐到**已实现、已验证、可回归**的代码与证据。
> 原则：诚实——做到的标 ✅ 并给证据，未做到/受限的标明边界，不夸大。
> 名词不懂先看 [FE-LLM术语表.md](FE-LLM术语表.md)（controller / headroom / taxonomy / belief / surprise / 自由能 等白话解释）。

## 0. 一句话结论

FE-LLM 的**主动推理控制闭环**（知道何时不该答 + 为何 + 会成长）已从 0 自建、端到端打通、并在合成任务型数据上证明有强价值；全量 **200 个回归测试**守护。
**核心引擎的"开放问"已有正面答案**：CAPCW（内容寻址预测编码工作空间）借鉴 Transformer 内核（内容寻址路由），在容量受限区间决定性胜单向量、接回真实 controller 活文本主动推理闭环，并已做到**对内多步推理**（复合所有格链式取回 + 可溯源 CoT trace）。详见第 2.5 节、`docs/FE-LLM核心引擎构想.md`(§9-29)、`docs/FE-LLM阶段总结.md`(§5.1)。
"是否只缺规模"的答案：**不是**——开放闲聊 belief 弱(0.655)、合成任务型极强(歧义子集 +0.5)，缺的首先是**对的任务**，其次理解层要学习化（已做）。规模能放大，但前提是任务对、理解学习化。
真实数据修正（B2 系列，2026-06-15）：在真实人标任务对话(CrossWOZ)上，belief 的价值精确定位在**状态/领域追踪与回复内容 grounding**（未明示跟进句 +0.15~+0.19），而非动作类型选择（−0.02，真实数据动作由当前 utterance 决定）。早期"belief 决定动作"的合成强结论部分是构造特性，已诚实修正；belief 机制在真实数据上的价值得到正面坐实，只是在"语境/内容"维而非"动作"维。

## 1. 道易循环 → 实现

道易草案第 7 节的循环，逐段对齐：

| 道易循环段 | 实现 | 证据 |
|---|---|---|
| Prompt enters as Observation | `active_inference/observation.py` `Observation.from_text` + 规则特征 | 110 tests |
| Attention routes evidence | `perception.py` `PerceptionEncoder`（IntentEncoder/hash） | — |
| Predictive Coding 比对先验 | `prediction.py` `Predictor` + `belief_update.py` | 多轮 belief 测试 |
| Surprise 分解（语义/逻辑/任务/风险） | `surprise.py` 多通道 prediction_error | 真实数据验证 test1 |
| Yi 情境编码（象/卦/爻） | 槽位级 belief（`known_slots`）+ 结构化意图 | 槽位测试 |
| Active Inference 评估 bian/policy | `policy.py` + `free_energy.py` + `context_policy.py` | bal_acc 0.929；任务型 1.0 |
| Dao 偏好低力稳定行动 | `_apply_slot_belief` + 低不确定不追问守卫 | EnergyDecoder 测试全绿 |
| Decoder 表达所选行动 | `energy_lm` EnergyDecoder（48k 重训） | held-out decode_loss 1.39 |
| After-state 记录 trace / 成长 | `trace.py` + `memory.py`（候选→确认→蒸馏） | 成长评测 PASS |

## 2. 规范架构方案 11 模块 → 实现状态

| 模块 | 状态 | 实现 / 证据 |
|---|---|---|
| perception | ✅ | `perception.py` + `observation.py`（规则特征 + 学习式 NLU 注入） |
| world_model（分层 latent） | ⛔ 封存 / ✅ CAPCW 取而代之 | 分层 `HierarchicalPredictiveEncoder` 经**组合泛化裁决**判过度设计、正式封存（flat 0.652 ≥ hierarchical 0.637，`compositional_generalization_eval.md`）；**核心引擎换成 CAPCW**（内容寻址横向工作空间，见 2.5 节）——把自由能/预测误差从失败的固定纵向分层搬到 Transformer 已验证的内容寻址结构上，已系统实证。 |
| predictive | ✅ | `prediction.py` `Predictor` + PER 块弛豫 |
| surprise | ✅ | `surprise.py` 多通道；真实数据上语义-surprise 有区分力（test1 PASS） |
| situation（立卦） | ✅ | 槽位级 belief（known_slots/required_slots） |
| policy（变） | ✅ | `policy.py` + `context_policy.py`（学习式，任务型留出 1.0） |
| free_energy（吉凶） | ✅ | `free_energy.py` EFE 分解 + 校准 |
| controller（通） | ✅ | `controller.py` 全闭环；可选接入学习式 NLU/policy |
| decoder（表达） | ✅ | `energy_lm` 能量解码（48k 重训提升） |
| memory（成长） | ✅ | `memory.py` 候选→confirmed→离线蒸馏（回流 held-out 0→1.0） |
| trace（溯源） | ✅ | `trace.py` 显式链路，不靠 attention 权重 |

## 2.5 核心引擎 CAPCW（内容寻址预测编码工作空间）→ 第一个有系统实证的核心引擎

分层世界模型封存后，按"借鉴 Transformer 真正力量=内容寻址路由"提出并验证 CAPCW。详见
`docs/FE-LLM核心引擎构想.md`(§9-29)、`docs/FE-LLM阶段总结.md`(§5.1)。判定全表（均先定任务+判据再写引擎）：

| CAPCW 判定 | 状态 | 证据 |
|---|---|---|
| 内容寻址 slot > 单向量（容量受限绑定） | ✅ PASS（高 K 平均 +0.29） | `capcw_binding_eval.md` |
| 显式自由能/可溯源形态(PCWorkspace)不丢能力 | ✅ PASS | `capcw_*`（§10） |
| slot 数=绑定容量 → 穷则变前提 | ✅ PASS | `capcw_growth_eval.md` |
| 接控制层：价值在内容/状态取回（与 B2 互证） | ✅ PASS（+0.39） | `capcw_action_eval.md` |
| 序列相邻算子 → induction（2×2 交互） | ✅ PASS | `capcw_induction_seq_eval.md` |
| 多跳链式：潜迭代读 | ⛔ FAIL | `capcw_multihop_eval.md` |
| 多跳链式：decode→re-embed + 中间监督(CoT) | ✅ PASS(机制)，+0.30 | `capcw_multihop_cot_eval.md` |
| 多跳 CoT 接回 controller（对内多步推理） | ✅ PASS（balacc 0.889/链尾 0.912） | `capcw_multihop_controller_eval.md` |
| 活文本多跳闭环（复合所有格→链式+CoT trace） | ✅ PASS（"A的经理的工位"→答；1.0/1.0/0.0） | `capcw_multihop_dialogue_eval.md` |
| 多跳主因（2×2 析因纠偏） | ✅ 中间监督 +0.27 ≫ decode +0.03 | `capcw_multihop_emergent_cot_eval.md` |
| 中间监督可自举（单跳标签免 GT 中间标签） | ✅ PASS（达 GT 97%） | `capcw_multihop_selfdistill_eval.md` |
| 朴素自举接活系统 | ⛔ 诚实负（单跳训练路由弥散→过度追问；校准纠偏） | `capcw_multihop_bootstrap_eval.md` |
| 开放关系 NLU（免"记住"标记绑定） | ✅ PASS（高精度窄触发） | §29 + `test_multihop_dialogue.py` |
| 自我成长（穷则变接活 WM） | 🟡 机制成立、小 WM 不划算 | `capcw_wm_growth_eval.md` |
| 小 d 绑定稳定性 | ✅ 已稳（iters=3 最优） | `capcw_binding_stability_eval.md` |
| 推理基元·比较/计数（检索之外） | ⛔ 诚实负·边界 | `capcw_reasoning_primitives_eval.md`（compare 优势=容量效应非新推理、count 无 headroom；CAPCW=内容寻址引擎，非算术聚合器） |
| 规则归纳·直面"连连看"（in-context 规则外推） | ✅ PASS(超连连看) | `capcw_rule_induction_eval.md`（UNSEEN 规则外推 0.97≫随机：不止查表、会归纳规则外推到未见输入；规则归纳=readout 之功，内容寻址负责取回/绑定） |
| 容量扩展曲线·"裸增 d 能否到好效果" | ⛔ 裸增 d 不抬反降 | `capcw_capacity_scaling_eval.md`（CAPCW 0.82@d16→0.07@d128，d≥64 弛豫塌缩非欠训；要 scale 须架构工程=重建标准大模型/按纪律不走；机制结论与 d 无关） |
| 从序列读关系（更像真语言） | ✅ PASS | `capcw_sequence_relation_eval.md`（扁平 token 序列读 (实体·关系→值) 三元组并取回，多 K≥6 CAPCW−flat +0.564；内容寻址不限于显式 pair） |
| 活文本工程加固：指代消解 + 有界工作记忆 | ✅ | 他/她/它→上文实体=自然录入链；链式 WM 词表满 FIFO 淘汰不崩=优雅降级（`test_multihop_dialogue.py`/`test_capcw_chain_memory.py`，§34） |

一句话：CAPCW 是 FE-LLM **第一个有系统实证的核心引擎**——内容寻址绑定/induction、接回真实 controller 活
文本主动推理闭环、对内多步推理（CoT 链式 + 可溯源 trace）都成立；多跳"关键"经 2×2 析因定位为**中间监督**
且可自举；并能**从示例归纳规则、外推到未见输入**（in-context rule learning，UNSEEN 0.97，§31——**不止查表/
连连看**）；边界=不做通用算术/聚合运算（§30）、小 d 容量（按容量纪律不增 d）。

## 3. 六要素 → 实现

| 要素 | 状态 | 证据 |
|---|---|---|
| 能量解释 | ✅ | EnergyDecoder 残余能量轨迹（trace 可读） |
| 预测误差 | ✅ | surprise 多通道显式；真实数据语义-surprise 有区分力 |
| 自由能平复 | ✅ | 弛豫 + 澄清满足后 surprise 显著下降（实验B/真实数据） |
| 主动推理 | ✅ | 动作选择 bal_acc 0.929；合成任务型 belief headroom 强（+0.5，含构造特性）。真实数据(CrossWOZ)精确定位：belief 价值在**状态/领域追踪(+0.19)与回复内容 grounding(+0.15)**，不在动作类型(−0.02)——见 B2 系列 |
| 可溯源 | ✅ | trace 全链路 + 显式 surprise 通道 + 能量轨迹；CAPCW 引擎额外给 responsibilities/误差/F-trace + 多跳 **CoT trace**（中间符号链可读） |
| 自我成长 | ✅ | 短期 belief → 长期 memory 候选→确认 → 离线蒸馏回训；CAPCW 穷则变（按需长 slot，机制成立、小 WM 不划算 🟡） |

> 六要素在 CAPCW 核心引擎路径上也已落到实处（能量/预测误差/自由能平复=PCWorkspace 弛豫；主动推理=query 路由 surprise→ASK/ANSWER + 多步推理；可溯源=trace+CoT trace；自我成长=穷则变），即"控制层"与"核心引擎层"两条路径都覆盖六要素。详见 `阶段总结.md` §5.1。

## 4. 关键验证证据（均为可复跑脚本 + 报告）

- 机制价值首证：`docs/reports/contextual_action_headroom.md`（belief 歧义子集 0.773→1.0）
- 真实 controller headroom：`docs/reports/contextual_controller_headroom.md`（1.0 vs 0.0）
- 成长闭环：`docs/reports/memory_growth_audit.md` + 离线蒸馏 `memory_distill_report.md` + 回流 `offline_retrain_eval.md`（0→1.0）
- 生成底座 48k 重训：held-out decode_loss 2.18→1.39、acc 0.50→0.66
- 学习式 NLU：`nlu_intent_eval.md`（改写 0.925 vs 关键词 0.06）+ 值标注 `nlu_value_eval.md`（DATE/TIME 0.54、CITY 容量受限）
- 真实数据验证：`real_data_validation.md`（开放闲聊 belief 0.655 弱）
- 任务型语料 headroom：`teacher_corpus_eval.md`（歧义 0.49→1.0）+ policy `context_policy_train.md`（留出 1.0）+ 任务 NLU `task_nlu_eval.md`（0.857）
- **真实任务数据 belief 价值地图（B2 系列，CrossWOZ 真实人标）**：动作类型 `crosswoz_headroom_eval.md`（−0.02 无）/ 状态·领域追踪 `crosswoz_domain_tracking_eval.md`（未明示子集 +0.19 强）/ 回复内容 grounding `crosswoz_response_content_eval.md`（+0.15 强）→ belief 真实价值在"语境/内容"维
- **预训练底座 N2（封存）**：`backbone_policy_probe.md` + `backbone_taskdomain_probe.md`（冻结 Qwen2.5-0.5B 句向量在 action 分类/领域理解上均不优于自建 8.9M 编码器，底座线再次封存）
- **核心引擎 CAPCW（系统实证）**：内容寻址绑定 `capcw_binding_eval.md`、induction 2×2 `capcw_induction_seq_eval.md`、接控制层 `capcw_action_eval.md`、多跳 CoT `capcw_multihop_cot_eval.md`、接回 controller `capcw_multihop_controller_eval.md`、活文本多跳 `capcw_multihop_dialogue_eval.md`、2×2 纠偏 `capcw_multihop_emergent_cot_eval.md`、自蒸馏 `capcw_multihop_selfdistill_eval.md`、校准纠偏 `capcw_multihop_bootstrap_eval.md`（见 2.5 节全表）

## 5. 诚实边界（蓝图未完全实现处）

- **分层预测编码世界模型**（道易"象/卦/爻"的深层学习版）：⛔ **正式封存**，**已由 CAPCW 取代为核心引擎**（2.5 节）。封存依据：组合泛化（分层被理论认为唯一不可替代的好处）上 flat ≥ hierarchical（−0.014）+ V2-M1 的 decode_loss 代价——本规模下过度设计，连理论主场都赢不了扁平。CAPCW 用的是 Transformer 已验证的内容寻址横向结构（非固定纵向分层），故成立。
- **CAPCW 容量边界**：小 d（=32，按容量纪律刻意做小）是硬边界——绑定/取回的**绝对**精度随负载/跳数下滑（高 K 均值 0.93→0.82、3 跳链尾 ~0.89）；机制（内容寻址优势、链式增量、自举增量）在容量受限区间稳健成立，但绝对天花板要靠增 d（违纪律，不取）。诚实：CAPCW 是 in-context 绑定/工作记忆引擎，非记忆型世界知识（真实可记忆槽值单向量记先验即赢，`capcw_crosswoz_eval`）。
- **生成质量**：小模型（8.9M 字符级）受容量限制，生成偏弱；价值在控制/推理/溯源/成长层，不在生成博学度。
- **理解层**：意图/动作已学习化（NLU/context_policy）；部分槽位值仍规则+gazetteer（开放实体 NER 受容量限制，诚实）。
- **taxonomy 统一**：已落地阶段一——`nlu/taxonomy.py` 单一真相源（canonical 9 领域 + legacy 映射 + 已知简化），controller/教师两侧均派生自它、`tests/test_taxonomy.py` 锁一致性（行为保持）。阶段二（controller `booking`→route+date 等行为对齐）待定；B2 提示真实数据未必需要此对齐。
- **belief 在真实数据的边界（B2 系列）**：belief 对真实任务对话的**动作类型**预测无 headroom（动作由当前 utterance 决定），价值在状态/领域追踪与内容 grounding。早期"belief 决定动作"的强结论部分是合成语料构造特性，已诚实修正。

## 6. 复现

- 全量回归：`python -m pytest -q`（**200 tests**，含 CAPCW 引擎/工作记忆/活文本闭环/会话隔离/多跳链式/活文本多跳/开放关系 NLU/指代消解/有界 WM）。
- 端到端 demo：`fe_llm_demo`（实录）、`fe_llm_demo_web`（HTML）、`fe_llm_cli`（交互）、`fe_llm_web_server`（网页）、`fe_llm_multidomain_demo`（多域 belief 追踪）、**`fe_llm_brain_demo`（统一活大脑 · Linear 风浅色 HTML：知道何时不该答+多跳推理+可溯源思维链+指代+grounded+surprise 平复+成长 一段对话全展示）**。
- 控制层各 eval：`fe_llm/active_inference/experiments/*.py --run`。
- CAPCW 核心引擎各 eval：`fe_llm/world_model/capcw_*.py --run`（绑定/induction/接控制层/多跳 CoT/接回 controller/活文本多跳/2×2 析因/自蒸馏/校准）。
