# FE-LLM 主动推理核心架构草案

> 本文只讨论当前真正主线：`fe_llm/`，尤其是 `fe_llm/energy_lm/intent_*`。  
> `kernel/` 可视为早期概念演示，不作为主架构依据。

## 1. 核心命题

FE-LLM 要研究的不是“另一个 next-token 模型”，而是：

> 外部 prompt 打破模型内部世界的稳定态，产生惊奇；模型为了重新平复自由能，会同时进行内部信念更新与外部行动选择，最终输出回答、追问、拒答或记忆更新。

也就是说，语言输出不是目标本身，而是主动推理中的一种行动：

- **对内平复**：改变内部信念、调用记忆、重构意图、调整世界模型。
- **对外平复**：回答、反驳、追问、要求补充条件、拒绝不可解析输入。

这个定位比“非自回归替代 GPT”更稳，也更贴近 `想法对话.md` 的原始哲学。

## 2. 当前代码对应关系

### 已经有雏形的部分

- `fe_llm/embedding/`
  - 对应感知层：把外部文本压成向量。
  - `cosine_distance()` 已经可以作为浅层语义惊奇。

- `fe_llm/energy_lm/intent_model.py`
  - `IntentEncoder`：prompt 经 PER 弛豫成意图向量 `z*`。
  - `EnergyDecoder`：给定意图向量和前缀，生成文本。
  - `IntentLM`：把“感知/思考/行动”合成一个模型。

- `fe_llm/energy_lm/intent_train.py`
  - `L_intent`：prompt 意图接近 response 意图。
  - `L_decode`：仍然用逐字交叉熵恢复语言能力。
  - `L_approach`：要求解码隐状态逐步接近意图，这是当前最接近“能量递减”的训练项。

- `fe_llm/energy_lm/intent_generate.py`
  - 已经写出三阶段：感知惊奇 -> PER 弛豫出意图 -> 朝意图递减生成。
  - 但当前生成选字仍主要用 `logit argmax` 近似，真正的“候选行动 EFE 评估”还没做。

### 目前缺的关键部分

- 没有显式的世界状态 `WorldState`。
- 没有把 prompt 与当前世界状态比较得到多层级 surprise。
- 没有真正的 action policy 选择：回答 / 追问 / 检索 / 内部更新 / 拒答。
- 没有 expected free energy 对候选行动打分。
- 没有把“用户下一轮反应是否更稳定”纳入训练或评估。

## 3. 建议的最小闭环

第一版不要先做宏大世界模型。先做一个可跑闭环：

```text
外部 prompt
   |
   v
[Sensory Encoder]
   - 文本向量 o_t
   - 噪声分数
   - 领域/任务分类
   |
   v
[Surprise Estimator]
   E_semantic = distance(o_t, expected_context)
   E_noise    = malformed / OOD / low parseability
   E_task     = task mismatch or missing slots
   |
   v
[Belief Relaxation / Intent Encoder]
   z_t = PER(prompt, memory, current_belief)
   目标：找到能解释输入的内部意图状态
   |
   v
[Policy Selector]
   候选行动:
     A1 answer      直接回答
     A2 ask         追问澄清
     A3 retrieve    检索记忆后再答
     A4 update      暂存新假设/局部世界更新
     A5 refuse      输入不可解析或风险过高
   每个行动估计 Expected Free Energy
   |
   v
[Active Layer / Decoder]
   输出文字行动
   |
   v
改变外部环境: 用户下一轮输入更清晰或问题被解决
```

## 4. 数学定义草案

### 4.1 当前时刻自由能

把 prompt 视为观测 `o_t`，内部信念为 `s_t`，模型参数/世界模型为 `M`。

理论上：

```text
Surprise(o_t) = -log P(o_t | M)
```

工程上直接算 surprise 很难，所以使用可计算的变分自由能上界：

```text
F_t = E_semantic + E_intent + E_task + E_noise + E_risk
```

各项建议：

- `E_semantic = 1 - cos(embed(prompt), expected_context)`
- `E_intent = || z_prompt - z_expected ||`
- `E_task = missing_required_slots + contradiction_score`
- `E_noise = OOD_score + parse_failure_score`
- `E_risk = safety_or_policy_conflict`

注意：这些不是最终真理，只是第一版可测的 surrogate。

### 4.2 候选行动的预期自由能

行动不是“生成哪个 token”，而是“采取哪种方式降低未来自由能”。

```text
G(pi) = Risk(pi) + Ambiguity(pi) - EpistemicValue(pi) + Cost(pi)
```

其中：

- `Risk(pi)`：执行该行动后，预期结果偏离目标/偏好的程度。
- `Ambiguity(pi)`：行动后用户是否仍可能不清楚、不稳定。
- `EpistemicValue(pi)`：行动能带来多少信息增益，比如追问能否减少歧义。
- `Cost(pi)`：算力、长度、风险、打扰成本。

示例：

- 用户问得清楚：`G(answer)` 最低。
- 用户缺关键条件：`G(ask)` 最低。
- 用户说乱码：`G(refuse/rephrase)` 最低。
- 用户提出新事实但可信度不明：`G(retrieve)` 或 `G(ask evidence)` 最低。

## 5. 输出文字如何产生

这里要承认一个现实：真正的人类可读文本仍需要顺序化。

因此建议把生成拆成两层：

### 第一层：行动选择

先决定要做什么：

- 回答；
- 追问；
- 反驳；
- 总结；
- 拒答；
- 检索后回答；
- 暂存假设。

这才是主动推理的核心。

### 第二层：语言实现

再把行动落成文字。此时可以用 `EnergyDecoder`：

```text
目标意图 z*
当前前缀 y_<i
候选 token w
打分 = 语言可读性 + 距离 z* 的残余能量 + 风险/冗余惩罚
选择使总能量下降最多的 token
```

第一版可以保留 cross-entropy 训练，不要硬说完全脱离概率。更准确地说：

> 语言能力用 CE 学，行动选择和可解释轨迹用自由能组织。

这比“彻底不用 softmax”更可落地。

## 6. 世界模型先做什么

不要一开始灌入“全世界的原理”。第一版世界模型只需要三类状态：

1. **对话状态**
   - 当前主题；
   - 用户目标；
   - 已知约束；
   - 缺失槽位；
   - 上一轮模型行动。

2. **任务状态**
   - 问答 / 写作 / 推理 / 闲聊 / 设定扮演 / 不可解析。

3. **稳定性状态**
   - 当前 surprise；
   - uncertainty；
   - 是否需要追问；
   - 是否需要检索；
   - 是否应该停止。

世界模型第一版可以只是一个结构化对象，不必马上是大型图数据库：

```python
WorldState = {
    "topic": ...,
    "user_goal": ...,
    "constraints": [...],
    "missing_slots": [...],
    "belief_vector": ...,
    "expected_next_observation": ...,
    "surprise_history": [...],
}
```

## 7. 最小实验设计

### 实验 A：高惊奇输入是否触发正确行动

数据集分四类：

- 清晰输入：应直接回答。
- 缺条件输入：应追问。
- 乱码输入：应要求重述。
- 世界观冲突输入：应解释冲突或询问是否进入假设设定。

指标：

- action accuracy；
- surprise 分数是否区分四类；
- 追问后用户补充信息时，下一轮 surprise 是否下降。

### 实验 B：对内平复是否有效

给定一个模糊 prompt：

```text
“帮我写一下那个方案”
```

模型不应硬写，而应识别缺槽：

- 什么方案？
- 给谁看？
- 什么目的？
- 多长？

若追问后用户补充信息，内部 `missing_slots` 减少，`F_t` 下降。

这就是“外部行动改变环境，从而降低未来自由能”的最小证据。

### 实验 C：对外平复是否优于直接回答

对比两种系统：

- baseline：永远直接回答。
- FE-agent：按 EFE 选择回答/追问/拒答/检索。

评测：

- 最终任务成功率；
- 平均轮数；
- 幻觉率；
- 用户补充率；
- 自由能是否单调下降。

如果 FE-agent 在不确定输入上更少胡编、最终成功率更高，这条路线就有硬证据。

## 8. 现阶段应避免的说法

避免：

- “彻底摒弃 Transformer”
- “根除幻觉”
- “不再需要概率”
- “世界原理驱动的大模型已经成立”

建议：

- “以自由能/主动推理组织推理与行动选择”
- “语言生成仍可使用神经解码器，但上层行动由 EFE 选择”
- “研究如何让模型在高惊奇输入下选择追问、检索或内部更新，而不是直接胡编”
- “探索 surprise trace 是否能预测错误和触发澄清”

## 9. 下一步代码路线

建议新增四个模块，而不是继续堆 demo：

```text
fe_llm/
  active/
    world_state.py          # 对话/任务/稳定性状态
    surprise.py             # 多层级 surprise 计算
    policies.py             # answer / ask / retrieve / update / refuse
    controller.py           # expected free energy 行动选择
```

然后把现有 `IntentLM` 接进来：

```text
prompt
 -> SurpriseEstimator
 -> IntentEncoder
 -> ActiveInferenceController
 -> EnergyDecoder or Ask/Refuse template
 -> update WorldState
```

第一阶段先让它会“何时不回答”，这比让它“回答得像 GPT”更符合 FE-LLM 的哲学。

