# FE-LLM 道易主动推理方案草案

## 1. 核心判断

FE-LLM 可以使用注意力机制，但不能把注意力机制当作可溯源解释本身。

注意力适合作为感知层的证据路由机制：它回答“当前输入中哪些片段彼此相关”。但 FE-LLM 的可溯源、主动推理、自我成长，必须来自更高层的显式状态更新：

- 世界状态：模型当前相信什么。
- 惊奇来源：哪些观察打破了稳定。
- 预测误差：模型预期与输入之间差在哪里。
- 行动策略：回答、追问、检索、拒答、更新记忆等候选行动。
- 自由能评估：每个行动为什么降低或不能降低未来不确定性。
- 成长记录：哪些信念被短期修正，哪些进入长期记忆或参数训练。

因此，注意力可以作为“目”，但不能作为“道”。目能见物，道决定如何顺势而行。

## 2. 注意力与可溯源

学界对 attention 是否等于 explanation 有持续争论。Jain 与 Wallace 的结论是，attention weight 不能直接等同于模型解释；Wiegreffe 与 Pinter 的反驳则更温和：attention 不是永远不能解释，但需要满足模型一致性、因果干预和任务约束。

对 FE-LLM 来说，保守结论是：

> attention trace 只能作为候选证据，不是最终溯源。

真正的溯源应来自如下结构化日志：

```text
Observation:
  用户输入了什么

Prior Belief:
  模型原本怎样理解世界/任务/用户意图

Prediction:
  模型原本预期会看到什么

Prediction Error:
  哪些语义、逻辑、目标、安全约束发生冲突

Precision:
  哪些误差被认为可靠，哪些可能是噪声

Policy Candidates:
  answer / ask / retrieve / refuse / update_memory

Expected Free Energy:
  每个策略的 risk、ambiguity、epistemic value、cost

Selected Action:
  为什么选择这个行动

After-State:
  输出后内部稳定性如何改变
```

## 3. 道：生成秩序，而不是一个对象

《道德经》中的“道”不适合被工程化成某个单一变量。它更适合作为 FE-LLM 的生成原则：

- 道不是某个具体知识，而是知识、行动、生成能够成立的底层秩序。
- 道不是强控制，而是让系统沿着最小阻力、最低扰动、最高协调的路径自组织。
- 道不是静态规则，而是允许“反”“复”“弱”“损益”这些调节机制存在。

可转译为：

```text
Dao = viability prior + generative law + stability condition
```

它不是模型输出的一部分，而是模型如何评价内部/外部状态是否协调的尺度。

## 4. 德：道在局部行为中的显现

如果“道”是全局秩序，“德”就是这个秩序在具体 agent 行为中的体现。

FE-LLM 中可以把“德”理解为：

```text
De = policy disposition under Dao
```

也就是模型在具体场景下形成的稳定行动倾向。例如：

- 信息不足时倾向追问，而不是胡答。
- 发现矛盾时倾向暴露矛盾，而不是掩盖。
- 用户意图不清时先压低生成冲动。
- 高风险问题下选择拒答或转安全解释。
- 长期反复出现的误差进入自我成长队列。

这样，“德”不是道德说教，而是稳定系统在局部条件下自然表现出的行动品质。

## 5. 无为：最小干预策略

“无为”不应理解成不行动，而应理解成不强行行动、不逆势行动、不用过度控制制造额外自由能。

工程化定义：

```text
wu_wei(policy) = argmin unnecessary_intervention
                 while reducing expected_free_energy
```

对应 FE-LLM 的策略是：

- 能不回答时，不强答。
- 能追问解决时，不猜测。
- 能引用证据时，不凭空编造。
- 能局部修正时，不全局重写。
- 能短期记忆处理时，不立即写入长期自我。

这会让模型呈现一种很不同的智能气质：不是“永远输出最多”，而是“在合适的时候做最小但有效的动作”。

## 6. 易：变化中的决策系统

《易》比《道德经》更接近工程架构，因为它处理的是：

- 观象：从现象中抽取结构。
- 立卦：把复杂局势编码成有限状态。
- 爻变：局部状态变化导致整体态势改变。
- 吉凶悔吝：行动后果评估。
- 穷变通久：当旧结构走到极限，就需要变；变了才能通，通了才能久。

可转译为 FE-LLM：

```text
xiang = situation representation
gua = compressed world-state pattern
yao = local factors / active constraints
bian = transition operator
ji_xiong = policy outcome evaluation
```

这里最关键的是：FE-LLM 不应该只有一个连续向量 intent，而应该有“象”的层级结构。

## 7. 道易结合后的 FE-LLM 循环

```text
Prompt enters as Observation
  -> Attention routes sensory evidence
  -> Predictive Coding compares observation with prior world state
  -> Surprise decomposes into semantic / logical / task / risk errors
  -> Yi-style Situation Encoder forms xiang/gua/yao
  -> Active Inference evaluates possible bian/policies
  -> Dao prior favors low-force, stable, coherent action
  -> Decoder expresses selected action as language
  -> After-state records trace and optional growth signal
```

## 8. 自我成长的边界

FE-LLM 的自我成长不应一开始就做在线改参数。更好的三层成长是：

1. 短期信念更新：只在当前对话内改变 belief state。
2. 长期记忆候选：重复出现、被验证的稳定模式进入 memory queue。
3. 离线结构成长：经过审计的数据再进入训练或蒸馏。

对应“易”的思想：不是遇到任何变化都立刻变，而是“穷则变”。只有当原结构解释力走到极限，才触发结构性更新。

## 9. 建议的第一版工程目标

第一版不要试图证明“我们比 Transformer 更强”。第一版要证明：

> FE-LLM 比普通语言模型更知道何时不该直接回答，并能说明自己为什么采取某种行动。

最小原型：

```text
fe_llm/
  active/
    world_state.py
    surprise.py
    situation.py
    policy.py
    trace.py
    controller.py
```

其中：

- `situation.py` 负责“观象/立卦”：把输入局势编码成结构化状态。
- `policy.py` 负责“变”：生成候选行动。
- `controller.py` 负责“通”：选择能降低未来自由能的行动。
- `trace.py` 负责“可溯源”：记录从观察到行动的整条链。

## 10. 文献与文本入口

- Jain & Wallace, Attention is not Explanation: https://arxiv.org/abs/1902.10186
- Wiegreffe & Pinter, Attention is not not Explanation: https://arxiv.org/abs/1908.04626
- Stanford Encyclopedia of Philosophy, Laozi: https://plato.stanford.edu/entries/laozi/
- Stanford Encyclopedia of Philosophy, Daoism: https://plato.stanford.edu/entries/daoism/
- Stanford Encyclopedia of Philosophy, Chinese Philosophy of Change: https://plato.stanford.edu/archives/spr2024/entries/chinese-change/
- Chinese Text Project, Dao De Jing: https://ctext.org/dao-de-jing/ens
- Chinese Text Project, Book of Changes - Xi Ci I: https://ctext.org/book-of-changes/xi-ci-shang/ens
- Chinese Text Project, Book of Changes - Xi Ci II: https://ctext.org/book-of-changes/xi-ci-xia/ens

