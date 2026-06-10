# FE-LLM 意图序列化架构草案（象的层级结构）

> 起因：opus-100 翻译泛化实验（docs/reports/translation_eval.md）证明，
> 单个 128 维全局意图向量能承载"行动选择级粗语义"（policy 95.8%、闲聊可用），
> 但独自承载不了"全句重建级细粒度语义"（200 条未见句 mean word-F1 ≈ 0.07）。
> 本草案是道易方案第 6 节"象/卦/爻层级"的工程化转译，按规范架构方案的命名
> 原则，代码中使用标准 ML 术语，哲学概念只出现在文档里。
>
> **判定结果（2026-06-11）：M2 FAIL。** 同预算重测 word-F1 0.083（仅 1.13x 基线），
> 详见第 7 节与 docs/reports/slot_translation_eval.md。按第 5 节预案转
> 预训练底座路线（docs/FE-LLM预训练底座路线草案.md）。

## 1. 问题定义

当前 `IntentLM` 的信息通路：

```text
prompt (≤48 token) → IntentEncoder → z ∈ R^128 → EnergyDecoder → response
```

瓶颈是确定的：48 个 token 的语义被压进单个 128 维向量后，
解码器只能恢复"这句话大概属于哪一簇"，恢复不了"具体说了什么"。
翻译任务需要 token 级对齐信息，单向量在信息论上就不够。

## 1.5 与核心思想的一致性审查（重要：这不是引入外来范式）

槽位化是否与 FE-LLM 的思想体系冲突？逐条对照原始文献：

### 1.5.1 道易草案的直接要求

道易方案草案第 6 节原文：

> 这里最关键的是：FE-LLM 不应该只有一个连续向量 intent，而应该有"象"的层级结构。

并给出了转译表 `xiang = situation representation`、`gua = compressed world-state
pattern`、`yao = local factors / active constraints`。本草案的映射完全对应：

| 道易概念 | 本草案 | 语义 |
|---|---|---|
| 卦（局势整体） | `global_intent` | 压缩的全局态势判断 |
| 爻（局势中的活动要素） | `intent_slots` | 局部要素/活动约束 |
| 爻位的当与不当（精度） | `slot_salience` | 各要素的显著性/精度权重 |

结论：**单向量才是当时的工程简化，槽位化是回归蓝图，不是偏离蓝图。**

### 1.5.2 想法对话.md 的分层预测编码要求

想法对话第 2 节（核心计算机制）原文要求"分层预测编码"：

> 架构分为多个层级，从底层的"字词语法层"，到中层的"事实逻辑层"，
> 再到顶层的"抽象概念层"。

当前单向量意图把所有层级压成一个点，恰恰没满足这条；
`global_intent`（抽象概念层）+ `intent_slots`（事实要素层）是朝这条要求迈出的一步。

### 1.5.3 自由能原理本身的检查

Friston 框架中 generative model 的 hidden states 本来就是
factorized / hierarchical 的，自由能 = 各因子预测误差的精度加权和。
本草案的复合能量 `E = E_global + λ·Σ_k salience_k·E_slot_k`
正是这个形式：salience 即 precision，槽位即 factor。
能量递减生成、吸引子语义都保留——只是吸引子从"一个点"变成
"一个结构化的谷底"（能量地貌中的谷本来就可以是多维的）。

### 1.5.4 必须警惕的一个真实张力（及其约束）

道易草案第 1 节：注意力可作"目"（证据路由），不能作"道"（溯源本身）。
槽位化引入了 cross-attention，约束如下：

- cross-attention 只承担**读取/路由**职责（目）；
- 可溯源仍然来自显式记录：逐字残余能量轨迹、槽位覆盖度、salience 权重
  ——这些都是能量量，不是 attention weight；
- 禁止把 slot attention map 直接当作解释输出；它最多作为候选证据进 trace，
  与 Jain & Wallace 的结论保持一致。

### 1.5.5 对"终极目标六要素"的逐项检查（经验.md）

| 要素 | 槽位化后 | 变化 |
|---|---|---|
| 能量解释 | 复合能量（全局收敛 + 槽位覆盖） | 加强：能量有了结构化分解 |
| 预测误差 | 可分槽计算：哪个要素的预期被违背 | 加强：surprise 来源可定位到要素级 |
| 自由能平复 | 追问可以瞄准低置信槽位 | 加强：缺什么问什么 |
| 主动推理 | EFE 行动选择层接口不变 | 不变 |
| 可溯源生成 | 能量轨迹 + 覆盖度记录入 trace | 加强 |
| 自我成长 | 记忆候选可挂到具体槽位 | 后续可加强 |

## 2. 设计目标

把意图从"一个点"升级为"一个结构"，同时保留能量递减解码的核心机制：

```text
IntentState = {
    global_intent: R^d          # 全局压缩态（对应"卦"：局势的整体判断）
    intent_slots:  R^{K×d}      # K 个局部意图槽（对应"爻"：局势的活动要素）
    slot_salience: R^K          # 各槽位的精度/显著性权重
}
```

- `global_intent` 继续服务行动选择层（policy/EFE/belief），接口不变；
- `intent_slots` 服务生成层：解码器逐字 cross-attention 到槽位，
  恢复细粒度语义；
- `slot_salience` 是显式精度权重，后续可对接主动推理的 precision 机制。

## 3. 模块设计

### 3.1 SlotIntentEncoder（观象）

```text
输入 token 序列 → 现有 PERBlock 堆叠 →
  [CLS] 位 → global_intent（与现状相同）
  K 个 learned query 经 PER 弛豫读取序列 → intent_slots（K=8 起步）
```

实现要点：

- learned queries 与序列拼接后过 PERBlock（同 Perceiver/Q-Former 思路，
  但保留 PER 的预测-误差弛豫形式，而不是标准 attention pooling）；
- 槽间正交正则防塌缩：`L_slot_div = ||S·Sᵀ − I||²`（S 为归一化槽位矩阵）。

### 3.2 SlotEnergyDecoder（行动）

```text
每个解码位置 i：
  h_i = CausalPERBlock(...) + intent_proj(global_intent)      # 现状保留
  c_i = cross_attention(h_i, intent_slots)                     # 新增局部通路
  logits_i = head(h_i + c_i)
```

残余能量从单距离升级为复合能量：

```text
E_i = ||h_i − global_intent||                                  # 全局收敛（现状）
    + λ_cov · Σ_k salience_k · min_j≤i ||c_j − slot_k||        # 槽位覆盖度
```

直觉：生成不仅要"朝整体意图走"，还要"把每个该说的要素都说到"。
覆盖度能量给了 [EOS] 一个有原则的停止条件：所有高显著槽位被覆盖后能量才低。

### 3.3 训练目标

```text
total = L_decode                       # CE，条件于 (global_intent, intent_slots)
      + 2.0 · L_intent                 # 全局 InfoNCE（现状保留）
      + 0.1 · L_approach               # 全局能量递减（现状保留）
      + λ1 · L_slot_div                # 槽间正交，防塌缩
      + λ2 · L_slot_coverage           # 解码隐状态应覆盖全部高显著槽位
```

铁律（翻译实验的教训，已写入经验.md）：

1. 解码器训练时条件于 `encoder(prompt)` 的输出，绝不条件于
   `encoder(response)`——训练/推理必须同分布；
2. 任何"目标侧"向量只能作为对比学习的正例，不得进入解码条件；
3. 每个新 loss 项上线前先单独验证不破坏 L_decode 收敛。

## 4. 与主动推理控制层的对接

- `BeliefState.intent_vector` 映射到 `global_intent`（接口不变，渐进迁移）；
- `PredictionError` 可以分槽计算：哪个槽位的预期被违背，surprise 就来自哪里
  ——这让"惊奇来源"第一次有了结构化定位，溯源粒度从句级进到要素级；
- `ask_clarification` 的生成可以直接以"低置信槽位"为目标：缺什么问什么，
  替代现在的固定追问模板。

## 5. 验证路线（每步有可判定指标）

| 里程碑 | 内容 | 通过标准 | 结果（2026-06-11） |
|---|---|---|---|
| M1 | SlotIntentEncoder + SlotEnergyDecoder 训练 | decode_loss 不劣于单向量版 | 通过：decode_loss 1.13、教师强制下一字 64.4%，全部辅助损失正常收敛（槽正交 ~0.05，覆盖 8.5→0.74） |
| M2 | 翻译重测（关键判定实验） | 未见句 word-F1 显著超过单向量版的 0.07（目标 ≥0.3） | **FAIL**：word-F1 0.083，仅 1.13x，远低于 2x（PARTIAL 线）与 0.3（PASS 线） |
| M3 | 槽位可解释性 | 槽位 attention 与输入要素（人名/时间/动作）有可视对应 | 终止（M2 未过，按预案转线） |
| M4 | 接入控制层 | 分槽 surprise 定位 + 缺槽追问，实验 B 口径不回退 | 终止（M2 未过，按预案转线） |

M2 是整个设计的判定实验：如果槽位化后翻译泛化仍无显著改善，
说明瓶颈不在表达结构而在模型容量/训练量，应转向预训练底座路线。

## 6. 风险

- 槽位塌缩（全部槽学成一样）：用 L_slot_div + 逐槽 dropout 对抗；
- 覆盖度能量与语言流畅性打架：λ_cov 从小起步，复用 hybrid 复合打分经验；
- 计算量：cross-attention 增加 ~K 倍 decoder 读取成本，K=8 时可忽略。

## 7. M2 实测结果与路线判定（2026-06-11）

同数据（opus-100 5 万对）、同规模（8.9M）、同预算（80 epochs）重测，
唯一变量是意图表示结构。完整报告：docs/reports/slot_translation_eval.{md,json}。

| 指标 | 单向量基线 | 槽位化（K=8） |
|---|---|---|
| mean word-F1 | 0.0735 | **0.0832（1.13x）** |
| mean char-F1 | 0.583 | 0.642 |
| 输出多样性 | 67/200 | 89/200 |
| word-F1=0 占比 | 134/200 | 118/200 |
| exact match | 0/200 | 0/200 |
| 全局能量下降率 | 100% | 7.5% |
| 槽位覆盖能量下降率 | —— | 100% |

结论与判定：

1. **机制本身工作了**：辅助损失全部按设计收敛，输出多样性、char-F1、
   word-F1=0 占比均小幅改善——槽位通路确实被解码器使用
   （生成动力学从"靠近全局意图"转移为"满足槽位覆盖"，
   全局能量下降率 100%→7.5% 而覆盖能量 100% 下降即是证据）；
2. **但改善幅度（1.13x）远不足以支撑判定**：8 个槽位仍装不下足够的
   源句细粒度信息。两次实验（单向量 0.073、槽位 0.083）共同指向：
   瓶颈不在意图的表达结构，而在 8.9M 字符级模型从零学习的容量上限
   ——尤其 InfoNCE 跨语言对齐在小模型上全程学不动（L_int≈ln(batch)）；
3. **按本草案第 5 节预案执行转线**：转预训练底座路线，
   设计草案见 docs/FE-LLM预训练底座路线草案.md。
   槽位结构（global+slots+salience）作为意图接口的设计保留，
   在底座路线中复用——被否决的是"从零训练承载它的小模型"，
   不是结构本身。

一个对实验纪律有价值的插曲：训练在 ep24/80 被中断时，中断点
checkpoint 的自由生成完全坍缩（200 句仅 2 种输出，word-F1 0.062）；
跑满预算后多样性恢复至 89/200。**LR 高位阶段的自由生成坍缩是
训练阶段现象，不能在中断点提前判卷**——同预算对照是铁律。
