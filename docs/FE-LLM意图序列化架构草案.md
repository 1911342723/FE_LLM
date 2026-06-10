# FE-LLM 意图序列化架构草案（象的层级结构）

> 起因：opus-100 翻译泛化实验（docs/reports/translation_eval.md）证明，
> 单个 128 维全局意图向量能承载"行动选择级粗语义"（policy 95.8%、闲聊可用），
> 但独自承载不了"全句重建级细粒度语义"（200 条未见句 mean word-F1 ≈ 0.07）。
> 本草案是道易方案第 6 节"象/卦/爻层级"的工程化转译，按规范架构方案的命名
> 原则，代码中使用标准 ML 术语，哲学概念只出现在文档里。

## 1. 问题定义

当前 `IntentLM` 的信息通路：

```text
prompt (≤48 token) → IntentEncoder → z ∈ R^128 → EnergyDecoder → response
```

瓶颈是确定的：48 个 token 的语义被压进单个 128 维向量后，
解码器只能恢复"这句话大概属于哪一簇"，恢复不了"具体说了什么"。
翻译任务需要 token 级对齐信息，单向量在信息论上就不够。

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

| 里程碑 | 内容 | 通过标准 |
|---|---|---|
| M1 | SlotIntentEncoder + SlotEnergyDecoder 在对话语料训练 | decode_loss 不劣于单向量版 |
| M2 | 翻译重测（关键判定实验） | 未见句 word-F1 显著超过单向量版的 0.07（目标 ≥0.3） |
| M3 | 槽位可解释性 | 槽位 attention 与输入要素（人名/时间/动作）有可视对应 |
| M4 | 接入控制层 | 分槽 surprise 定位 + 缺槽追问，实验 B 口径不回退 |

M2 是整个设计的判定实验：如果槽位化后翻译泛化仍无显著改善，
说明瓶颈不在表达结构而在模型容量/训练量，应转向预训练底座路线。

## 6. 风险

- 槽位塌缩（全部槽学成一样）：用 L_slot_div + 逐槽 dropout 对抗；
- 覆盖度能量与语言流畅性打架：λ_cov 从小起步，复用 hybrid 复合打分经验；
- 计算量：cross-attention 增加 ~K 倍 decoder 读取成本，K=8 时可忽略。
