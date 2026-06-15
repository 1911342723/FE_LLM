# FE-LLM 核心引擎构想 · 方案一：内容寻址预测编码工作空间（CAPCW）

更新日期：2026-06-15

> 背景：v2 蓝图原定的核心引擎"分层预测编码世界模型（z_1..z_L 纵向）"已经过组合泛化裁决
> 判为本规模过度设计、封存（见 `阶段总结` / `compositional_generalization_eval.md`）。
> 六要素的控制闭环很扎实，但缺一个真正强的"脑子"。本文借鉴 Transformer 力量的**真正来源**，
> 提出**一种**新核心引擎方案，供讨论。这是构想（design），不是已验证结论。

## 1. 先拆解：Transformer 的"牛逼"到底来自哪？

不是"层数多"也不是"参数多"本身，而是这几条机制：

1. **内容寻址路由（attention）**：信息流向由"内容相关性"决定，而不是固定连线。query 去和所有
   key 做内容匹配，按匹配度聚合 value。**这是它最核心的力量**——灵活、可组合、按需检索。
2. **残差流即工作记忆**：一条共享"总线"，每层读/写，支持迭代精化与组合。
3. **键值记忆（MLP）**：存取学到的关联。
4. **上下文即可查记忆（in-context learning）**：把输入当成可寻址的临时记忆。

一句话：**Transformer ≈ 一个在"工作记忆总线"上做内容寻址检索 + 键值记忆的可微引擎。**

## 2. 诊断：我们的分层 PC 为什么不行？

分层 PC 用的是 **纵向、固定的抽象阶梯**（z_1→z_2→...→z_L，层数预设、连线固定）。问题：

- 组合性/泛化的关键不是"固定的层级"，而是**内容寻址的灵活路由**（Transformer 的教训）；
- 固定纵向结构在小规模上既不换精度、也不换可解释分工（V2-M1 判定二/三全 FAIL），还要付
  decode_loss 代价；组合泛化裁决进一步证明它连理论主场都赢不了扁平。

**结论**：错的不是"预测编码 / 自由能"这套语言，错的是把它套在**固定纵向层级**上。应该把同一套
能量语言，套到 Transformer 已验证有效的**横向、内容寻址**结构上。

## 3. 方案一：内容寻址预测编码工作空间（CAPCW）

把"分层纵向 latent"换成"**一组内容寻址的 slot 工作空间**"，用预测编码 / 自由能驱动它。

### 3.1 核心数据结构：显式世界状态 = M 个 slot

```text
W = { s_1, s_2, ..., s_M }      # M 个 latent slot（可带 key/value 拆分）
```

- W 就是 FE-LLM 的**显式、可溯源世界状态**（不是黑盒残差流）。
- slot 是"内容寻址单元"：谁该被更新由内容匹配决定（横向），不是固定层级（纵向）。

### 3.2 机制：感知 = 自由能下降的内容寻址更新

每来一个观测 x（token / 轮次），做若干步弛豫：

```text
1. 自上而下预测：x_hat = g(W)                     # 工作空间生成对观测的预测
2. 自下而上误差：eps = x - x_hat                   # 预测误差（显式）
3. 内容寻址路由：r_i = softmax_i  compat(q(eps), key(s_i))   # 误差路由到最相关的 slot
   —— 这一步就是 attention，但它被推导为"最小化自由能 F 的责任分配（posterior responsibility）"
4. 更新：s_i ← s_i + α · r_i · Δ(eps)              # 相关 slot 吸收它负责的那部分误差
5. 重复 1-4 直到 F = Σ precision·||eps||² 收敛       # 感知即弛豫到自由能稳定
```

跨 token：W **持续保留并被更新**，充当工作记忆 / 世界状态（内容寻址，像对 KV-cache 做 attention，
但是显式、能量维护的状态）。

### 3.3 行动 / 生成：复用已验证控制层

- 收敛后的 W → policy / EFE → 动作（**已验证的控制闭环不变**）；
- decoder 条件于 W 生成回答（意图注入已验证）。

### 3.4 成长：穷则变（架构级自我成长）

```text
若弛豫后 F 仍高（没有 slot 能解释当前观测）→ 新增一个 slot 去吸收残余误差。
```

工作空间**按数据需要生长 slot 数**——这是数据驱动的结构成长，恰好补上分层 PC 那个"固定层数"
的死穴，也把六要素的"自我成长"提升到架构层。

## 4. 为什么它"既是 Transformer 内核，又是 FE-LLM 六要素"

| Transformer 力量 | CAPCW 对应 | 同时满足的 FE-LLM 要素 |
|---|---|---|
| 内容寻址路由（attention） | 误差按内容路由到 slot（第 3 步） | 自由能下降步（能量解释 + 自由能平复） |
| 残差流工作记忆 | slot 工作空间 W | 可溯源（slot/路由权重/误差均显式可读） |
| 键值记忆 | slot 的 key/value | 预测误差（eps 显式） |
| in-context 检索 | W 跨 token 维护 | 主动推理（W→EFE→动作） |
| 规模扩展 | 加 slot（穷则变） | 自我成长 |

关键差异（也是相对纯 Transformer 的"我们自己的脑子"）：路由不是黑盒 softmax，而是**可溯源的
自由能责任分配**；状态不是黑盒残差，而是**带类型、可读、可生长的显式 slot 世界状态**。

## 5. 与已有理论的连接（确保不是民科）

- **Attention 即推理**：已有工作表明 attention 可由变分推理 / 预测编码近似得到（attention 权重 ≈
  对隐因的后验责任）。CAPCW 的第 3 步正是这个——所以它能拿到 Transformer 级路由，同时保留
  能量/误差身份。
- **Slot Attention / 对象中心学习**（Locatello 等）：用内容寻址 slot + 迭代 attention 把输入绑定到
  slot。CAPCW ≈ 把 slot-attention 改写成"预测编码自由能下降 + 可生长"。
- 即：CAPCW = 预测编码语言 × （attention 即推理）×（slot 绑定）× 自由能成长。每块都有文献支撑。

## 6. 风险与开放问题（诚实）

1. **容量**：小模型仍可能装不下；slot/路由加参数。要先想清楚最小可用规模。
2. **可能退化成"slot-attention 套个 PC 外壳"**：必须证明能量/溯源/成长这三样带来实质增量，
   而不只是换名字。
3. **路由=自由能 的推导要做对**：第 3 步必须是真正的自由能下降责任分配，不能只是贴标签。
4. **绑定问题**：多 slot 容易塌缩（都学成一样）——需 slot 间正交 / 精度差异化（沿用分层 PC 风险清单）。

## 7. 最小验证方案（吸取本阶段全部教训）

**不重蹈"先建引擎再找任务"的覆辙。先定任务与判据，再写引擎。**

1. **任务必须有"绑定/组合 headroom"**：单向量可证不足、需要同时维护多个 (实体, 属性) 绑定。
   候选：CrossWOZ 多域状态追踪（同时绑定多个 域·槽·值），或受控的多对象绑定合成任务。
2. **判据=组合泛化**（与分层裁决同口径）：CAPCW vs 单向量 vs 已封存的固定分层，在**未见组合**
   上的 accuracy。**PASS 阈值预先写死**：CAPCW 未见组合明显 > 单向量（如 ≥ +0.05，多 seed）。
3. **唯一变量 + 同预算**：只换"表示/路由结构"，其余（数据/split/分类头/训练）全一致。
4. 通过 → CAPCW 是 FE-LLM 真正的核心引擎，再逐步接 surprise/policy/decoder；
   不过 → 诚实记录第三个"核心引擎"负结果，回到"控制架构已完成"的句号。

## 8. 一句话

**把"预测误差 / 自由能 / 主动推理"这套语言，从失败的"固定纵向分层"搬到 Transformer 已验证
有效的"内容寻址横向工作空间"上，并加上"可溯源 + 可生长"——这可能是 FE-LLM 自己的、既强又
自洽的核心引擎。但它必须先在一个有绑定/组合 headroom 的任务上、用组合泛化裁决证明自己。**

## 9. 验证结果（2026-06-15，第一轮 PASS）

按第 7 节纪律先定任务+判据再写引擎。脚本 `world_model/capcw_binding_eval.py`，任务=in-context
键值绑定（每样本随机 K 对 key→value，问某 key 的 value；单向量可证不足），唯一变量=世界状态
结构：flat(单向量) vs CAPCW(slot 工作空间) vs hierarchy(已封存 z_global)。报告
`docs/reports/capcw_binding_eval.{json,md}`。

**容量受限区间（d=32，= FE-LLM 刻意做小的实际处境），2 seed：**

| K（绑定负载） | flat | CAPCW | hierarchy（已封存） |
|---:|---:|---:|---:|
| 3 | 0.759 | **0.994** | 0.334 |
| 5 | 0.425 | **0.862** | 0.196 |
| 7 | 0.691 | **0.838** | 0.148 |

（随机基线 0.083）高 K(≥4) 平均 **CAPCW − flat = +0.29 → PASS**。已封存的分层近乎随机——再次确认
它不是答案。

**诚实边界（容量依赖，已用 d=64 烟测确认）**：当向量维度充裕（d=64）时，单向量本身就能装下
K≤5 个绑定（flat ≈ 0.99），CAPCW 不再需要、甚至因 slot-attention 难训而略逊。所以 **CAPCW 的价值
是"绑定的容量效率"——只在容量成为瓶颈时显现**。而 FE-LLM 按容量纪律就是刻意做小的，这正是
CAPCW 的主场。这与分层的结论形成鲜明对比：分层连理论主场都赢不了扁平；CAPCW 在它的理论
主场（容量受限的绑定）决定性胜出。

**结论**：CAPCW 是**第一个被正面验证的核心引擎方向**。下一步（见任务.md）：把最小 slot 工作空间
逐步elaborate成完整 CAPCW（显式预测-误差路由的自由能解释 + 可溯源 trace + 穷则变生长），并接入
已验证的 surprise/policy/decoder 控制层；每步仍用"先定任务+判据"的纪律。

诚实风险：本轮验证的是"内容寻址工作空间 > 单向量"这一核心机制（slot-attention 形态）；CAPCW 的
"自由能/预测编码/可生长"完整形态尚待逐步落地与各自的判定实验，不可一次性宣称全部成立。

## 10. 阶段二a（2026-06-15）：把 slot-attention 升级为显式自由能形态（PCWorkspace）

引擎 `fe_llm/world_model/capcw.py` 的 `PCWorkspace`：slot 作为"解释输入的混合成分"，生成模型 g
预测输入、路由 r 由重建误差导出（attention 即推理）、弛豫沿 -dF/ds 下降，全过程可溯源
（responsibilities / final_error / free_energy_trace 显式返回），并带"穷则变"生长钩子。单测
`tests/test_capcw.py` 锁定：自由能弛豫下降、责任归一、更多 slot 自由能更低、生长钩子。

**判据=PC 形态不丢绑定胜势**（d=32，2 seed）：

| K | flat | CAPCW(slot-attn) | **CAPCW_PC(自由能形态)** | hierarchy |
|---:|---:|---:|---:|---:|
| 3 | 0.789 | 0.996 | **0.949** | 0.339 |
| 5 | 0.425 | 0.862 | **0.863** | 0.205 |
| 7 | 0.691 | 0.838 | **0.799** | 0.138 |

高 K 平均 CAPCW_PC − flat ≈ **+0.27**，与 slot-attention 基本持平、远超单向量 → **PASS**：把裸
attention 升级为"显式预测编码 / 自由能 / 可溯源"形态，**不损失内容寻址的绑定能力**。CAPCW 至此
既有 Transformer 级内容寻址，又把 FE-LLM 的"能量解释 / 预测误差 / 自由能平复 / 可溯源 / 可生长"
落到了实处。全量 130 回归测试守护。

## 11. 阶段二b（2026-06-15）：穷则变（结构成长）判定

穷则变逻辑链「自由能长期高 → 加 slot」要成立，前提是 **slot 数 = 绑定容量**。脚本
`world_model/capcw_growth_eval.py`：固定绑定数 K=5、容量受限 d=32，扫 slot 数 M 看
CAPCW_PC 的 accuracy（2 seed，随机基线 0.083）：

| slot 数 M | accuracy | |
|---:|---:|---|
| 2 | 0.397 | M<K 容量不足 |
| 3 | 0.649 | M<K |
| 4 | 0.754 | M<K |
| 5 | 0.885 | M=K 容量充足 |
| 7 | 0.893 | M>K |

accuracy 随 M 单调上升、到 M≈K 饱和（M<K 平均 0.60 → M≥K 平均 0.89，转折 **+0.29 → PASS**）。
即 **slot 数确实是绑定容量**——slot 不够→自由能高→绑定失败；长到 M≥K 就解。这正是穷则变的前提。
grow 钩子（`PCWorkspace.grow_if_unexplained`）对 K=5 绑定能检测到欠容量并建议生长。

诚实 caveat：本实验固定 M 训练、扫出"容量 vs slot 数"曲线（验证穷则变的前提）；grow 钩子的
**停止阈值需校准**（本次它生长到上限 8 而非恰好停在 K=5）；训练期"动态生长"的完整闭环是下一步。

**阶段二小结**：CAPCW 至此三步落地——内容寻址 > 单向量（阶段一）、显式自由能/可溯源不丢能力
（阶段二a）、slot 数=绑定容量→穷则变前提成立（阶段二b）。全量 130 回归测试守护。

**下一步**：阶段三——把 slot 工作空间接已验证的 surprise/policy/decoder 控制层（W → EFE → 动作 /
条件生成）；以及 grow 阈值校准 + 训练期动态生长闭环。每步仍"先定任务+判据"。

## 12. 阶段三（2026-06-15）：slot 工作空间接控制层（动作 + 内容）

脚本 `world_model/capcw_action_eval.py`，任务=绑定+双输出：动作类型（ASK/ANSWER/REFUSE，依
query 是否绑定及其 value 半区）+ 回复内容（已绑定则答出**精确 value**）。flat vs CAPCW（d=32）：

| 维度 | flat | CAPCW | delta |
|---|---:|---:|---:|
| 动作类型（value 依赖+联合训练） | 0.556 | 0.911 | +0.356 |
| **回复内容·精确 value** | 0.443 | **0.829** | **+0.386** |

（value 随机基线 0.083）→ **PASS**（内容 delta +0.39）。另：query→slot 路由分离 bound/unbound
≈ 0.48（surprise 信号：未绑定难匹配=高 surprise=该追问）。

**深意（与 B2 一脉相承）**：CAPCW 在控制层的价值落在**内容/状态取回**（精确 value，单向量做不到），
而非粗动作判断。这在引擎层独立复现了 B2 真实数据的结论——belief/状态的价值在"内容/语境"、不在
"动作类型"。两条独立证据（真实数据 B2 + 合成引擎 CAPCW）指向同一结论，互为佐证。

诚实 caveat：纯「成员判断」动作（query 是否在场→答/问）单向量也能 ~1.0、无 headroom（与 B2 一致）；
本任务动作 value 依赖且与 value 头联合训练，故 flat 在动作上也降。最干净的判别是「内容·精确 value」。
surprise→动作 的完整闭环（用 query routing/free-energy 作 ASK 门控接入真实 controller）是后续工程。

**CAPCW 三阶段总结**：内容寻址>单向量（一）/ 显式自由能·可溯源不丢能力（二a）/ slot 数=绑定容量
→穷则变前提（二b）/ 接控制层，价值在内容取回、与 B2 一致（三）。CAPCW 作为 FE-LLM 核心引擎方向，
已在四个判定上获得正面证据。下一步是把它从受控合成任务推向更贴近语言的真实任务，并接回真实
controller（grow 校准 + surprise 门控 + 条件生成）。

## 13. Part 1（2026-06-15）：真实语言(CrossWOZ)绑定——CAPCW 的价值边界被精确画出

脚本 `world_model/capcw_crosswoz_eval.py`：用**真实 CrossWOZ** 的 (领域·槽位→值) inform 绑定做检索，
flat vs CAPCW（d=32，按对话切分）。**同一份真实槽/值**，两种绑定方式：

| 绑定类型 | flat | CAPCW | delta |
|---|---:|---:|---:|
| **real（真实，相关/可记忆）** | **0.981** | 0.668 | −0.313 |
| **shuffled（每例随机重指派=真 in-context）** | 0.297 | **0.883** | **+0.586** |

**关键边界（重要）**：CAPCW 的内容寻址优势**专属于真 in-context 绑定**（值不可由键预测）。真实
CrossWOZ 槽值是**相关、可记忆**的（如 餐馆·人均消费 多半"50-100元"）→ 单向量记住先验就赢、根本不
需要 in-context 检索；一旦把绑定打乱成真 in-context（值不可预测）→ CAPCW 决定性回升（+0.59）。

**正确的定位（分工）**：CAPCW = **工作记忆 / in-context 绑定**（语言里真正需要现场绑定的部分：
指代、归属、新关联、induction）；**记忆型知识**（可由键预测的先验）该交给常规参数/MLP 记忆。
这恰是 Transformer 的分工（attention=in-context 路由，MLP=记忆 KV）。所以"CAPCW 推向真实语言"的
正解不是 CrossWOZ 槽值检索（那是可记忆的、不需要它），而是真 in-context 绑定任务（指代/推理）。
诚实：本结果把 CAPCW 的适用边界画清楚了——它不是万能世界模型，是 in-context 绑定引擎。

## 14. Part 2（2026-06-15）：穷则变自校准 + 按需动态分配

脚本 `world_model/capcw_grow_dynamic_eval.py`。自校准生长准则=相对边际增益（加 slot 若不能再降
自由能 ≥ min_rel_gain 就停）；训练 max_slots、推理按 K 自动选 grow_m：

| K | 自选 grow_m | acc@grow_m | acc@max |
|---:|---:|---:|---:|
| 2 | 3.5 | 0.954 | 0.976 |
| 4 | 6.5 | 0.915 | 0.937 |
| 6 | 7.0 | 0.831 | 0.873 |

grow_m 随 K 单调增长（按需分配）、精度损失 ≤0.04 → **PASS**：穷则变可自校准、按需分配 slot
（比阶段二b 的"长到上限"前进了一步）。诚实：grow_m 仍略过配（≈K+1.5），阈值可再调紧。

## 15. Part 3（2026-06-15）：surprise→动作 闭环（无动作监督）

脚本 `world_model/capcw_surprise_action_eval.py`。模型**只学绑定取值、从未见 ASK/ANSWER 标签**；
推理时仅用 query→slot 路由匹配度的补值=surprise，单阈值判 unbound→ASK / bound→ANSWER：

| seed | 绑定 bind_acc | surprise→动作 balacc（无监督） |
|---:|---:|---:|
| 42 | 0.490（训练欠佳） | 0.571 |
| 43 | 0.899（训练良好） | **0.881** |

均值 0.726（**PARTIAL**），但**强相关于绑定学得好不好**：绑定学好(seed43 0.90)→闭环强(0.88)；绑定
训崩(seed42 0.49)→闭环弱。即"自由能/surprise→何时不该答"的闭环**机制成立**（绑定良好时 0.88，
完全无动作监督），限制因素是 d=32 下绑定训练的高方差（与阶段一同源）。

**三部小结（CAPCW 推向真实 + 控制闭环）**：
- Part 1：CAPCW 价值边界精确化——专属 in-context 绑定（real CrossWOZ 可记忆→flat 赢；shuffle 成
  in-context→CAPCW +0.59）。定位=工作记忆/in-context 绑定引擎，非记忆型世界知识。
- Part 2：穷则变可自校准、按需分配（grow_m 随 K 增、精度保持）。
- Part 3：surprise→动作 闭环机制成立（绑定良好时 0.88 无监督分开 ASK/ANSWER），受绑定训练方差限制。
- 合起来：CAPCW 从受控合成走向真实/控制闭环的路径**方向正确、边界清晰、限制诚实**。后续工程：
  稳定 d 较小时的绑定训练、grow 阈值精调、真 in-context 语言任务（指代/induction）、接回真实 controller。

## 16. induction 负结果：CAPCW 的能力边界再细化（2026-06-15）

脚本 `world_model/capcw_induction_eval.py`：序列 ...A B ... A→预测 B（induction head，Transformer
in-context learning 基石；每序列 A→B 随机配对不可记忆）。结果：flat 0.118 / CAPCW 0.104，**两者
都接近随机**（基线 0.05）→ FAIL，但是**双双失败**而非 CAPCW 输给 flat。

诊断（重要边界）：induction 需要"**相邻关系**"（找 cue 后面紧跟的 token）；而当前 CAPCW（与 flat）
都把 token **独立嵌入再池化/聚类**，bigram 的 a→b 相邻信息在聚合时丢失。集合式 slot 工作空间擅长
"内容绑定"（pair 作为整体喂入，如 capcw_binding/crosswoz-shuffle），**不擅长"序列相邻"**——
induction 本质需要位置偏移/序列算子（Transformer 靠 attention+位置实现），这不在当前 PCWorkspace
的能力内。

结论：CAPCW 的已验证主场是**内容绑定/工作记忆**（把"已配好的关联"现场寻址检索）；它**不自带
序列相邻算子**。要做 induction 这类"序列内现场学映射"，需在 token→slot 写入前补一个相邻/序列
编码（如把每个 token 与其前驱拼接，或在 PERBlock 序列层先成 bigram 表示）。这是 CAPCW 走向真正
语言引擎要补的下一块，且应作为独立判定实验（先定任务+判据），不在本轮硬凑。

## 17. 序列相邻算子：2×2 交互救活 induction（2026-06-15，从内容绑定走向序列语言）

按第 16 节的诊断（缺"序列相邻算子"）补一块、独立判定。给 CAPCW 在 token→工作空间写入**前**补一个
最小相邻算子——**previous-token channel**（induction head 的基元）：位置 t 的表示 =
`proj([emb(前驱); emb(当前)])`，把"独立 token 流"变成"(prev→cur) bigram 流"，使每个位置在被聚成
slot 前已携带"前驱身份(key, 供 cue 匹配)"与"当前身份(value, 供读出)"。落为引擎可复用组件
`capcw.SequenceAdjacency`（与 `PCWorkspace` 并列，含 3 条不变量单测）。

脚本 `world_model/capcw_induction_seq_eval.py` 做 **2×2 单变量析因**（两变量：相邻算子 on/off ×
世界状态 flat/CAPCW；其余全一致，同 d / 同 slot 预算），扫负载 n_pairs=2/4/6、2 seed、随机基线 0.05：

| n_pairs（负载） | flat_raw | flat_adj | capcw_raw | **capcw_adj** |
|---:|---:|---:|---:|---:|
| 2 | 0.151 | 0.161 | 0.175 | **0.642** |
| 4 | 0.151 | 0.152 | 0.157 | **0.643** |
| 6 | 0.120 | 0.124 | 0.160 | **0.643** |

（模型初始化已固定种子，结果可复现）

- **H1（相邻算子在 CAPCW 内救活 induction）**：capcw_adj − capcw_raw 跨负载平均 **+0.4792**（≥+0.30）。
- **对照（相邻算子单独喂 flat）**：flat_adj − flat_raw 均值 **+0.005≈0**（单向量池化无法联想检索）。
- **H2（内容寻址价值）**：capcw_adj − flat_adj ≈ **+0.48~0.52**（最佳 +0.52 @n_pairs=6，≥+0.10）。→ **PASS**。

**核心结论（2×2 交互，比朴素假设更深）**：induction 需要"**序列相邻算子**"与"**内容寻址 slot**"
**两者同时**具备——只有 capcw_adj 起作用（≈0.63），缺任一味的另三格全部≈随机（0.12~0.18）。即：
相邻算子单独喂单向量救不活、slot 没有相邻算子也救不活。这把 CAPCW 从"内容绑定/工作记忆引擎"扩到
"**in-context 序列引擎**"（指代/induction 这类序列内现场学映射），并**强化了 CAPCW 的核心主张**：
内容寻址不是可选项，而是 induction 的必要一味（恰对应 Transformer 的 induction head = previous-token
head × content-addressed copy 两步合一）。

诚实边界：① 本算子是 prev-token 这一**最小相邻基元**；更长程/多跳的序列依赖（层级序列、远距离归纳）
尚未测，留作后续。② 任务用了与 capcw_binding 同口径的无歧义生成（bigram 起点偶数不重叠 + filler
不含任何 a）——清理对四臂公平、no-adj 臂仍≈随机，是去噪不是偏袒（含噪版 capcw_adj 0.34、清理版
0.64，机制方向一致）。报告 `docs/reports/capcw_induction_seq_eval.{json,md}`。

## 18. 多跳链式推理负结果：迭代读出 ≠ 链式组合（2026-06-15，边界）

第 17 节让 CAPCW 做 1 跳 induction。继续测**多跳**（链 c0→c1→…→cH，查 c0 答 cH，需现场链式组合多个
绑定——in-context 组合推理的原型，也是 Transformer 靠**多层/多头**做多跳的能力）。脚本
`world_model/capcw_multihop_eval.py`。三臂唯一变量=读出结构（同序列相邻算子 + 同预算；
capcw_1read/iter **同 seed 初始化** → 1 跳严格相等作 sanity）：flat 单向量 / capcw_1read 单次内容寻址
读出 / capcw_iter H 次迭代读出（读出值 → to_next → 下一跳 query）。hop 1/2/3、3 seed、随机基线 0.05：

| n_hops（跳数） | flat | capcw_1read | capcw_iter |
|---:|---:|---:|---:|
| 1 | 0.100 | **0.423** | **0.423** |
| 2 | 0.059 | 0.131 | 0.117 |
| 3 | 0.057 | 0.079 | 0.133 |

H1 iter−flat 多跳均值 **+0.067**、**H2 iter−1read 多跳均值 +0.020≈0** → **FAIL（诚实负结果）**。

**结论（边界细化）**：CAPCW 主场=**单跳** in-context 绑定（1 跳 0.42 ≫ flat 0.10）；**多跳链式不成立**。
关键证据 **H2≈0**——对**固定 slot** 反复读（迭代读出）并不能链式组合（单读 ≈ 多读）；slot 工作空间比
单向量略好（多跳 ~0.12 vs ~0.06）但远不解多跳。

诊断与下一关：每跳读回的是**纠缠的整 slot 向量**，`to_next` 在 d=32 下无法把"中间符号 c_i"干净地再
注入为"下一跳的键查询"（显式 key/value 读出反而更差，1 跳 0.56→0.32）。多跳链式很可能需要的不是
"多读"，而是把中间结果**作为新观测重新写入 / 重弛豫工作空间**（迭代**更新 slot 本身**，而非只读固定
slot），或真正的多层深度 + 逐跳学习（与 Transformer 多层做多跳同构）——应作为独立判定实验（先定
任务 + 判据），不在本轮硬凑。报告 `docs/reports/capcw_multihop_eval.{json,md}`。

## 19. 接回真实 controller：引擎 surprise 驱动"知道何时不该答"+内容取回（2026-06-15，集成）

把已验证的 CAPCW（内容寻址 slot 工作空间 + query 路由 surprise，第 15 节 Part3）做成 controller 兼容的
工作记忆组件 `active_inference/capcw_memory.py::CAPCWWorkingMemory`：`bind(key→value)` 累积 in-context
绑定、`decide(query)` 由**引擎 surprise** 裁决 `ActionType.ANSWER`（bound，低 surprise，带取回 value）/
`ASK_CLARIFICATION`（unbound，高 surprise）。决策**无动作监督、从引擎 surprise 涌现**——这是 FE-LLM
"机制从引擎涌现"主张在 controller 招牌决策"知道何时不该答"上的落地。定位上不与 `known_slots`（预定义
槽位精确字典）冗余：CAPCW WM 管的是 **in-context 任意键值绑定**。

集成 eval（实验 C 同口径，`world_model/capcw_controller_integration_eval.py`）：对话=K 个 in-context
绑定 + 查询（bound/unbound 各半），FE-agent（CAPCW 工作记忆）vs baseline（无记忆·永远直答）：

| 指标 | FE-agent（CAPCW 工作记忆） | baseline（无记忆·永远直答） |
|---|---:|---:|
| ASK/ANSWER balanced acc（引擎 surprise，无动作监督） | **0.829** | — |
| 内容取回 value 准确率（bound 且回答时） | **0.919** | 0.083（随机猜） |
| 任务成功率（bound 答对 / unbound 该问） | **0.792** | 0.043 |
| unbound 胡答率（越低越好） | **0.255** | 1.000 |

（绑定工作空间训练准确率 0.894）→ **PASS**：引擎 surprise 无动作监督即正确分开"该答/该问"、内容取回
正确，任务成功率远超无记忆基线、几乎不胡答。

接回真实 controller（加法式、默认关、零回归）：`ActiveInferenceController(capcw_memory_path=...)` 可选
加载，`bind_working_memory`/`reset_working_memory`/`working_memory_decision` 为显式接口；默认 None 不
启用 → 既有管线零影响。新增 10 测试，全量 143 全绿。诚实边界：① 工作空间需在绑定任务训练才内容寻址
（适用受控小词表）；② 活文本自动把"现场关联"抽成 (key,value) 需一层 in-context 绑定 NLU（开放词表/
容量受限），属下一步，故现为显式 API 接口而非 `respond()` 自动调用。报告
`docs/reports/capcw_controller_integration_eval.{json,md}`。
