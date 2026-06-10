# FE-LLM 预训练底座路线草案（能量解码与意图槽位接入预训练模型）

> 起因：翻译判定实验两连阴性——单向量意图 word-F1 0.0735、
> 槽位化意图 0.0832（仅 1.13x，M2 FAIL，见 docs/FE-LLM意图序列化架构草案.md 第 7 节）。
> 两次实验的辅助损失全部正常收敛，共同把瓶颈锁定在
> **8.9M 字符级模型从零训练的容量与初始化**，而非意图的表达结构。
> 本草案按意图序列化草案第 5 节预案，设计"预训练底座 + FE-LLM 机制层"的转线路线。
> 文献支撑：文本 EBM 的主流可行路线本来就不是从零替代 LM，
> 而是在预训练 LM 之上加能量（Deng et al. 2020; Bakhtin et al. 2020）。

## 1. 转线逻辑：换底座，不换思想

从零训练路线证伪的是"小模型能从零学出承载细粒度语义的意图空间"，
没有证伪 FE-LLM 的任何核心机制。转线后分工：

```text
预训练底座（外购能力）：字词语法、世界知识、跨语言对齐 —— 解决容量问题
FE-LLM 机制层（自有贡献）：结构化意图、能量递减解码、surprise/EFE 控制层、
                          显式 trace —— 解决"何时说/为何说/如何溯源"问题
```

### 1.1 不变量（无论接入深度如何，以下六要素不动摇）

| 要素 | 底座路线中的形态 |
|---|---|
| 能量解释 | 残余能量/覆盖能量由我们的 EnergyHead 显式计算，不冒充底座内部量 |
| 预测误差 | 控制层 PredictionError 不变；分槽 surprise 定位继承槽位草案 M4 目标 |
| 自由能平复 | EFE 行动选择层接口不变（BeliefState/policy/trace 全保留） |
| 主动推理 | controller 闭环不变，底座只替换感知与语言实现两端 |
| 可溯源生成 | 溯源只来自显式能量量与 InferenceTrace；底座 attention 仍只是"目" |
| 自我成长 | 三层成长边界不变；底座权重属"离线结构成长"层，默认冻结 |

### 1.2 必须诚实面对的叙事风险（判定纪律）

接入预训练底座后，"能力来自底座还是来自机制层"必须用消融回答，
否则就是借底座能力贴牌（文献地图判断三的延伸）：

- 每个判定实验必须带**同底座对照组**：同底座、同预算、
  去掉意图条件与能量打分的纯微调/纯解码版本；
- 机制层的增益只能主张"对照组与实验组的差值"，
  不得把底座本身的能力计入 FE-LLM 贡献；
- opus-100 等公开语料可能在底座预训练数据中，报告必须声明
  这一风险，并补一套新采样的验证集做交叉检查。

## 2. 三级接入路线（按改动深度分级，逐级判定）

### P1：冻结底座 + 外挂意图/能量头（首发，最快出判定）

```text
prompt → Backbone(frozen) → 隐状态 H_p
           ├─ IntentAdapter：learned queries 读 H_p → global_intent + K 槽位 + salience
           └─ 底座自回归提议 top-p 候选字
response 每步 i：
  EnergyHead：proj(h_i) 与 global/槽位算残余能量 + 覆盖能量（显式、可溯源）
  hybrid 选字：score = logP_backbone(w) − α·候选内归一化残余能量
```

- 只训练 IntentAdapter + EnergyHead（百万级参数），底座完全冻结；
- 训练目标沿用槽位草案 3.3 节（InfoNCE + approach + slot_div + coverage），
  铁律不变：意图只来自 prompt 侧，训练/推理同分布；
- hybrid 复合打分直接复用生成层已验证经验（α=1.0 起步，能量参与方式是
  "复合"不是"替代"）；
- 文献定位：外挂判别头引导解码是成熟路线（PPLM、FUDGE、residual EBM），
  FE-LLM 的差异点在于打分量是**结构化意图的残余/覆盖能量**且全程入 trace。

### P2：LoRA 注入意图条件（P1 通过后进入）

- 在底座 decoder 层加 LoRA + 槽位 cross-attention 适配器，
  让生成显式条件于 IntentState（不再只靠解码期打分）；
- 训练量上升一个量级，但意图对生成的控制从"事后筛选"变成"事中条件"；
- 判定点：相对 P1，意图编辑实验（改槽位 → 输出对应变化）成功率显著提升。

### P3：蒸馏回自有架构（中期，可选）

- 用底座当 teacher，蒸馏初始化 PER 架构的自有模型
  （保住"预测-误差弛豫"的机制研究线，呼应文献地图判断三：
  PER 的新意要用消融证明）；
- 仅当 P1/P2 证明机制层有效后才值得投入。

## 3. 模块设计（标准命名，新建子包不破坏现有原型）

```text
fe_llm/backbone_lm/
  __init__.py
  backbone.py          # PretrainedBackbone：HF 模型封装（hidden states + logits 暴露）
  intent_adapter.py    # IntentAdapter：底座隐状态 → IntentState（global/slots/salience）
  energy_head.py       # EnergyHead：解码隐状态 → 残余能量/覆盖能量（显式溯源量）
  hybrid_decode.py     # 复合打分解码 + 能量轨迹记录（接 InferenceTrace）
  slot_translation_p1_train.py / _eval.py   # N1 判定实验（口径与 M2 完全一致）
```

底座选型标准（N1 第一个交付物，候选先验：Qwen3-0.6B / Qwen2.5-0.5B 一类）：

- ≤1B 参数，单张消费级显卡可推理、可训头部/LoRA；
- 中英双语预训练，开放许可（Apache/MIT）；
- 提供 base 版（避免 instruct 对齐行为污染判定实验）。

## 4. 与主动推理控制层对接

- `fe_llm/embedding/` 工厂新增 backbone embedder：PerceptionEncoder 与
  IntentAdapter 共享同一底座，延续"控制层与生成层共享意图空间"的已验证设计；
- `BeliefState.intent_vector` 映射到 global_intent，维度经 proj 对齐，
  控制层代码零改动；
- **policy 重训纪律**（既有教训直接适用）：嵌入空间一换，PolicySelector 必须
  重训并跑全部场景验收；`encoder_kind` 门控继续防止 hash 回退向量错配融合；
- 能量量纲必然变化：所有"能量下降"门控继续用相对比较，且按第 7 节
  槽位草案的教训，明确各分量（全局/覆盖）谁承担生成进度解释。

## 5. 验证路线（每步可判定，对照组强制）

| 里程碑 | 内容 | 通过标准 |
|---|---|---|
| N1 | 底座选型 + P1 接入 + 翻译判定重测（200 句同口径） | word-F1 ≥0.3 且显著高于同底座无机制对照组 |
| N2 | 对话域接入：IntentChat 换底座，policy 重训 | realization_eval 口径不回退（answer 接入率 100%）；policy bal_acc ≥0.92；实验 B 5/5 |
| N3 | 能量信号有效性（文献地图实验 5 口径） | 残余能量与输出质量显著负相关；能量轨迹可区分熟悉/陌生输入 |
| N4 | 结构化意图全链路：分槽 surprise 定位 + 缺槽追问（继承 M4） | 实验 B/C 口径不回退；意图编辑实验成功率报告 |

N1 是本路线的判定实验：若冻结底座 + 机制层仍到不了 0.3，
说明问题在机制层设计本身，届时必须回到机制层重新审视，
而不是继续加大底座。

## 6. 风险

- **贡献归属混淆**：靠 1.2 节对照组纪律兜底，报告模板强制双列；
- **显存/算力**：P1 只训头部可控；P2 LoRA 需要梯度检查点与短序列预算；
- **底座行为入侵**：instruct 版会自带拒答/对齐行为，干扰 EFE 行动选择的
  判定——优先 base 版，对齐行为留给控制层；
- **数据泄漏**：公开平行语料可能在底座预训练集内，N1 报告必须声明并
  附新采样验证集结果；
- **机制层失效**：若 N1 对照组与实验组无显著差异，诚实记录阴性结果，
  按文献地图"最短研究路线"回收 PER/能量解释为独立研究点。

## 7. 文献锚点

- Deng et al., 2020, Residual Energy-Based Models for Text Generation——
  预训练 LM 残差上加 sequence-level energy 的主流路线背书；
- Bakhtin et al., 2020, Residual EBM——"先做能量修正器，别急着推翻 AR"；
- Dathathri et al., 2019, PPLM；Yang & Klein, 2021, FUDGE——
  外挂头引导冻结底座解码的成熟先例（FE-LLM 差异：打分量是结构化意图能量且入 trace）；
- Khandelwal et al., 2020, kNN-LM——记忆层后续接入的对照基线；
- Assran et al., 2023, I-JEPA——"语义 latent 能量"叙事的同源线。
