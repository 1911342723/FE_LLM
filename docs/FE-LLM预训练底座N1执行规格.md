# FE-LLM 预训练底座 N1 执行规格

> 目的：把“预训练底座 + FE-LLM 机制层”的第一步做成可编码、可判定、可对照的实验，而不是直接把底座能力包装成 FE-LLM 能力。

## 1. N1 判定问题

N1 只回答一个问题：

> 在冻结预训练底座的前提下，外挂结构化意图与显式能量头，是否能在翻译/短生成任务上带来可测增益？

通过标准沿用路线草案：

- 主指标：opus-100 zh→en 未见 200 句 `mean word-F1 >= 0.3`；
- 归属指标：必须显著高于同底座无机制对照组；
- 溯源指标：每步保留 `logP_backbone`、残余能量、hybrid score 与选字分歧；
- 风险声明：公开语料可能进入底座预训练集，报告需附新采样验证集。

## 2. P1 机制层接口

本轮新增 `fe_llm/backbone_lm/`，先落不依赖具体 HuggingFace 模型的机制骨架：

```text
Backbone(frozen) hidden states H
  ├─ IntentAdapter(H) -> IntentState(global_intent, intent_slots, slot_salience)
  ├─ EnergyHead(decoder_hidden, IntentState)
  │    ├─ residual_energy = ||h_i - global_intent||
  │    └─ coverage_energy = Σ_k salience_k · min_{j<=i} ||h_j - slot_k||
  └─ hybrid_decode:
       score(w) = logP_backbone(w) - α · normalized_residual_energy(w)
```

约束：

- `IntentAdapter` 与 `EnergyHead` 是 FE-LLM 贡献；底座只提供 hidden states 与候选 token logP；
- `slot_reader` 的 attention 只作为读取/路由，不作为解释；
- 可解释性来自显式能量量、覆盖度、salience 和 trace；
- `alpha=1.0` 作为起点，后续必须扫参并报告分歧率。

## 3. 对照组

N1 至少保留三组：

| 组别 | 底座 | FE 机制 | 解码 |
|---|---|---|---|
| A | frozen | 无 | 底座原生 greedy/top-p |
| B | frozen | IntentAdapter + EnergyHead | hybrid score |
| C | frozen | 只加随机/打乱 intent | hybrid score 负对照 |

可主张的 FE-LLM 增益只来自 B 相对 A/C 的差值。

## 4. 当前已落地

- `IntentAdapter`：从底座 hidden states 读出 `global_intent + intent_slots + slot_salience`；
- `EnergyHead`：计算残余能量与前缀覆盖能量；
- `hybrid_decode`：固化 `logP - α·候选内归一化能量` 的选字公式；
- `PretrainedBackbone`：封装真实 causal LM，延迟导入 `transformers`，P1 默认冻结底座；
- N1 首发底座：`Qwen/Qwen2.5-0.5B`（详见 `docs/FE-LLM预训练底座N1选型.md`）；
- `slot_translation_p1_train.py`：P1 训练骨架，默认 dry-run；显式 `--run` 后用正负配对能量排序训练 adapter/head；
- `slot_translation_p1_predict.py`：P1 预测骨架，按 A/B/C 生成 `slot_translation_p1_predictions.jsonl`；
- `slot_translation_p1_eval.py`：P1 评估骨架，读取 A/B/C 预测 JSONL，输出 N1 判定报告；
- 环境检查：`python -m fe_llm.backbone_lm.slot_translation_p1_train --check-env` 已确认训练数据、`transformers` 与 CUDA 可用；
- 极小样本真实运行：`--limit 32 --batch 4 --epochs 1 --run` 已跑通，`rank_loss=3.1931`，输出 `checkpoints/backbone_lm/slot_translation_p1_head.pt`；
- 预测+评估烟测：`--limit 1 --max-new 8 --top-k 4 --run` 已生成 A/B/C 预测并跑通 eval；当前 B 组 `word-F1=0.25`，低于阈值且不优于 A/C，仅作链路验证；
- mini-run 训练：`--limit 256 --batch 4 --epochs 1 --run` 已跑通，`rank_loss=1.5253`，checkpoint 已更新；
- mini-run 评估阴性：20 条验证样本下，当前排序损失无法带来生成收益；候选评分修正后 B=0.153、C=0.149，3 epoch 继续训练后 B=0.142、C=0.156；
- 候选级 ranking loss 烟测：B 绝对分数可升到 0.208，但 B/C 分离不稳定；下一步需加入真实 intent vs 随机/错配 intent 对比损失；
- intent contrast + seed 复测：A=0.161、B=0.198、C=0.201，P1 rerank 提升了 B 相对 A，但未通过 B>C 归因纪律；
- P1.5 启动：新增 `IntentLogitsAdapter`，从事后 energy rerank 转向 intent-conditioned logit bias；
- `tests/test_backbone_lm.py`：覆盖形状、覆盖能量单调性、hybrid 选字分歧、底座封装输出契约。

## 5. 下一步实现顺序

1. 将 `IntentLogitsAdapter` 接入 P1.5 训练和预测。
2. 用同一 A/B/C 口径复测 P1.5，要求 B 稳定优于 A/C。
3. 若 N1 未过 0.3 或 B 不显著优于 A/C，停止加大底座，回到机制层重新审视。
