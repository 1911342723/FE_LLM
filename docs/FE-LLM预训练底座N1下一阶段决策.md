# FE-LLM 预训练底座 N1 下一阶段决策

更新日期：2026-06-13

## 1. 当前分叉

P1/P1.5/P2/P2b 已经把“外挂控制”路线压到了边界：

- 能影响输出；
- 能局部提升；
- 但不能稳定证明真实 intent 优于随机/错配 intent。

下一步有两条路：

1. **继续加深生成端注入**：真正 hook/LoRA 到 decoder 层内；
2. **先诊断 intent 表达**：检查 `IntentAdapter` 产出的 structured intent 是否本身能区分不同源句。

## 2. 决策

先走第 2 条：诊断 `IntentAdapter` 表达。

原因：

- 如果 intent 表达本身不分离，LoRA/hook 只会把不稳定信号注入更深；
- 当前 B/C 不分离可能有两种根因：生成端太浅，或 intent 表达太弱；
- 先做诊断成本低，能决定后续是修 `IntentAdapter`，还是继续做更深注入。

## 3. 诊断目标

新增 `intent_adapter_diagnostic.py`，检查：

- `global_intent` 的两两 cosine 分布；
- 每条样本最近邻是否是自身或语义相近样本；
- 随机/错配 intent 与真实 intent 的距离是否足够大；
- slot salience 熵是否塌缩；
- 不同 checkpoint 的 `IntentAdapter` 是否表现一致。

## 4. 判定

若诊断显示：

- intent 两两距离很小；
- nearest neighbor 混乱；
- slot salience 近似均匀且无结构；

则下一步回到 `IntentAdapter` 训练目标。

若诊断显示：

- intent 表达有明显分离；
- B/C 不分离主要发生在生成端；

则下一步进入真正 LoRA/hook 层内注入。

## 5. 下一步

1. 新增 intent diagnostic 脚本；
2. 先对当前 P2/P1.5 checkpoint 跑 200 条或 50 条样本；
3. 写诊断报告；
4. 再决定修 intent 还是修生成端。

## 6. 诊断结果（2026-06-13）

使用 `slot_translation_p15_logits_128_seed42.pt`，50 条验证样本：

```text
mean_offdiag_cosine = 0.8562
max_offdiag_cosine  = 0.9715
min_offdiag_cosine  = 0.3143
nearest_unique_count = 25 / 50
slot_salience_entropy = 2.0794 ≈ ln(8)
```

结论：

- `global_intent` 彼此高度相似；
- `slot_salience` 几乎完全均匀，说明槽位显著性没有学出结构；
- B/C 不分离不只是生成端太浅，intent 表达本身也不足。

下一步决策：

> 先修 `IntentAdapter` 训练目标，再继续更深生成注入。

优先方向：

- 源句区分损失；
- batch 内 hard negative contrast；
- slot salience 稀疏/低熵约束；
- 重新诊断 intent 表达后，再进入 P2b/LoRA。

## 7. IntentAdapter 分离训练初试（2026-06-13）

训练：

```text
512 pairs, 1 epoch
zh/en contrast + salience entropy + slot diversity
```

诊断：

```text
before: mean_offdiag_cosine=0.8562, slot_entropy=2.0794
after : mean_offdiag_cosine=0.8376, slot_entropy=1.4995
```

结论：

- salience entropy 明显下降，槽位显著性不再完全均匀；
- global intent 仍高度相似，分离不足；
- 单纯 zh/en 对齐不够，需要更强的源句区分 / hard negative 目标。

## 8. Source Spread 训练结果（2026-06-13）

训练：

```text
512 pairs, 3 epochs
zh/en contrast + source spread + salience entropy + slot diversity
```

诊断：

```text
original: mean_offdiag_cosine=0.8562, slot_entropy=2.0794
1 epoch : mean_offdiag_cosine=0.8376, slot_entropy=1.4995
spread  : mean_offdiag_cosine=0.4688, slot_entropy=0.0410
```

结论：

- source spread 明显降低了全局 intent 相似度；
- slot salience 已从均匀变为极尖锐，可能需要后续调权重；
- 现在可以回到 P2b/P2 生成端复测，判断更分离的 intent 是否带来 B>C。

## 9. 回到 P2 的复测结果（2026-06-13）

使用 `intent_adapter_spread_512_e3.pt` 作为外部 IntentAdapter，P2 residual adapter 重新训练并用 mismatch C 组评估：

```text
A = 0.161
B = 0.172
C = 0.161
```

结论：

- B 再次高于 A/C；
- 说明“先修 intent 表达”方向有效；
- 下一步应基于 spread intent 扩大到 200 条正式评估；
- 同时需要注意 slot entropy=0.041，槽位可能过尖，后续可调低 entropy 约束。

200 条正式评估：

```text
A = 0.1816
B = 0.1834
C = 0.1823
```

判定：

- B 只比 C 高 0.0011，不足以主张机制有效；
- disagreement 约 0.5%，说明最后一步 residual 对底座生成影响仍很弱；
- intent 表达修复是必要但不充分，下一步需要更强的生成端注入。
