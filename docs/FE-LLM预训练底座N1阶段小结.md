# FE-LLM 预训练底座 N1 阶段小结

更新日期：2026-06-13

## 1. 当前结论

N1 已完成从 P1 到 P2 最小形态的连续验证：

```text
P1   : EnergyHead 事后 rerank
P1.5 : IntentLogitsAdapter logits bias
P2   : IntentResidualAdapter 最后一步 hidden residual
```

总判定：

> 三条最小外挂路线均未正式通过 N1。它们能影响输出，部分场景能提升 B 相对 A，但都不能在正式或更强对照中稳定证明真实 intent 优于随机/错配 intent。

这不是失败，而是把路线边界压清楚了：**FE-LLM 机制层不能只外挂在候选分数末端，也不能只改最后一步 hidden；如果要让 structured intent 真正控制生成，必须进入更深的 decoder 内部状态更新。**

## 2. 结果矩阵

| 阶段 | 机制 | 最好局部结果 | 正式/强对照结果 | 判定 |
|---|---|---:|---:|---|
| P1 | `EnergyHead` hybrid rerank | B=0.208（20 条，C=0.205） | B/C 不稳定 | 部分阳性，归因不足 |
| P1.5 | `IntentLogitsAdapter` logit bias | 20 条：A=0.161, B=0.172, C=0.142 | 200 条：A=0.182, B=0.158, C=0.175 | smoke 阳性，正式阴性 |
| P2 | `IntentResidualAdapter` 最后一步 hidden residual | random C：A=0.161, B=0.178, C=0.161 | mismatch/hard negative：B=C | 通用修正阳性，intent 归因阴性 |
| P2c | 单层 `IntentLayerHook` | layer0 smoke B=0.213；layer12 smoke B=0.188 | layer0 200：B=0.1806<A=0.1816；layer12 200：B=0.1772<C=0.1788 | smoke 阳性，正式阴性 |

## 3. 已确认的工程资产

- `PretrainedBackbone`：冻结底座封装；
- `IntentAdapter`：底座 hidden → structured intent；
- `EnergyHead`：显式 residual/coverage energy；
- `IntentLogitsAdapter`：intent-conditioned logit bias；
- `IntentResidualAdapter`：intent-conditioned hidden delta；
- P1/P1.5/P2 train/predict/eval 脚本；
- A/B/C 评估口径与 JSONL 报告链路；
- 多组可复现实验报告在 `docs/reports/`。

## 4. 关键经验

1. **训练损失下降不等于生成能力出现。** P1 的 ranking loss 能下降，但不改善生成。
2. **随机 intent 对照太弱。** 能压随机 C 不代表机制有效，mismatch C 才更接近归因纪律。
3. **外挂分数控制太浅。** rerank/logit bias 都容易变成通用扰动。
4. **最后一步 hidden residual 仍太浅。** P2 最小实现能修正底座，但 B/C 不分离。
5. **下一步必须更深。** 需要多步/多层 hidden 注入，或真正的 LoRA/cross-attention intent injection。

## 5. 下一阶段建议

进入 N1-P2b：

```text
多层/多步 IntentResidualAdapter
```

最小目标：

- 不训练全量底座；
- 在多个 decoder step 的 hidden 上注入 intent residual；
- 或在若干层加 LoRA-style intent adapter；
- 继续用 A/B/C，且 C 默认使用 mismatch intent；
- 先 20 条 smoke，B>A/C 后再 200 条正式。

停损条件：

- 当前单层 hook 已证明 smoke 可行但正式集不泛化；
- 下一步若继续底座路线，应训练真正层内 LoRA/多层注入，而不是推理时外挂 hook；
- 若多层注入仍失败，再回到 `IntentAdapter` 数据/目标，检查语义表达是否仍不足。
