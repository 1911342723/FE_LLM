# FE-LLM 预训练底座 P2 执行方案

更新日期：2026-06-13

## 1. 进入 P2 的原因

P1 与 P1.5 已完成完整实验链路：

- P1：`IntentAdapter + EnergyHead + hybrid rerank`；
- P1.5：`IntentLogitsAdapter` 在 logits 阶段加 intent bias。

结果：

```text
P1 正式归因：未过，真实 intent 不能稳定优于随机 intent。
P1.5 smoke：B > A/C。
P1.5 200 条正式：A=0.182, B=0.158, C=0.175，未过。
```

结论：

> 外挂 rerank / logits bias 能影响输出，但不足以让真实 intent 稳定控制生成。下一步需要让 intent 进入底座 decoder 的内部状态更新，而不是只在候选分数末端施加偏置。

这正对应预训练底座路线草案中的 P2：LoRA 意图条件注入。

## 2. P2 核心假设

P2 的假设：

> 若 structured intent 在 decoder hidden state 更新过程中参与计算，真实 intent 对生成的控制会比随机 intent 更稳定。

工程形式：

```text
h_t' = h_t + LoRA_intent(h_t, IntentState)
logits_t = LMHead(h_t')
```

最小实现不直接改 Qwen 权重，而是在冻结底座输出 hidden states 后增加一个轻量 residual adapter：

```text
IntentResidualAdapter:
  candidate_hidden + IntentState -> delta_hidden
  adapted_hidden = candidate_hidden + gamma * delta_hidden
  logits = LMHead(adapted_hidden)
```

这比 P1.5 更深一层：P1.5 只给候选 logit 一个标量 bias；P2 改变候选 hidden state，再经底座 `lm_head` 读出 logits。

## 3. 模块边界

新增模块：

- `IntentResidualAdapter`：生成 intent-conditioned hidden delta（已新增）；
- `slot_translation_p2_train.py`：训练 residual adapter（已新增）；
- `slot_translation_p2_predict.py`：A/B/C 预测（已新增）；
- 继续复用 `slot_translation_p1_eval.py`。

不做：

- 不训练全量 Qwen；
- 不先上复杂 LoRA 注入所有层；
- 不改变 N1 的 A/B/C 判定口径。

## 4. 训练目标

### 4.1 语言目标

同一前缀下，真实 intent adapted hidden 经 `lm_head` 后应提高 gold token 概率：

```text
CE(lm_head(h_gold + delta_true), gold_next_token)
```

### 4.2 intent 对比

同一 gold token，真实 intent 的 logit 应高于错配 intent：

```text
logit_gold(true_intent) > logit_gold(wrong_intent)
```

### 4.3 保守正则

限制 `delta_hidden` 范数，避免 adapter 破坏底座语言能力：

```text
lambda_norm * ||delta_hidden||^2
```

## 5. 对照组

继续 A/B/C：

- A：纯底座；
- B：真实 intent + residual adapter；
- C：随机/错配 intent + residual adapter。

可选 D：

- D：无 intent residual adapter，排除“只是多了一个通用 adapter”。

## 6. 判定标准

Smoke：

- 20 条验证样本；
- B > A/C；
- B-C ≥ 0.02。

正式：

- 200 条验证样本；
- B ≥ 0.3；
- B > A/C；
- 报告 disagreement、输出多样性、样例。

停损：

- 若 B<C，说明 intent 注入无效；
- 若 B>C 但 B<A，说明有归因但伤语言；
- 若 B>A/C 但低于 0.3，扩大训练/调 gamma；
- 若 P2 仍不能提升到 0.3，P2 记为部分阳性，考虑进入 P3 或回到更强的训练目标。

## 7. 实施顺序

1. 新增 `IntentResidualAdapter` 并单测形状、delta 范数。（已完成）
2. 新增 P2 train dry-run 与参数测试。（已完成）
3. 新增 P2 predict dry-run。（已完成）
4. 128 对、1 epoch、20 条验证 smoke。（已完成：B/C 未分离）
5. 修正 C/D 对照，区分通用 adapter 与真实 intent 控制。
6. 若 B>A/C，跑 200 条正式评估。

## 8. Smoke 结果（2026-06-13）

128 对、1 epoch、20 条验证样本：

```text
A = 0.161
B = 0.172
C = 0.172
mismatch control: A=0.161, B=0.172, C=0.178
contrast2 random: A=0.161, B=0.178, C=0.161
contrast2 mismatch: A=0.161, B=0.178, C=0.178
hard negative mismatch: A=0.161, B=0.167, C=0.167
multistep hard negative: A=0.161, B=0.167, C=0.167
```

结论：

- residual adapter 可改善底座输出；
- 但 B/C 完全不分离，说明当前收益更像通用 adapter；
- mismatch control 下 C 仍高于 B，说明错配 intent 没有被压低；
- 强化 contrast 可以压随机 intent，但压不住相近错配 intent；
- hard negative 后 B/C 仍同分；
- 多步训练后 B/C 仍同分，且 disagreement 极低；
- 结论：shallow 外挂 residual adapter 已到瓶颈；
- 若继续 P2，应真正 hook/LoRA 到 decoder 层内，或回到 `IntentAdapter` 训练目标检查 structured intent 表达。
