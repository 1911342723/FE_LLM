# FE-LLM 预训练底座 P1.5 执行方案

更新日期：2026-06-13

## 1. 为什么进入 P1.5

P1 已经跑通冻结底座 + `IntentAdapter` + `EnergyHead` + hybrid rerank 的完整链路，但没有通过 N1 归因纪律：

- A 组：纯底座；
- B 组：真实 intent + FE rerank；
- C 组：随机/错配 intent + FE rerank。

当前可复现实验结果为：

```text
A = 0.161
B = 0.198
C = 0.201
```

结论：P1 证明了 FE 机制能影响输出，也能让 B 相对 A 有提升，但没有稳定证明真实 intent 优于随机 intent。问题不在链路，而在作用方式：**事后 energy 扣分太晚、太弱，容易变成通用 reranker，而不是真实 intent 控制器。**

## 2. P1.5 核心假设

P1.5 的假设：

> structured intent 不应该只在 token 选完前一刻做 energy 惩罚，而应在 logits 阶段直接给候选 token 加 bias。

工程形式：

```text
score(w) = logP_backbone(w)
         + beta * bias_intent(w | IntentState)
         - alpha * normalized_energy(w | IntentState)
```

其中：

- `logP_backbone`：冻结底座的语言能力；
- `bias_intent`：P1.5 新增 `IntentLogitsAdapter` 输出；
- `energy`：保留 P1 的可溯源能量轨迹，但不再独自承担控制；
- `beta`：intent bias 权重；
- `alpha`：energy rerank 权重。

## 3. 模块边界

已完成：

- `IntentAdapter`：底座 hidden states → `IntentState(global, slots, salience)`；
- `EnergyHead`：显式残余/覆盖能量；
- `IntentLogitsAdapter`：候选 hidden + `IntentState` → logit bias。

下一步新增：

- `slot_translation_p15_train.py`：训练 logits adapter（已新增）；
- `slot_translation_p15_predict.py`：A/B/C 预测，支持 `beta` 扫描（已新增）；
- 沿用 `slot_translation_p1_eval.py`：评估 JSONL，不重复造指标。

## 4. 训练目标

P1.5 先训练 logits adapter，不动底座。

### 4.1 候选级 gold ranking

对同一前缀：

```text
bias(gold_next_token, true_intent)
  > bias(top-k_negative_token, true_intent)
```

这直接服务于“下一步选哪个 token”。

### 4.2 intent 对比

对同一个 gold token：

```text
bias(gold, true_intent)
  > bias(gold, wrong_or_random_intent)
```

这直接服务于 B > C 的归因纪律。

### 4.3 能量辅助

P1 的 `EnergyHead` 不删除，但降为辅助：

- 继续记录 residual/coverage trace；
- 可作为弱正则；
- 不再把能量惩罚当作唯一控制量。

## 5. 对照组

P1.5 仍沿用 A/B/C：

- A：底座原生生成；
- B：真实 intent + logits adapter；
- C：随机/错配 intent + logits adapter。

可选 D 组：

- D：无 intent 的 trainable scalar/head，用于排除“只是多了一个通用 reranker”。

## 6. 判定标准

烟测：

- 20 条验证样本；
- B > A 且 B > C；
- B/C 差值至少 > 0.02，才进入 200 条正式评估。

正式 N1：

- 200 条验证样本；
- B 组 mean word-F1 ≥ 0.3；
- B 显著高于 A/C；
- 输出 disagreement rate、residual/coverage descent、样例报告。

停损：

- 若 B 不能稳定高于 C，说明 intent 没有真正进入生成控制；
- 若 B 高于 C 但低于 A，说明机制有归因但伤语言能力；
- 若 B 高于 A/C 但低于 0.3，继续扩大训练或调 `beta/alpha`；
- 若扩大训练后仍低于 0.3，P1.5 记为部分阳性，转 P2 LoRA/cross-attention 注入。

## 7. 实施顺序

1. 新增 P1.5 train 脚本，只训练 `IntentLogitsAdapter`。（已完成）
2. 新增 P1.5 predict 脚本，支持 `beta`。（已完成）
3. 先跑 128 对、1 epoch、20 条验证样本。（已完成：B>A/C）
4. 若 B > A/C，再跑 256/512 对与 200 条正式评估。
5. 写入 `docs/reports/slot_translation_p15_eval.*`。
6. 根据结果决定继续 P1.5、转 P2，或回到机制设计。

## 8. Smoke 结果（2026-06-13）

128 对、1 epoch、20 条验证样本：

```text
beta=1.0: A=0.161, B=0.172, C=0.142
beta=0.5: A=0.161, B=0.166, C=0.156
256 pairs, beta=1.0: A=0.161, B=0.165, C=0.156
200-val formal, beta=1.0: A=0.182, B=0.158, C=0.175
```

结论：

- P1.5 首次稳定满足 B > A/C；
- 说明 `IntentLogitsAdapter` 比 P1 的事后 energy rerank 更能体现真实 intent 控制；
- 但 20 条 smoke 结果没有泛化到 200 条正式评估；
- 200 条正式评估中 B 低于 A/C，不足以通过 N1；
- 256 对复测仍保持 B>A/C，但不如 128 对配置；
- 下一步转 P2：LoRA / 层内 intent-conditioned injection。
