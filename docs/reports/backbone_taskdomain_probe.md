# N2 step1b · 有 headroom 任务上的句向量表示对照（9 领域任务 NLU）

更新日期：2026-06-15

> 唯一变量 = utterance 表示（charbow / 自建 IntentEncoder / 冻结 Qwen2.5-0.5B mean·last）。
> 同分类器结构 / 同训练超参 / 同 session 切分；判定指标 = balanced accuracy。
> 对照动机：step1 在已饱和的 action 分类上 backbone 无增益；本实验换到有 headroom 的任务验证表示价值。

## 配置

- 语料：`data\dialogue\teacher_task_oriented.jsonl`（9 领域：appointment, delivery, flight, food, hotel, repair, restaurant, topup, train）
- 切分：按 session，train 9177 / val 3050（seed=42）
- 底座：`Qwen/Qwen2.5-0.5B`（hidden=896）
- 分类器：Linear→ReLU→Linear(hidden=128)，epochs=250, lr=0.005, class_weights=on
- 判定边界：|delta| ≥ 0.02

## 结果

| 臂 | 表示来源 | 输入维度 | val accuracy | val balanced_acc |
|---|---|---:|---:|---:|
| charbow | char-BoW (213d) | 213 | 0.8574 | 0.8565 |
| intent | IntentEncoder 8.9M | 128 | 0.8400 | 0.8387 |
| backbone_mean_full | Qwen2.5-0.5B mean-pool (896d) | 896 | 0.8544 | 0.8537 |
| backbone_mean_pca128 | Qwen2.5-0.5B mean-pool→PCA128 | 128 | 0.8554 | 0.8545 |
| backbone_last_full | Qwen2.5-0.5B last-pool (896d) | 896 | 0.8574 | 0.8559 |
| backbone_last_pca128 | Qwen2.5-0.5B last-pool→PCA128 | 128 | 0.8570 | 0.8565 |

## 判定

- **intent vs charbow**：delta **-0.0179** → **PARTIAL**
- **best_backbone(full) vs charbow**：delta **-0.0006** → **PARTIAL**
- **best_backbone(full) vs intent**：delta **+0.0173** → **PARTIAL**

## 结论

在有 headroom 的 9 领域任务上，backbone 与自建 IntentEncoder 仍持平（delta +0.0173）：表示不是该任务的瓶颈（领域措辞重叠需更多上下文/数据，而非更强句向量）。 参照：charbow=0.8565、intent=0.8387、best_backbone=0.8559；intent−charbow -0.0179（PARTIAL）、backbone−charbow -0.0006（PARTIAL）。
