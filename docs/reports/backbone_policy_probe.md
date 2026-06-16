# N2 step1 · 底座句向量做 policy 特征（离线对照 probe）

更新日期：2026-06-15

> 唯一变量 = 句向量来源（自建 IntentEncoder vs 冻结 Qwen2.5-0.5B，mean-pool 与 last-token 两种池化）。
> 同 split / 同 18 标量 / 同 TinyPolicyNet 超参；判定指标 = balanced accuracy。

## 配置

- 数据：`data\active_inference\policy_teacher.jsonl`（train 8946 / val 2236）
- 底座：`Qwen/Qwen2.5-0.5B`（hidden_size=896）
- 超参：hidden=128, epochs=80, seed=42, batch=64, lr=0.001, class_weights=True
- 判定边界：|delta| ≥ 0.02 记「明显」

## 结果

| 臂 | 句向量来源 | 输入维度 | val accuracy | val balanced_acc |
|---|---|---:|---:|---:|
| intent | IntentEncoder 8.9M (128d) | 146 | 0.9750 | 0.9566 |
| backbone_mean_full | Qwen2.5-0.5B mean-pool (896d) | 914 | 0.9575 | 0.9385 |
| backbone_mean_pca128 | Qwen2.5-0.5B mean-pool→PCA128 | 146 | 0.9647 | 0.9465 |
| backbone_last_full | Qwen2.5-0.5B last-pool (896d) | 914 | 0.9472 | 0.9346 |
| backbone_last_pca128 | Qwen2.5-0.5B last-pool→PCA128 | 146 | 0.9696 | 0.9496 |

## 判定

- **头条（best_backbone(full) vs intent）**：best_backbone 0.9385 − intent 0.9566 = delta **-0.0181** → **PARTIAL**
- backbone_mean_full vs intent：backbone 0.9385 − intent 0.9566 = delta **-0.0181** → PARTIAL
- backbone_mean_pca128 vs intent：backbone 0.9465 − intent 0.9566 = delta **-0.0100** → PARTIAL
- backbone_last_full vs intent：backbone 0.9346 − intent 0.9566 = delta **-0.0220** → FAIL
- backbone_last_pca128 vs intent：backbone 0.9496 − intent 0.9566 = delta **-0.0070** → PARTIAL

## 逐类召回（balanced_acc 分解）

- intent：answer=0.991, ask_clarification=0.898, retrieve=0.924, refuse=1.000, update_memory=0.969
- backbone_mean_full：answer=0.975, ask_clarification=0.856, retrieve=0.930, refuse=0.978, update_memory=0.954
- backbone_mean_pca128：answer=0.981, ask_clarification=0.869, retrieve=0.951, refuse=0.978, update_memory=0.954
- backbone_last_full：answer=0.963, ask_clarification=0.822, retrieve=0.941, refuse=0.971, update_memory=0.977
- backbone_last_pca128：answer=0.987, ask_clarification=0.873, retrieve=0.941, refuse=0.978, update_memory=0.969

## 结论

即便取最有利的池化方式，backbone 句向量与自建 IntentEncoder 仍只持平（best delta -0.0181）：冻结 0.5B 底座的通用句向量在 action 分类上无额外优势——自建 8.9M IntentEncoder 已吃满该任务，符合「瓶颈在任务不在表示规模」。N2 step1 不构成接入理由。 各对照：backbone_mean_full vs intent -0.0181(PARTIAL)；backbone_mean_pca128 vs intent -0.0100(PARTIAL)；backbone_last_full vs intent -0.0220(FAIL)；backbone_last_pca128 vs intent -0.0070(PARTIAL)。
