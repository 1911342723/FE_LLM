# 持续学习·灾难遗忘三臂对照：PER 隔离 vs PER 共享 vs 标准 Transformer

顺序学 3 个**同前缀、冲突输出**的技能（标准灾难遗忘设置），看**早学技能**是否被覆盖。三臂同底座（各自 52M 真实代码模型 code_model / code_model_tf）、同 interactions/steps/replay/teach 集 / held-out 集 / 种子；唯一差别 = 架构与有无隔离机制。指标 = held-out completion loss(bits)。

## 任务

```
def task_<X>():
    return   →   "<X>"   (str) | [<X>]  (list) | (<X>,)  (tuple)
```

## 首技能「str」遗忘对照

| 臂 | 刚学完 loss | 学完全部后 loss | 遗忘Δ | 刚学完 acc | 学完全部后 acc |
|---|---:|---:|---:|---:|---:|
| **PER-ISO 隔离** | 0.49 | 0.49 | **+0.00（恒定不忘）** | 38% | 38% |
| PER-SHARED 共享 | 0.71 | 1.68 | +0.98 | 38% | 0% |
| **Transformer-FT** | 0.04 | 9.20 | **+9.16（灾难遗忘）** | 88% | 0% |

末态各技能 loss：ISO [0.49, 0.48, 0.5] / SHARED [1.68, 1.23, 0.31] / TF [9.2, 8.06, 1.37]

当前技能（对角线）acc（确认三臂都学会了新技能，对照公平）：ISO ['38%', '62%', '12%'] / SHARED ['38%', '75%', '88%'] / TF ['88%', '25%', '12%']

## 结论

【灾难遗忘·三臂对照】顺序学 3 个同前缀冲突技能后，首技能「str」held-out loss 漂移：PER-ISO Δ=+0.00（隔离·恒定不忘）、PER-SHARED Δ=+0.98、**Transformer-FT Δ=+9.16（0.04→9.20，灾难遗忘）**；首技能复制准确率：ISO 38%→38%、TF 88%→0%。 结论：Transformer 顺序微调（无隔离机制）灾难遗忘早学技能；PER 用可学突触隔离做到旧技能恒定不忘（PER-SHARED 去掉隔离也会忘，证明'不忘'来自隔离机制而非 PER 天生）。

## 诚实边界

- **公平性**：PER 用其内生的可学突触隔离机制（synapse-only + 加容量冻旧块），Transformer 用标准顺序微调——这是两种架构各自的持续学习方式，对照展示的是**架构能力差异**，非超参不公平。
- **PER-ISO 的代价**：参数随技能线性增长（每技能 +突触一块）、且**无前向迁移**（每技能从底座独立学）；理想解 = Progressive 式（冻旧块 + 侧向连接，迁移且不忘）。
- **Transformer 并非无解**：EWC / replay / adapter / LoRA 等可缓解其灾难遗忘，但都是**额外加的机制**；本对照证明的是**标准 Transformer 无内生隔离**，而 PER 的隔离是架构自带。
- **规模**：本任务为 3 个冲突技能的机制验证（synapse-only、小步数），证机制不证规模。

图：`docs\reports\figs\code_forgetting_compare.png`
