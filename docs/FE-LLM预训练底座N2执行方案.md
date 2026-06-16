# FE-LLM 预训练底座 N2 执行方案（主动推理 + 底座）

更新日期：2026-06-13

## 1. 为什么转 N2

翻译注入路线（P1 / P1.5 / P2 / P2c）已经探到边界：

- 纯底座 A 组翻译 word-F1 ≈ 0.182；
- 各种机制 B 组正式集都在 0.177~0.183，和底座同水平；
- 根因：翻译任务“源句几乎完全决定译文”，intent 没有自由度，底座自己就把分数占满，机制层没有发挥空间。

而 FE-LLM 真正被验证有价值的是**主动推理控制层**：

- 实验 C：FE-agent 任务成功率 100% vs 永远直接回答的 baseline 22%；
- action selection（回答/追问/检索/拒答/记忆）是有巨大决策空间的地方。

所以 N2 把底座用在刀刃上：**不是用底座去硬证翻译，而是用底座增强 action selection / 多轮 surprise / 记忆**。

## 2. N2 step1：底座句向量做 policy 特征（离线对照 probe）

### 现状

- `PerceptionEncoder` 用 8.9M `IntentEncoder`（或 hash 回退）把 prompt 编码成句向量；
- `build_policy_feature_vector` = 句向量 base + 18 个标量特征；
- `TinyPolicyNet` 在该特征上分类 5 个 action；
- 现有基线：balanced accuracy ≈ 0.929（IntentEncoder 特征）。

### 改动

唯一变量：把句向量来源从 8.9M `IntentEncoder` 换成冻结 `Qwen2.5-0.5B` 的 mean-pooled hidden state。

- 复用同一份 `policy_teacher.jsonl`；
- 复用 `split_samples`（seed=42, val_ratio=0.2），划分完全一致；
- 复用 `build_policy_feature_vector` 的 18 个标量特征；
- 复用 `TinyPolicyNet` 结构与训练超参；
- 只换句向量 → 公平对照。

### 不做

- 不改 `controller` / `PerceptionEncoder`（零风险，先离线 probe）；
- 不接入主流程，直到 probe 证明底座特征更好。

## 3. 判定标准

- 主指标：balanced accuracy（类不平衡，answer 占多数）；
- 通过：backbone 特征 balanced accuracy 明显高于 0.929 基线；
- 部分：持平（说明底座句向量对 action 分类无额外优势，但也不差）；
- 失败：明显低于 0.929（说明 mean-pool 句向量不如小模型 intent 向量）。

## 4. 后续

- step1 通过 → N2 step2：把 backbone 句向量接入 `PerceptionEncoder`（`encoder_kind="backbone"`），policy 重训，跑 realization_eval / 实验 B / 实验 C 不回退；
- step1 失败 → 记录阴性，检查是否需要用 backbone 末 token / 特定层而非 mean-pool，或回到机制层。

## 5. 当前已落地

- `fe_llm/active_inference/experiments/backbone_policy_probe.py`：N2 step1 离线对照脚本（默认 dry-run，`--run` 真跑）。一次前向同时取 mean-pool 与 last-token 两种句向量，各跑「全维」+「PCA 维度匹配」两条对照，与 intent 基线同 split/同 18 标量/同 TinyPolicyNet 超参公平对比。

## 6. step1 结果（2026-06-15，已执行）

环境：torch 2.10+cu128（CUDA）、transformers 4.57.6、Qwen2.5-0.5B 本地缓存。
数据：`policy_teacher.jsonl`，train 8946 / val 2236（seed=42, val_ratio=0.2）。
超参：hidden=128, epochs=80, batch=64, lr=1e-3, class_weights=on。

| 臂 | 句向量来源 | 输入维度 | val balanced_acc | vs intent |
|---|---|---:|---:|---:|
| intent | 自建 IntentEncoder 8.9M | 128+18 | **0.9566** | — |
| backbone_mean_full | Qwen2.5-0.5B mean-pool | 896+18 | 0.9385 | −0.0181 (PARTIAL) |
| backbone_mean_pca128 | mean-pool→PCA128 | 128+18 | 0.9465 | −0.0100 (PARTIAL) |
| backbone_last_full | Qwen2.5-0.5B last-token | 896+18 | 0.9346 | −0.0220 (FAIL) |
| backbone_last_pca128 | last-token→PCA128 | 128+18 | 0.9496 | −0.0070 (PARTIAL) |

头条判定（给 backbone 最强机会 = 全维里最好的一条 vs intent）：delta **−0.0181 → PARTIAL（持平偏负）**。
报告：`docs/reports/backbone_policy_probe.{json,md}`。

判定：**4 种 backbone 变体（mean/last × 全维/PCA）全部未跑赢自建 IntentEncoder**，最好的也差 −0.007。即便：
- 给了 last-token（因果 LM 更合理的句表示）；
- 给了 PCA 维度匹配（排除「维度更高=参数更多才赢」的混淆，反而是 PCA 版更接近，说明 896 维全维对 tiny MLP 是噪声/过拟合负担）；

底座通用句向量在 action 分类上仍无额外优势。

## 7. 决策与下一步

- step1 **未通过**（无 PASS），按第 4 节「不接入主流程」——**不进 step2**（不把 backbone 句向量接入 `PerceptionEncoder`）。
- 根因（第三次印证同一教训）：action 分类本身已被 8.9M 小编码器吃满（balanced_acc ~0.957，无 headroom），近饱和任务上「更强表示」没有发挥空间——与「翻译被底座填满分数空间」「V2-M1 单向量吃满小标签任务」同理。**瓶颈在任务不在表示规模。**
- 诚实边界：仅试了最后一层（hidden_layer=-1）的 mean/last 池化；中间层（特定层）未穷尽。但鉴于任务饱和 + 反过扫纪律，继续层扫描收益低、不做。
- 推论（指导后续真正用底座的场景）：要检验底座价值，必须放到**有 headroom 的任务**（生成质量 / 开放理解 / 更难的多轮推理），而非已被小模型饱和的 action 分类。这把 N2 的「底座用在刀刃上」收敛到一个更准的命题：刀刃不是 action 分类的句向量，而是小模型容量真正不够的环节。

## 8. step1b 结果：有 headroom 任务上的表示对照（2026-06-15，已执行）

承接第 7 节推论，把同一「只换表示」对照搬到一个**有 headroom 的域内任务**——9 领域任务 NLU（`task_nlu_eval` 记录 balanced acc ~0.857，flight/train 等相邻领域措辞重叠而未饱和）。脚本 `backbone_taskdomain_probe.py`，按 session 切分（train 9177 / val 3050）。

| 表示臂 | val balanced_acc |
|---|---:|
| char-BoW（现基线） | **0.8565** |
| 自建 IntentEncoder 8.9M | 0.8387 |
| Qwen mean-pool (896d) | 0.8537 |
| Qwen last-token (896d) | 0.8559 |
| Qwen last-token→PCA128 | 0.8565 |

对照：intent−charbow −0.0179、best_backbone−charbow −0.0006、best_backbone−intent +0.0173（均 PARTIAL）。报告 `docs/reports/backbone_taskdomain_probe.{json,md}`。

判定与洞察：
- **即便在有 headroom 的任务上，更强句向量也没用**——所有表示挤在 ~0.84–0.857，backbone 没超过最朴素的 char-BoW。
- 关键洞察：**不是所有 headroom 都能被「更强表示」吃掉**。本任务的 headroom 来自领域措辞固有歧义（需更多上下文/数据），而非表示弱；与 belief headroom（缺的是 belief/槽位信息，换上就 0.49→1.0）是不同性质的 headroom。判定一个杠杆有没有用，要先诊断 headroom 的**来源**。
- 旁证：自建 IntentEncoder 句向量在本任务上反而略低于 char-BoW（−0.018）——8.9M 编码器的压缩丢了部分领域判别的表层信息，char-BoW 反而保留。

## 9. N2 总结论（2026-06-15）

- step1（饱和 action 分类）+ step1b（有 headroom 的领域 NLU）两面夹证：**冻结 Qwen2.5-0.5B 的句向量在 FE-LLM 的控制/理解任务上都没有杠杆**——饱和任务无 headroom 可用，本类 headroom 任务的瓶颈又不在表示。
- 底座真正的强项（世界知识、生成流畅度）恰恰是 FE-LLM 按容量纪律**主动不竞争**的方向。因此 N2「把底座用在刀刃上」在 FE-LLM 现有 scope 内**找不到刀刃**。
- 决策：**底座线再次收敛/封存**（翻译方向已封存，现 action 分类 + 领域理解也排除）。主线回到自建 v2 控制闭环深做。底座若未来再用，应放到明确由"知识/流畅度容量"决定的环节，并预先声明那不属于 FE-LLM 机制收益。
