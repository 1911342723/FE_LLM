# B2c · 真实任务对话(CrossWOZ) belief 对回复内容的 headroom

- 判定：**PARTIAL: belief 对回复内容有正向 headroom**
- 数据：`data\crosswoz\train.json.zip`（5012 对话 / 33300 样本 / 33 个 领域·槽位 标签）
- 领域未明示子集占比：0.2449（验证集 2021 条）

> 任务=预测系统主告知 (领域·槽位)；唯一变量=是否加 belief（活跃领域）；同 MLP/split/train。

## 总体 balanced accuracy
- 盲：0.8165 → 感知：0.8977（delta +0.0812）

## 领域未明示子集（跟进句，headroom 关键）
- 盲：0.6707 → 感知：0.825（delta **+0.1544**）

## belief 价值地图（B2 系列合并结论）
- 动作类型选择（B2）：belief 无 headroom（−0.02）
- 状态/领域追踪（B2b）：belief 强 headroom（未明示子集 +0.19）
- 回复内容 grounding（B2c）：见上

- 说明：回复内容代理=系统主告知 (领域·槽位)；唯一变量=belief(活跃领域)。跟进句里槽位常在 utterance、领域不在，故 belief 补领域→帮系统说对内容。与 B2/B2b 合成 belief 价值地图：动作类型(无)、状态/领域追踪(强)、回复内容 grounding(本实验)。
