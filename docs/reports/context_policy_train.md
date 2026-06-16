# FE-LLM 上下文感知 policy 训练 + 留出 session 端到端评测

- 判定：**PASS: 学习式 context-aware policy 在留出 session 上 belief 强有效**
- 训练轮 10777 / 留出轮 3615（750 个留出 session）
- 留出总体 balanced acc：1.0

## 歧义子集（同句多动作）
- 上下文感知：1.0
- 清空 belief 盲对照：0.5
- delta：0.5

- checkpoint：`checkpoints\active_inference\context_policy.pt`
- 说明：按 session 切分防泄漏；端到端逐轮用该轮 belief 预测动作。belief_cleared_blind=同模型清空 belief 的盲对照。
