# FE-LLM 成长闭环可审计评估（candidate → confirmed）

- 判定：**PASS: 重复偏好晋升 confirmed，一次性事实维持 candidate**
- 记忆条目：3（confirmed 1 / candidate 2）

## 审计明细

| 文本 | 次数 | 会话数 | 置信 | 状态 |
|---|---|---|---|---|
| 记住我喜欢简短回答 | 3 | 3 | 1.0 | confirmed |
| 记住我住在北京 | 1 | 1 | 0.3333 | candidate |
| 记住我的生日是5月 | 1 | 1 | 0.3333 | candidate |

- 说明：短期 belief→长期 memory：单次=candidate，重复稳定=confirmed（可进入离线再训练）。
