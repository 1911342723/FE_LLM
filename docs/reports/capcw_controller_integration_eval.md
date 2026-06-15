# CAPCW 接回 controller · 引擎 surprise 驱动 ASK/ANSWER + 内容取回

- 判定：**PASS: CAPCW 引擎接回 controller 决策成立——引擎 surprise 无动作监督即正确分开该答(bound)/该问(unbound)，且取回 value 正确；任务成功率远超无记忆基线、几乎不胡答。**
- 设置：n_keys=10, n_vals=12, K=5, ask_threshold=0.5, n_eval=2000；value 随机基线 0.083
- 工作空间绑定训练准确率：0.8941

| 指标 | FE-agent（CAPCW 工作记忆） | baseline（无记忆·永远直答） |
|---|---:|---:|
| ASK/ANSWER balanced acc（引擎 surprise，无动作监督） | 0.8290 | — |
| 内容取回 value 准确率（bound 且回答时） | 0.9188 | 0.0833（随机猜） |
| 任务成功率（bound 答对 / unbound 该问） | 0.7925 | 0.0430 |
| unbound 胡答率（越低越好） | 0.2551 | 1.0000 |

- 说明：FE-agent=CAPCW 工作记忆(引擎 surprise 裁决 ASK/ANSWER + 内容取回，无动作监督)；baseline=无 in-context 记忆、永远直答(bound 随机猜、unbound 胡编)。决策从引擎 surprise 涌现，对应 controller 招牌'知道何时不该答'，且 value 取回对应 B2c 的'内容/grounding'价值。
