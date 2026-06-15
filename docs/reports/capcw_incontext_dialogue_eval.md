# CAPCW 活文本闭环 · 绑定 NLU → 工作记忆 → 引擎 surprise 驱动 controller

- 判定：**PASS: 活文本闭环成立——绑定 NLU 把现场关联喂工作记忆，引擎 surprise 在 controller 里正确驱动 ANSWER(取回 value)/ASK，且不劫持寒暄。CAPCW 引擎接回真实 controller 活文本路径。**
- 工作空间绑定训练准确率：0.9330

## 脚本会话实录（grounded 回答=取回内容入生成；surprise=引擎 query 路由）

| 用户输入 | 动作 | 回答（grounded） | 取回 | surprise |
|---|---|---|---|---:|
| 记住会议室是B302 | answer | 好的，已记住会议室是B302 | — | — |
| 项目代号对应X9 | answer | 好的，已记住项目代号是X9 | — | — |
| 会议室是多少 | answer | 会议室是B302 | B302 | 0.182 |
| 项目代号是什么 | answer | 项目代号是X9 | X9 | 0.104 |
| 门禁卡是多少 | ask_clarification | 信息还不够，请补充你想让我具体做什么。 | — | 1.000 |
| 记住门禁卡是8821 | answer | 好的，已记住门禁卡是8821 | — | — |
| 门禁卡是多少 | answer | 门禁卡是8821 | 8821 | 0.057 |
| 你好 | answer | 你好，我在。 | — | — |

## 聚合指标（多段随机会话）

- 决策 balanced acc（bound→ANSWER / unbound→ASK，引擎 surprise）：**1.0000**
- 内容取回 value 准确率：**1.0000**
- 寒暄劫持率（越低越好）：**0.0000**（n=40）

- 说明：绑定 NLU 高精度规则触发（记住/对应/设为/等于/的{属性}是 + 查询词），裸'X是Y'与寒暄不触发；工作记忆决策由引擎 query 路由 surprise 涌现（无动作监督）；value 由取回的符号 id 映回字符串；回答 grounded（取回内容入生成，可溯源）；主动推理：问未绑定→高 surprise→追问，用户补绑定后再问→surprise 下降→grounded 回答（自由能平复）。
