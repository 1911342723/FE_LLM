# FE-LLM 任务领域识别 NLU（任务语料训练）

- 判定：**PASS: 任务领域 NLU 在留出 session 上有效**
- 领域数：9（appointment, delivery, flight, food, hotel, repair, restaurant, topup, train）
- 训练轮 9177 / 留出轮 3050
- 留出 session 领域 balanced acc：0.8565

- 说明：学习式领域识别 NLU 用任务语料训练（非合成模板），按 session 切分留出评测。
