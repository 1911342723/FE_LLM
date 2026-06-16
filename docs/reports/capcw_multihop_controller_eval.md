# CAPCW 多跳链式工作记忆接回 controller · decode→re-embed（潜在 CoT）

- 判定：**PASS: 多跳 CoT 链式取回接回 controller 成立——start 绑定→ANSWER + 链式取回链尾 value、未绑定→ASK（balacc 0.889）；多跳链尾取回 0.912，比只读 1 跳的baseline 任务成功率高 +0.3920（**链式组合的增量**）。对内多步推理 + 知道何时不该答 + 可溯源 CoT trace 在 controller 决策框架内成立。cot−latent(held-out 多跳取回)=+0.2282（显式解码中间符号的作用）。**
- 设置：n_sym=20, n_pairs=5, d=32, ask_threshold=0.5, n_eval=1500；随机基线 0.050
- 训练准确率（链尾,H=3）：cot=0.8863 / latent=0.7150

| n_hops | cot 任务成功 | cot 链尾取回 | cot balacc | baseline(单跳) 成功 | latent 链尾取回 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.877 | 0.973 | 0.889 | 0.877 | 0.683 |
| 2 | 0.862 | 0.933 | 0.892 | 0.443 | 0.663 |
| 3 | 0.839 | 0.890 | 0.884 | 0.474 | 0.704 |

- 多跳(H≥2)：cot 链尾取回 **0.912**；cot−baseline 任务成功 **+0.3920**（链式组合增量）；cot−latent 链尾取回 **+0.2282**（显式解码中间符号的作用）。
- ASK/ANSWER balanced acc（均值，首跳 surprise 驱动，无动作监督）：**0.8886**

- 说明：决策从引擎首跳路由 surprise 涌现（无动作监督）；链式由 decode→re-embed（潜在 CoT）实现，每跳解码的中间符号=可溯源 CoT trace；baseline 只读 1 跳故够不到链尾（H≥2），凸显链式组合的增量。
