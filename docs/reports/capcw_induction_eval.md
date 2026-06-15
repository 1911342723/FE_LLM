# CAPCW · 真 in-context 语言机制：induction（归纳头）

- 判定：**FAIL: induction 上未明显胜**
- 任务：序列 ...A B ... A → 预测 B（每序列 A→B 随机配对，不可记忆）；n_sym=20, n_pairs=4, seq_len=16, d=32；随机基线 0.050

| 序列聚合结构 | induction accuracy |
|---|---:|
| flat（单向量均值池化） | 0.1178 |
| CAPCW（slot 工作空间） | 0.1035 |

- delta（CAPCW − flat）= **-0.0142**

- 说明：induction head 是 Transformer in-context learning 的基石；每序列 A→B 随机配对(不可记忆)，必须现场绑定。唯一变量=序列聚合结构(单向量 vs slot 工作空间)。这是 Part1 指出的'真 in-context 语言任务'。
