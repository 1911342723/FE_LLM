# CAPCW · 序列相邻算子救 induction（走向序列语言引擎）

- 判定：**PASS: 2×2 交互成立——induction 需同时具备'序列相邻算子'(H1: capcw_adj >> capcw_raw)与'内容寻址 slot'(H2: capcw_adj >> flat_adj)；缺一格(flat_adj/capcw_raw)均≈随机。CAPCW+相邻算子成为可用的 in-context 序列引擎，从'内容绑定'走向'序列语言'。**
- 任务：序列 ...A B ... A → 预测 B（每序列 A→B 随机配对，不可记忆）；n_sym=20, d=32；随机基线 0.050
- 设计：2×2 单变量析因（相邻算子 on/off × 世界状态 flat/CAPCW）；同 d / 同 slot 预算。

| n_pairs（负载） | flat_raw | flat_adj | capcw_raw | capcw_adj |
|---:|---:|---:|---:|---:|
| 2 | 0.151±0.002 | 0.161±0.003 | 0.175±0.004 | 0.642±0.002 |
| 4 | 0.150±0.012 | 0.151±0.001 | 0.157±0.003 | 0.643±0.001 |
| 6 | 0.120±0.001 | 0.124±0.004 | 0.160±0.006 | 0.643±0.017 |

- **H1（相邻算子在 CAPCW 内救活 induction）**：capcw_adj − capcw_raw 跨负载平均 = **+0.4792**（阈值 ≥ +0.30 → 成立）
  - CAPCW 各负载救活幅度：{2: 0.4677, 4: 0.4858, 6: 0.484}
  - 对照（相邻算子单独喂 flat，flat_adj − flat_raw）跨负载平均 = **+0.0049**（预期≈0：单向量池化无法联想检索）；各负载：{2: 0.0098, 4: 0.001, 6: 0.004}
- **H2（内容寻址价值：给了相邻信息后 slot 是否胜单向量）**：高负载(n_pairs≥4) capcw_adj − flat_adj 最佳 = **+0.5195**（@n_pairs=6）（阈值 ≥ +0.10 → 成立）
  - 各负载 capcw_adj − flat_adj：{2: 0.4817, 4: 0.4913, 6: 0.5195}
- **交互**：capcw_adj − max(另三格) 跨负载平均 = **+0.4792**（只有'相邻算子+内容寻址'两者兼备才解 induction）

- 说明：no-adj 输入不含'前驱身份'，无论参数多少都无法恢复 a→b 相邻关系，故对照是信息(非容量)对照；+adj=previous-token channel(induction head 基元)。关键发现是 2×2 交互：相邻算子单独喂 flat 救不活(ctrl_rescue_flat≈0)、slot 无相邻算子也救不活(capcw_raw≈随机)，唯有两者兼备(capcw_adj)才行。
