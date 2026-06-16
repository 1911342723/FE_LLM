# 分层预测编码 · 组合泛化裁决实验

- 判定：**FAIL: 分层连理论主场(组合泛化)都没赢扁平——本规模下分层是过度设计，应封存**
- 任务：compare（compare=A<B/==/> ; modadd=(A+B)%N），未见 (A,B) 组合泛化；唯一变量=分层机制(relax_steps)。
- 配置：N=6, L=8, filler=10, test_combos=10, intent_dim=64, relax=5, seeds=3

| 臂 | seen (训练组合) | unseen (未见组合) |
|---|---:|---:|
| flat (relax0) | 1.000±0.000 | 0.652±0.083 |
| hierarchical (relax5) | 1.000±0.000 | 0.637±0.123 |
| hier_full concat (relax5) | 1.000±0.000 | 0.600±0.082 |

- 头条：best_hier_unseen − flat_unseen = **-0.0142**

- 说明：唯一变量=分层机制(relax_steps)；seen=训练组合留出样本、unseen=未见组合。组合泛化是分层/结构被理论认为不可替代的好处，故为公平裁决（非凑 headroom）。
