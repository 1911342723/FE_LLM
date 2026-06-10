# FE-LLM 生成层评测：EnergyDecoder 接入 answer 动作

可溯源生成的证据：answer 由能量递减解码产生，控制层信念意图注入为生成目标，
逐字残余能量轨迹随 trace 返回，未收敛的生成被门控拒绝并可溯源回退。

- answer 动作数：11/12
- energy_decoder 接入率：100%（11/11，回退 0）
- 信念意图注入（belief_mixed）：11/11
- 能量整体下降通过：11/11
- 平均能量降幅：0.5%
- 单调下降步占比：95.2%
- 选字决策分歧率（hybrid vs 纯 logit argmax）：13.8%

## 样例明细

| prompt | action | engine | 意图来源 | 能量降幅 | 单调步比 | 输出 |
|---|---|---|---|---:|---:|---|
| 我有点累 | `answer` | `energy_decoder` | belief_mixed | 0.5% | 100% | 我也是，尤其睡多了的时候 |
| 晚安 | `answer` | `energy_decoder` | belief_mixed | 0.3% | 100% | 晚安 |
| 我今天很开心 | `answer` | `energy_decoder` | belief_mixed | 0.5% | 88% | 我帮你用手机叫。 |
| 你吃饭了吗 | `ask_clarification` | `template` | - | - | - | 信息还不够，请补充你想让我具体做什么。 |
| 周末打算去爬山 | `answer` | `energy_decoder` | belief_mixed | 0.4% | 100% | 我也想你 |
| 我喜欢这个故事的开头 | `answer` | `energy_decoder` | belief_mixed | 0.5% | 85% | 我喜欢粗眉的，羡慕你怎么办 |
| 最近工作压力好大 | `answer` | `energy_decoder` | belief_mixed | 0.4% | 100% | 我们也是工作的 |
| 我想休息一下 | `answer` | `energy_decoder` | belief_mixed | 0.7% | 100% | 早安，今天起得真早。 |
| 今天加班到很晚 | `answer` | `energy_decoder` | belief_mixed | 0.6% | 100% | 下周一上午如何？ |
| 我朋友过生日送什么好 | `answer` | `energy_decoder` | belief_mixed | 0.6% | 83% | 我也是，但是我不太方便。 |
| 刚看完一部电影 | `answer` | `energy_decoder` | belief_mixed | 0.6% | 100% | 你们那边约会啊 |
| 外面好像下雨了 | `answer` | `energy_decoder` | belief_mixed | 0.5% | 92% | 我也是，但是我不太方便。 |
