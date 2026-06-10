# FE-LLM 生成层评测：EnergyDecoder 接入 answer 动作

可溯源生成的证据：answer 由能量递减解码产生，控制层信念意图注入为生成目标，
逐字残余能量轨迹随 trace 返回，未收敛的生成被门控拒绝并可溯源回退。

- answer 动作数：11/12
- energy_decoder 接入率：100%（11/11，回退 0）
- 信念意图注入（belief_mixed）：11/11
- 能量整体下降通过：11/11
- 平均能量降幅：34.7%
- 单调下降步占比：95.5%

## 样例明细

| prompt | action | engine | 意图来源 | 能量降幅 | 单调步比 | 输出 |
|---|---|---|---|---:|---:|---|
| 我有点累 | `answer` | `energy_decoder` | belief_mixed | 41.9% | 100% | 没事，我知道你是无心。 |
| 晚安 | `answer` | `energy_decoder` | belief_mixed | 14.0% | 50% | 哈喽 |
| 我今天很开心 | `answer` | `energy_decoder` | belief_mixed | 17.6% | 100% | 嗨 |
| 你吃饭了吗 | `retrieve` | `template` | - | - | - | 这个问题需要外部信息或实时数据，我需要先检索后再回答。 |
| 周末打算去爬山 | `answer` | `energy_decoder` | belief_mixed | 42.1% | 100% | 没事，我知道你是无心。 |
| 我喜欢这个故事的开头 | `answer` | `energy_decoder` | belief_mixed | 38.5% | 100% | 没事，我不急 |
| 最近工作压力好大 | `answer` | `energy_decoder` | belief_mixed | 42.9% | 100% | 没事，别自责。 |
| 我想休息一下 | `answer` | `energy_decoder` | belief_mixed | 18.8% | 100% | 嗨 |
| 今天加班到很晚 | `answer` | `energy_decoder` | belief_mixed | 42.5% | 100% | 没事，别自责。 |
| 我朋友过生日送什么好 | `answer` | `energy_decoder` | belief_mixed | 39.4% | 100% | 不太忙。 |
| 刚看完一部电影 | `answer` | `energy_decoder` | belief_mixed | 42.2% | 100% | 没事，我知道你是无心。 |
| 外面好像下雨了 | `answer` | `energy_decoder` | belief_mixed | 41.9% | 100% | 没事，我知道你是无心。 |
