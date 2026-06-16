# FE-LLM 上下文动作选择：belief 机制是否带来 headroom

- 判定：**PASS: belief 机制在歧义子集显著胜出（headroom 真实存在）**
- 样本：14013，歧义占比：0.5591，字表：97

## 总体 balanced accuracy
- baseline（只看当前句）：0.9184
- belief-aware（看 belief）：1.0

## 歧义子集 balanced accuracy（关键：机制 headroom）
- baseline：0.7731
- belief-aware：1.0
- delta（belief - baseline）：0.2269

- 说明：唯一变量=能否访问 belief（已知城市/有无 pending 请求）。歧义轮次同句不同标签，只看当前句必翻车。
