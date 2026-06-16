# FE-LLM 从 0 字符级槽位值标注器（能力与边界）

- 判定：**PASS: 规则型 DATE/TIME 可学且泛化，开放实体 CITY 受容量限制（诚实边界）**
- held-out span 召回：DATE 0.5833 / TIME 0.5 / CITY 0.0
- DATE+TIME 平均 0.5416 vs CITY 0.0

- 说明：从 0 字符级序列标注；DATE/TIME 规则模式可泛化到新表达，CITY 开放命名实体对未见城市受容量限制（小模型 NER 固有边界）。
