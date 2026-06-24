# PER 随交互成长证明（synapse-only · 冻结 backbone）

任务：复制规则 `def make_<X>(): return Widget("<X>")`，teach 8 个 X、held-out 8 个没教过的 X。只更新可学突触 S（786k 参数）。

- teach loss 0.8667→0.2512（降 0.616）
- **held-out loss 0.899→0.3777（降 0.521）**
- held-out 复制准确率 0%→75%
- control（无关代码）1.085→1.6351（Δ+0.550）

## 结论

✅ 成长成立：held-out（没教过的同规则实例）loss 随交互下降 +0.52 bits、复制准确率 0%→75%——只动突触就把规则刻进结构记忆并泛化到新实例。

对照（无关代码）loss 变化 +0.550 bits：有一定遗忘（#2 抗遗忘弱，如实）。

图：`docs\reports\figs\per_code_growth.png`
