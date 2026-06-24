# PER 解决 Transformer 痛点：一个自带"持续学习不遗忘 + 知识可定位编辑 + 自动成长"的知识模块

> 更新日期：2026-06-24
>
> 本报告把一条完整的研究链（6 个可复跑实验）收口为一句话结论：
> **在从 0 训练的因果 PER 语言模型（SeqEnergyNet，真实 52M 字符级 Python 代码模型）之上，
> 一个"LoRA 式隔离知识模块"同时回答了标准 Transformer 的两大痛点——灾难性遗忘、黑箱不可编辑——
> 并进一步做成端到端自动成长系统、用 Progressive 式侧向连接拿到前向迁移。**
> 两大能力同源于一个架构选择：**每个知识一个独立低秩块、冻结隔离**。

所有结论均为公平对照（同底座 / 同协议 / 同种子，各臂取合理超参）、带真实数据、可一键复跑，负面与边界如实列出。

---

## 0. 北极星与战场

- 北极星：证明能做一款"新语言模型"，从架构上解决 Transformer 的真实痛点（而非拼规模/博学度）。
- 战场选择（Transformer 真正缺、PER 可补的）：**① 灾难性遗忘**（持续学习）、**② 黑箱不可编辑**（可溯源）。
- 不竞争：世界知识 / 生成博学度（容量纪律，小模型物理装不下，主动不拼）。

Transformer 痛点 × 本研究的回答：

| Transformer 痛点 | 本研究的回答 | 证据实验 |
|---|---|---|
| 灾难性遗忘（学新忘旧，参数共享覆盖） | LoRA 式隔离模块：既学得好(83%)又不忘(Δ0) | B · C · D |
| 黑箱·知识不可定位编辑（参数纠缠） | 知识可定位精准擦除（特异性 0.77，旁观 0 影响） | A |
| 静态权重（学新要重训、需人工干预） | surprise 门控自动成长 + 推理路由（端到端） | Router |
| 隔离的代价：无前向迁移 | Progressive 式侧向连接拿前向迁移（+34%）且仍不忘 | E |

---

## 1. 实验 B — 灾难性遗忘三臂对照（发现痛点）

**问题**：顺序学 3 个"同前缀、冲突输出"的技能，早学技能会不会被覆盖？

| 臂 | 首技能学完后 | 遗忘Δ |
|---|---|---|
| PER-ISO（可学突触隔离） | acc 恒定 | **Δ+0.00 不忘** |
| PER-SHARED（共享） | 退化 | Δ+0.98 部分忘 |
| **Transformer-FT（标准顺序微调）** | **acc 88%→0%** | **Δ+9.16 灾难遗忘** |

**结论**：标准 Transformer 顺序学习会灾难性地毁掉旧技能（这是参数共享的内生痛点）；PER 用隔离机制不忘；PER-SHARED 去掉隔离也会忘——证明"不忘"来自**隔离**。
报告：`reports/code_forgetting_compare.md`；图：`reports/figs/code_forgetting_compare.png`。

---

## 2. 实验 C — 参数效率诊断（诚实修正）

**质疑**："学得多记得少，是不是很多无效参数、架构不够好？"

同一个学新技能任务，冻结 backbone 只训不同模块（各取最佳 lr，held-out 24 实例）：

| 可训练模块 | 参数 | held-out acc |
|---|---:|---:|
| LoRA r=32（低秩内容） | 653K | **92%** |
| LoRA r=8（低秩内容） | **163K** | 83% |
| full（全参上限） | 52M | 79% |
| synapse（位置门控） | 786K | 75% |
| head（输出层） | 939K | 33% |

**诚实修正**：先前"synapse 把参数用错地方"的判断被推翻——synapse 用对 lr 达 75%，**并非无效参数**（之前低分是 lr 未调最优；方法论教训：归因架构前先排除超参）。但**内容相关低秩适配（LoRA）参数效率确实更高**（1/5 参数即超过 synapse）。"学得多记得少"的真因是 **stability-plasticity**（共享参数学新覆盖旧），非参数无效。
报告：`reports/code_param_efficiency.md`；图：`reports/figs/code_param_efficiency.png`。

---

## 3. 实验 D — LoRA 式隔离模块：既学得好又不忘（解决痛点①）

合并"实验 B 的隔离不忘"+"实验 C 的 LoRA 高效"：每技能一组低秩 LoRA 块、冻旧块。

| 臂 | 学会程度 | 首技能遗忘Δ |
|---|---:|---:|
| **LoRA-ISO（本模块）** | **83%** | **+0.00** |
| synapse-ISO | 71% | +0.00 |
| Transformer-FT | —（88%→0%） | +9.16 |
| LoRA-SHARED | 67% | +4.65 |

**结论**：LoRA-ISO 用每技能 163K 低秩参数，**同时拿到高效学新(83%，比 synapse 隔离高 12 点)与隔离不忘(Δ0)**；Transformer 学得快但灾难遗忘；LoRA-SHARED（共享）会忘——再次证明"不忘来自隔离"。
报告：`reports/code_lora_isolation.md`；图：`reports/figs/code_lora_isolation.png`。

---

## 4. 实验 A — 知识可定位编辑（解决痛点②）

让模型同时掌握 3 个 base 不会的独立知识，再"外科式擦除其中一个"，量目标被擦 vs 旁观被误伤：

| 臂 | 编辑特异性(目标降−旁观降) | 目标降 | 旁观降 |
|---|---:|---:|---:|
| **LoRA-ISO（隔离模块）** | **+0.77** | 77% | **0%** |
| Transformer-full（共享） | +0.00 | 92% | 92% |
| PER-full（共享，诚实对照） | +0.00 | 58% | 58% |

**结论**：LoRA-ISO 卸载目标块即**精准擦除、旁观零影响**；Transformer/PER 共享参数擦目标必殃及旁观。PER-full 与 Transformer-full 同样殃及——证明"可定位编辑"来自**隔离结构**，与"不忘"同源。
报告：`reports/code_knowledge_editing.md`；图：`reports/figs/code_knowledge_editing.png`。

---

## 5. 实验 Router — 端到端成长系统（让模块可用）

| 能力 | 结果 |
|---|---|
| surprise 门控自动成长 | 知识流 make→get→make→tag→get → 自动长 3 块、重复的**复用不重复长**（全对✅）|
| 推理路由（无人指定） | 能量/置信度 **94%**，接近触发词上限 100% |
| 隔离不忘 | 各知识用正确块平均 **75%** 保持 |

**结论**：LoRA-ISO + surprise 门控 + 路由 = **端到端持续成长系统**（自动判断该长则长、不忘旧、按需取用），无人显式指挥。
报告：`reports/code_router_growth.md`；图：`reports/figs/code_router_growth.png`。

---

## 6. 实验 E — Progressive 式侧向连接：拿前向迁移（补隔离的短板）

隔离类方法的共性代价是"无前向迁移、每列从零学"。借鉴 Progressive Networks：新列侧向连接冻结的旧列。

| 技能(学习顺序) | LoRA-ISO 刚学完 acc | Progressive 刚学完 acc |
|---|---:|---:|
| a（第1个，无旧列） | 6% | 6% |
| b（第2个） | 25% | 38% |
| c（第3个，可借 a+b） | 6% | **62%** |

**结论**：少步数下 Progressive 借旧列特征获得**前向迁移 +34%**（越往后越强），且**仍不忘（Δ0）**。诚实：迁移在**步数/数据受限时显现**，预算充足后两者都学会、迁移消失（符合 Progressive 加速早期学习的特性）。
报告：`reports/code_progressive.md`；图：`reports/figs/code_progressive.png`。

---

## 7. 综合结论：一个新知识模块的能力矩阵

| 能力 | 标准 Transformer | 本模块（LoRA 式隔离 + Progressive + 门控路由） |
|---|---|---|
| 持续学习不遗忘 | ❌ 灾难遗忘(88%→0%) | ✅ Δ0（隔离的数学保证） |
| 高效学新（参数效率） | full 52M 才 79% | ✅ 163K → 83% |
| 知识可定位编辑/擦除 | ❌ 殃及旁观(特异性 0) | ✅ 特异性 0.77（旁观 0 影响） |
| 自动判断该不该学（成长触发） | ❌ 需人工 | ✅ surprise 门控自动 |
| 学过的按需取用（路由） | — | 🟡 能量路由 94%（触发词 100% 兜底） |
| 前向迁移 | ✅（共享但会忘） | ✅ +34%（Progressive，且不忘） |

**一句话**：这些能力**同源于一个架构选择**——"每个知识一个独立低秩块、冻结隔离 + 侧向连接 + surprise 门控"。它正是标准 Transformer（一锅端的共享参数黑箱）结构上给不了的。

---

## 8. 诚实边界

- **规模**：全部为机制验证（3 技能、held-out 16~24 实例、字符级代码模型），证机制不证规模。
- **隔离共性代价**：参数随知识线性增长（可低秩压缩）；学习化路由（能量 94%）仍弱于触发词上限（100%）——学习化路由是开放问题，触发词/小分类器可作可靠兜底。
- **前向迁移**：仅在步数/数据受限时显著；充足预算下消失。
- **不与通用 LLM 拼**世界知识 / 生成博学度（容量纪律）。
- **诚实修正**：研究过程中"synapse 参数无效"的假设被实验 C 推翻并修正——保留这一修正以体现方法论（归因架构前先排除超参）。

---

## 9. 复现

```bash
pip install -r requirements.txt
# 训练底座（PER + Transformer 对照，各 52M）：
python -m fe_llm.energy_lm.training.code_train --arch per --hours 4.5 --dim 768 --depth 12
python -m fe_llm.energy_lm.training.code_train --arch transformer --dim 768 --depth 7 --max-steps 78000
# 六个对照实验：
python -m fe_llm.energy_lm.evaluation.code_forgetting_compare       # B 灾难遗忘
python -m fe_llm.energy_lm.evaluation.code_param_efficiency_eval    # C 参数效率诊断
python -m fe_llm.energy_lm.evaluation.code_lora_isolation_eval      # D LoRA式隔离·不忘
python -m fe_llm.energy_lm.evaluation.code_knowledge_editing_eval   # A 知识可定位编辑
python -m fe_llm.energy_lm.evaluation.code_router_growth_eval       # Router 端到端成长
python -m fe_llm.energy_lm.evaluation.code_progressive_eval --interactions 6  # E 前向迁移
```
