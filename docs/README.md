# FE-LLM 文档

> **2026-07-19 核心重做：**项目正在从 attention-like 的旧 PER 原型回到“显式自由能下降即推理”这条
> 原始主线。新核心、数学契约、已完成验证与边界见 **[自由能核心重做.md](自由能核心重做.md)**。
> 下文的 SeqEnergyNet / LoRA 实验保留为历史基线，不再代表最终核心定义。
>
> 首项三种子裁决已通过：完整弛豫 `100%` vs 0-step `13.4%`，结构 surprise AUROC
> `0.992±0.002`；详见 [free_energy_sequence_eval.md](reports/free_energy_sequence_eval.md)。
> 第二项“穷则变”闭环也已通过：未知结构自动触发生长 `100%`、已知流误触发 `0%`、
> 旧 logits 零变化、新通路学习 `100%`、最低能路由 `97.7%`；详见
> [free_energy_growth_eval.md](reports/free_energy_growth_eval.md)。

> **因果 PER 语言模型 · 从 0 训练字符级 Python 代码模型**
>
> 以「状态趋稳 / 自由能最小化」为第一性原理的非自回归语言模型 **SeqEnergyNet**
>（因果预测-误差-弛豫 PER + 可学突触基底 #2，**非 Transformer 底座**）。在同参数 /
> 同 token 预算下对标标准 Transformer，并系统验证泛化、可溯源、后天成长。

## 核心结论（均为可复跑脚本 + 报告）

| 命题 | 结论 | 报告 |
|---|---|---|
| 同参数 / 同 token 对标标准 Transformer | PER **val_bpc 1.0365** < Transformer **1.1949**；语法合法率 29% vs 21% | [code_per_vs_transformer.md](reports/code_per_vs_transformer.md) |
| 真泛化（非记忆） | 未见分片 bpc 1.083 ≈ 训练 1.002（gap +0.08） | [code_per_vs_transformer.md](reports/code_per_vs_transformer.md) |
| 架构 scaling 对标 Transformer | opus100 字符 5 档完整 PER 5/5 优于阉割；大规模反超 TF | [lm_scaling_eval.md](reports/lm_scaling_eval.md) |
| 可学突触 #2 可溯源 | 剪某规则突触通路 → 该规则精准崩（1.00→0.00），干预特异性≈1.0 | [per_synapse_proof_eval.md](reports/per_synapse_proof_eval.md) |
| 真实语言突触收益 | 中文聊天 LM held-out ppl 完整 < 阉割 **3/3 种子** | [per_synapse_lm_eval.md](reports/per_synapse_lm_eval.md) |
| 突触规模趋势 | 0.31M→2.06M 完整全低于阉割 **8/8 种子** | [per_synapse_scaling_eval.md](reports/per_synapse_scaling_eval.md) |
| 后天成长（synapse-only） | held-out 复制规则 0%→75%（须经验回放，naive 单样本发散） | [per_code_growth.md](reports/per_code_growth.md) |
| 真成长不覆盖（加容量 + 隔离） | 旧技能 loss 学后续**恒定**（数学保证不遗忘）vs 共享突触灾难遗忘 | [per_code_growth_isolated.md](reports/per_code_growth_isolated.md) |

可视化在 [reports/figs/](reports/figs/)：逐字惊奇 / 跨层 ‖ε‖ / 各层 η / 可学突触 S 热图 / 内容路由 g 六联图、突触通路热力图、scaling 曲线。

## 专题：PER 解决 Transformer 两大痛点

> 完整报告见 **[PER解决Transformer痛点.md](PER解决Transformer痛点.md)**

在真实 52M 代码模型之上，一个"LoRA 式隔离知识模块"同时回答 Transformer 的两大痛点，并做成端到端成长系统：

| 痛点 / 能力 | 标准 Transformer | 本模块 | 实验 |
|---|---|---|---|
| 持续学习不遗忘 | ❌ 88%→0% | ✅ Δ0、学会 83% | B/C/D |
| 知识可定位编辑 | ❌ 殃及旁观(特异性0) | ✅ 特异性 0.77、旁观 0 影响 | A |
| 自动成长 + 路由 | ❌ | ✅ surprise 门控自动长块、路由 94% | Router |
| 前向迁移（且不忘） | ✅ 但会忘 | ✅ +34% 且 Δ0 | E |
| 真实库 API 知识上验证 | — | ✅ 不忘100% / 编辑+1.00 / import路由100% | Real |

7 个对照实验脚本：`code_forgetting_compare` / `code_param_efficiency_eval` / `code_lora_isolation_eval` / `code_knowledge_editing_eval` / `code_router_growth_eval` / `code_progressive_eval` / `code_real_library_eval`。

> **新会话快速接续**：先读 **[项目现状与下一步.md](项目现状与下一步.md)**（交接文档：现状/文件地图/边界/下一步）。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（codeparrot-clean 清洗去重 GitHub Python，落盘 data/code/python_corpus.txt）
python -m fe_llm.energy_lm.data.prepare_code --target-mb 200

# 3. 冒烟自检（小配置验证显存 / 跑通）
python -m fe_llm.energy_lm.training.code_train --smoke

# 4. 从 0 训练 PER 代码模型
python -m fe_llm.energy_lm.training.code_train --arch per --hours 4.5 --dim 768 --depth 12

# 5. 标准 Transformer 对照（同 token 量口径）
python -m fe_llm.energy_lm.training.code_train --arch transformer --dim 768 --depth 7 --max-steps 78000

# 6. 采样生成
python -m fe_llm.energy_lm.training.code_train --sample --arch per --prompt "def quicksort(arr):"
```

## 评测与复现

```bash
python -m fe_llm.energy_lm.evaluation.code_ab_eval               # PER vs Transformer A/B 对照
python -m fe_llm.energy_lm.evaluation.code_generalization_eval   # seen vs 未见分片泛化
python -m fe_llm.energy_lm.evaluation.code_trace_viz             # 可溯源六联图
python -m fe_llm.energy_lm.evaluation.code_growth_eval           # 后天成长（synapse-only）
python -m fe_llm.energy_lm.evaluation.code_growth_isolated_eval  # 真成长不覆盖（加容量 + 隔离）
python -m fe_llm.energy_lm.evaluation.per_synapse_proof_eval     # 可学突触 #2 可溯源铁证
python -m fe_llm.energy_lm.evaluation.per_synapse_lm_eval        # 真实聊天 LM 突触消融
python -m fe_llm.energy_lm.evaluation.lm_scaling_eval            # 架构 scaling 对标 Transformer
```

## 包结构

```
fe_llm/
├── energy_lm/         # 因果 PER 语言模型（SeqEnergyNet）
│   ├── models/        #   seq_net (SeqEnergyNet) / transformer_lm (对照) / tokenizer
│   ├── data/          #   prepare_code (代码语料) / corpus / teacher_gen
│   ├── training/      #   code_train (主入口) / chat_train / seq_train
│   ├── evaluation/    #   code_* / per_synapse_* / lm_scaling 等评测
│   ├── generation/    #   生成相关
│   ├── diagnostics/   #   能量坍缩 / 塌缩诊断
│   └── demos/         #   code_web / growth_web 等网页 demo
└── config.py          # 设备探测 + 教师模型配置（读 .env）
```

## 核心机制

**SeqEnergyNet = 因果 PER（预测-误差-弛豫）**：每个位置向其它位置发预测，按「突触电导」
（可学突触基底 #2，经验刻高 = 低阻通路）汇聚预期，用预测误差驱动弛豫到稳定。标准注意力是
它的退化特例。生成 = 标准自回归 next-char，但内核用 PER 弛豫而非纯 softmax 注意力。

**可溯源**：逐字惊奇、跨层预测误差 ‖ε‖、各层弛豫率 η、可学突触 S、内容路由 g 全程可读
（attention 给不了的结构化记忆）——见 `code_trace_viz`。

**后天成长**：冻 backbone 只学突触 + 经验回放可长出新规则；每技能新开一块突触 + 冻旧块，
旧技能 loss 数学保证不被覆盖（隔离不遗忘）——见 `growth.py::GrowthLearner`。

## 诚实边界

小模型字符级（此规模生成「成品度」弱），价值在**架构对标 / 可溯源 / 后天成长**，
不在生成博学度与世界知识。A/B 为**同 token 量（数据效率）**口径，Transformer 每步更快，
同墙钟差距会缩小——非碾压，是该规模 / 口径下的稳健小胜。
