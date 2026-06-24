# -*- coding: utf-8 -*-
"""
fe_llm.energy_lm —— 最小能量对话模型（M6）
==========================================
用"能量从不稳定走向稳定"这一第一性原理，训练一个最小的、能生成连贯语言的模型，
证明这条新路可行（"你好"→"你好"级别即可，不对标 GPT、不做产品）。

核心：离散去掩码能量坍缩
    回应从全 [MASK]（最高能/最无序）出发，能量网络 E_θ 反复把每个空位填成
    "让整体最稳定"的字，直到无 [MASK] 且能量收敛 —— 回应在能量下降中浮现。

哲学：
    - 万物趋于稳定：生成 = 能量耗散到谷底。
    - 经验就是省电：训练为高频对话刻出低能深沟，熟悉输入坍缩快、陌生输入坍缩慢。
    - 可溯源：每个字为何填在此——因为它让该位置能量最低，可打印。

包结构（按功能分层）：
    models/       模型定义（energy_net / seq_net / intent_model / slot_intent_model / tokenizer）
    data/         语料与数据准备（corpus / real_data / prepare_lccc / teacher_gen）
    training/     训练脚本（train / seq_train / intent_train / chat_train 聊天模型入口）
    evaluation/   评测脚本（lm_scaling_eval 规模→困惑度曲线 / eval_by_length / ablation_per / scale_test）
    generation/   生成入口（intent_generate）
    diagnostics/  能量坍缩诊断（collapse / seq_collapse）
    demos/        交互演示（demo / seq_demo / growth_demo）

注：因果 PER 语言模型（SeqEnergyNet）已验证 held-out 困惑度随规模单调下降、与 Transformer 同档
（docs/reports/lm_scaling_eval），是当前从零可溯源 LM 的主线。
"""
