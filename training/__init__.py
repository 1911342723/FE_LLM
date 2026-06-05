# -*- coding: utf-8 -*-
"""
training 包 —— 训练层（与运行时严格隔离）
==========================================
本层独立于 fe_llm 运行时包，只在"训练阶段"使用，产出固化的权重文件到 checkpoints/。

按文档要求，整个架构里只有两个部分需要被深度学习框架真正训练并固化成权重：
    1) SurpriseNet (自由能数学引擎) —— train_surprise_net.py
    2) DecoderNet  (能量递减解码器) —— train_decoder_net.py

训练数据通过"教师蒸馏"自动生成：
    - SurpriseNet 的标签来自 RuleFreeEnergyEngine（解析版误差打分）。
    - DecoderNet  的标签来自几何版能量残余（余弦距离）。
这样无需人工标注即可冷启动；后续可用真实人类反馈替换标签来源。
"""
