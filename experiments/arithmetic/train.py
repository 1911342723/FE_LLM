# -*- coding: utf-8 -*-
"""
arithmetic/train.py —— 训练算术认知网络（EnergyNet + SolverNet）
================================================================
运行：python experiments/arithmetic/train.py

训练目标（两个损失，共享答案码本）：
    1) 能量对比损失（EnergyNet）：
       让正确答案吸引子的能量低、错误答案吸引子的能量高。
       用类似 InfoNCE 的方式：对所有候选答案算能量，正确答案应是能量最低者。
       这等价于让系统"对错误答案感到惊奇"。

    2) 意图回归损失（SolverNet）：
       让解码器输出的意图向量贴近正确答案吸引子坐标（拉近几何距离）。
       生成时滚落到最近吸引子即得答案。

教师：DeepSeek（先抽样校验管道），训练数据由本地程序生成的"绝对正确教师"提供。
设备：自动用 CUDA（RTX 5060）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fe_llm.config import CHECKPOINT_DIR, get_device
from experiments.arithmetic.encoding import (AnswerSpace, answer_range,
                                             encode_question)
from experiments.arithmetic.model import AnswerCodebook, EnergyNet, SolverNet
from experiments.arithmetic.teacher import generate_problems, verify_with_deepseek

# 实验超参（小任务，分钟级训练）
OPS = ("+", "-")
MAX_VAL = 50
EMBED_DIM = 64
N_TRAIN = 12000
N_TEST = 1000
EPOCHS = 120
BATCH = 256
LR = 1e-3
ANSWER_SCALE = 100.0   # 标量答案归一化尺度（答案 ÷ scale 后落在约 [-0.5, 1.0]）
CKPT_SUBDIR = os.path.join(CHECKPOINT_DIR, "arithmetic")


def build_tensors(problems, space: AnswerSpace, device):
    """把题目列表转成张量：题目特征 + 正确答案码本索引 + 归一化标量答案。"""
    q = np.vstack([encode_question(p.a, p.b, p.op) for p in problems])
    y = np.array([space.to_index(p.answer) for p in problems], dtype=np.int64)
    val = np.array([[p.answer / ANSWER_SCALE] for p in problems], dtype=np.float32)
    return (torch.tensor(q, device=device),
            torch.tensor(y, device=device),
            torch.tensor(val, device=device))


def main(verify_teacher: bool = True):
    device = get_device()
    print(f"[arith] 设备：{device}  "
          f"({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")

    # —— 答案空间（能量地貌的吸引子集合）——
    lo, hi = answer_range(OPS, MAX_VAL)
    space = AnswerSpace(lo, hi)
    print(f"[arith] 答案空间：[{lo}, {hi}]  共 {space.size} 个答案吸引子")

    # —— 生成数据 ——
    train_p = generate_problems(N_TRAIN, MAX_VAL, OPS, seed=1)
    test_p = generate_problems(N_TEST, MAX_VAL, OPS, seed=2)

    # —— 教师管道校验（抽样让 DeepSeek 解题）——
    if verify_teacher:
        print("[arith] 校验 DeepSeek 教师管道（抽样解题）...")
        try:
            acc = verify_with_deepseek(train_p, sample=5)
            print(f"[arith] 教师一致率：{acc*100:.0f}%")
        except Exception as exc:
            print(f"[arith] 教师校验跳过（{exc}）；继续用程序生成的标准答案训练。")

    qx, qy, qv = build_tensors(train_p, space, device)
    tx, ty, tv = build_tensors(test_p, space, device)
    loader = DataLoader(TensorDataset(qx, qy, qv), batch_size=BATCH, shuffle=True)

    # —— 模型（两个网络 + 共享码本）——
    # 用结构化固定吸引子：数值相近的答案天然在空间中相近且整体有序可分，
    # 让解码器的"滚落到最近吸引子"稳定准确。
    codebook = AnswerCodebook(space.size, EMBED_DIM,
                              learnable=False, lo=lo).to(device)
    energy_net = EnergyNet(EMBED_DIM).to(device)
    solver_net = SolverNet(EMBED_DIM).to(device)

    params = list(energy_net.parameters()) + list(solver_net.parameters())
    # 固定码本无可训练参数；若改用 learnable 码本则需把其参数也加入优化器
    if codebook.learnable:
        params += list(codebook.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    # 余弦退火：后期降低学习率，缓解联合训练的准确率震荡，让解码器稳定收敛
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    print(f"[arith] 开始训练：{EPOCHS} epochs, {len(qx)} 训练样本")
    for epoch in range(1, EPOCHS + 1):
        energy_net.train(); solver_net.train(); codebook.train()
        tot_e = tot_s = 0.0
        for bx, by, bv in loader:
            opt.zero_grad()

            # 1) 能量对比：对所有答案算能量，正确答案应能量最低
            #    把"低能量=高得分"，用 -energy 当 logits 做交叉熵 → 正确答案 argmin 能量
            energies = energy_net.energy_over_all(bx, codebook)   # (B, N)
            logits = -energies
            loss_energy = F.cross_entropy(logits, by)

            # 2) 意图回归：解码器同时回归"高维意图向量"和"归一化标量答案"
            #    标量头是数值认知的核心：先形成"答案约为多少"，再滚落(取整)到最近整数吸引子。
            intent_vec, scalar = solver_net(bx)
            target_vec = codebook(by)                # 结构化固定吸引子坐标（无梯度）
            loss_vec = F.mse_loss(intent_vec, target_vec)
            loss_scalar = F.mse_loss(scalar, bv)
            loss_solver = 10.0 * loss_scalar + 0.05 * loss_vec   # 标量为绝对主导

            loss = loss_energy + loss_solver
            loss.backward()
            opt.step()
            tot_e += loss_energy.item() * len(bx)
            tot_s += loss_scalar.item() * len(bx)
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            e_acc, s_acc = evaluate(energy_net, solver_net, codebook, tx, ty, space)
            print(f"[arith] epoch {epoch:3d} | E_loss={tot_e/len(qx):.4f} "
                  f"S_loss={tot_s/len(qx):.4f} | 能量法准确率={e_acc*100:.1f}% "
                  f"解码器准确率={s_acc*100:.1f}%")

    # —— 固化权重 ——
    os.makedirs(CKPT_SUBDIR, exist_ok=True)
    torch.save({"embed_dim": EMBED_DIM, "lo": lo, "hi": hi, "ops": OPS,
                "max_val": MAX_VAL, "learnable": codebook.learnable,
                "answer_scale": ANSWER_SCALE,
                "codebook": codebook.state_dict(),
                "energy_net": energy_net.state_dict(),
                "solver_net": solver_net.state_dict()},
               os.path.join(CKPT_SUBDIR, "arith_fe.pt"))
    print(f"[arith] 权重已保存：{os.path.join(CKPT_SUBDIR, 'arith_fe.pt')}")


@torch.no_grad()
def evaluate(energy_net, solver_net, codebook, tx, ty, space: AnswerSpace):
    """
    两种生成方式各测一次准确率：
        能量法  ：argmin 能量（EnergyNet 在所有吸引子上找最低能量）。
        解码器法：SolverNet 输出归一化标量答案，× scale 后"滚落"(取整)到最近整数吸引子。
    """
    energy_net.eval(); solver_net.eval(); codebook.eval()

    # 能量法
    energies = energy_net.energy_over_all(tx, codebook)   # (T, N)
    e_pred = energies.argmin(dim=1)
    e_acc = (e_pred == ty).float().mean().item()

    # 解码器法：标量答案 → 取整滚落到最近整数 → 映射回码本索引
    _, scalar = solver_net(tx)                            # (T, 1)
    pred_val = torch.round(scalar.squeeze(1) * ANSWER_SCALE)  # 滚落到最近整数吸引子
    pred_val = pred_val.clamp(space.lo, space.hi)
    true_val = torch.tensor([space.to_value(int(i)) for i in ty.cpu()],
                            device=tx.device, dtype=pred_val.dtype)
    s_acc = (pred_val == true_val).float().mean().item()

    return e_acc, s_acc


if __name__ == "__main__":
    main()
