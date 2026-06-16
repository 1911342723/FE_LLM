# FE-LLM V2-M1 重定向：分层在「动作选择」标尺上的消融

- 判定：**FAIL: 分层未提升动作选择**
- 样本：11182，字表：1921，epochs：40
- 对照 relax_steps=0 best balanced acc：0.9002
- 实验 relax_steps=5 best balanced acc：0.9034
- delta（实验-对照）：0.0032
- 说明：唯一变量=分层弛豫(relax_steps)；同初始化/数据/预算。动作选择是 FE-LLM 真正有价值的标尺。
