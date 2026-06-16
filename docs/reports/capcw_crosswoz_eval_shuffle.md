# CAPCW Part1 · 真实语言(CrossWOZ)对话状态绑定检索 [shuffled(真in-context)]

- 判定：**PASS: CAPCW 在真实语言(CrossWOZ)绑定检索上同样明显胜单向量**
- 任务：真实 CrossWOZ informed (领域·槽位→值) 绑定，query 槽位检索其值；按对话切分。
- 词表：slot 53 / value 40；K=4, d=32（容量受限）；随机基线 0.025

| 世界状态结构 | 检索 accuracy |
|---|---:|
| flat（单向量） | 0.2966 |
| CAPCW_PC（slot 工作空间） | 0.8825 |

- delta（CAPCW − flat）= **+0.5859**

- 说明：真实 CrossWOZ informed (领域·槽位→值)；同槽在不同对话取不同值→必须 in-context 检索；按对话切分 test 未见。唯一变量=世界状态结构(单向量 vs slot 工作空间)。
