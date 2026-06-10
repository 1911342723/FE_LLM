# FE-LLM 实验 B：多轮 surprise 平复评测

验证主动推理核心命题：对外行动（追问/记忆更新）改变环境后，后续观测的自由能应当下降。

- 用例数：5（通过 5）
- 检查点：15/15
- surprise 下降判定阈值：相对降幅 ≥ 25%
- 澄清满足后 surprise 下降通过：3/3

## 各用例轨迹

### vague_then_clarified_report [PASS]

| turn | prompt | action | surprise | 召回记忆 |
|---:|---|---|---:|---|
| 1 | 帮我写一下 | `ask_clarification` | 0.325 | - |
| 2 | 帮我写一份给老板的项目周报，大约两百字 | `answer` | 0.040 | - |

- [PASS] action (turn 1): expected=ask_clarification selected=ask_clarification
- [PASS] action (turn 2): expected=answer selected=answer
- [PASS] surprise_drop (turn 2): prev=0.325 now=0.040 drop_ratio=87.57%

### vague_then_clarified_translation [PASS]

| turn | prompt | action | surprise | 召回记忆 |
|---:|---|---|---:|---|
| 1 | 帮我弄一下 | `ask_clarification` | 0.325 | - |
| 2 | 把这段话翻译成英文：早上好，今天的会议改到三点 | `answer` | 0.046 | - |

- [PASS] action (turn 1): expected=ask_clarification selected=ask_clarification
- [PASS] action (turn 2): expected=answer selected=answer
- [PASS] surprise_drop (turn 2): prev=0.325 now=0.046 drop_ratio=85.94%

### conflict_then_corrected [PASS]

| turn | prompt | action | surprise | 召回记忆 |
|---:|---|---|---:|---|
| 1 | 我昨天明天去了北京 | `ask_clarification` | 0.304 | - |
| 2 | 抱歉说错了，我是昨天去了北京，想问当地有什么好玩的 | `answer` | 0.044 | - |

- [PASS] action (turn 1): expected=ask_clarification selected=ask_clarification
- [PASS] action (turn 2): expected=answer selected=answer
- [PASS] surprise_drop (turn 2): prev=0.304 now=0.044 drop_ratio=85.60%

### vague_then_still_vague [PASS]

| turn | prompt | action | surprise | 召回记忆 |
|---:|---|---|---:|---|
| 1 | 帮我写一下 | `ask_clarification` | 0.325 | - |
| 2 | 帮我弄一下 | `ask_clarification` | 0.309 | - |

- [PASS] action (turn 1): expected=ask_clarification selected=ask_clarification
- [PASS] action (turn 2): expected=ask_clarification selected=ask_clarification
- [PASS] surprise_stay_high (turn 2): prev=0.325 now=0.309 drop_ratio=4.77%

### memory_then_recalled [PASS]

| turn | prompt | action | surprise | 召回记忆 |
|---:|---|---|---:|---|
| 1 | 记住我喜欢简短回答 | `update_memory` | 0.075 | - |
| 2 | 给我讲讲什么是自由能原理 | `answer` | 0.061 | 记住我喜欢简短回答 |

- [PASS] action (turn 1): expected=update_memory selected=update_memory
- [PASS] action (turn 2): expected=answer selected=answer
- [PASS] memory_recall (turn 2): recalled=True applied_in_text=True

