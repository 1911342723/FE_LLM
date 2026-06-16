# FE-LLM 真实 controller 的 belief headroom（订票上下文）

- 判定：**PASS: 真实 controller 的 belief 在 headroom 轮决定性胜出**
- session 数：60
- 轮A（route 未知应追问）ASK 率：1.0

## headroom 轮 C（提供过 route 后再说『帮我订票』）ANSWER 准确率
- stateful（belief 记住 route）：1.0
- memoryless（无跨轮记忆）：0.0
- delta：1.0

- 说明：真实 ActiveInferenceController；headroom 轮='提供过 route 后再说帮我订票'。stateful 用 known_slots 记住 route→ANSWER，memoryless 无记忆→ASK。
