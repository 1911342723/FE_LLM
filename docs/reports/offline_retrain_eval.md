# FE-LLM 离线再训练量化：confirmed 记忆回流的泛化提升

- 判定：**PASS: 离线回流显著提升对新偏好的识别（泛化）**
- held-out 新偏好（update_memory）召回：baseline 0.0 → +distill 1.0（delta 1.0）
- 训练偏好 40 / held-out 偏好 40 / 其它动作 21

- 说明：held-out 偏好的模板/填充均与训练不重叠；baseline 训练集不含 update_memory。提升=从 confirmed 记忆学到的泛化识别能力。
