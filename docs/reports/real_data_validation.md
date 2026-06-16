# FE-LLM 控制层真实数据验证（LCCC 真人对话）

## 测试1：belief 预测降低连贯续接的 surprise
- 真实下一句 surprise 均值：0.0764
- 随机句 surprise 均值：0.0818
- pairwise 胜率（真实<随机）：0.655
- 判定：**PASS**

## 测试2：surprise 检测真实表面异常
- 正常 surprise 均值：0.0807
- 字符打乱 surprise 均值：0.0788
- 打乱更高比例：0.0

- 说明：真实 LCCC 对话；test1 验证 belief 预测让连贯续接更不惊奇，test2 验证 surprise 对真实表面异常敏感。阳/阴均如实记录。
