# FE-LLM V2-M1 判定一：v1 单向量 vs v2 分层（同预算 decode_loss）

- 判定：**FAIL: v2 明显劣于 v1**
- 对话条数：3000，字表：1452，epochs：40
- 参数量：v1 5.84M / v2 6.16M
- v1 best decode_loss：1.6981
- v2 best decode_loss：1.878
- v2/v1 比值：1.1059
- 说明：架构级对照（v2 解码器多 slot cross-attention），非单变量消融。
