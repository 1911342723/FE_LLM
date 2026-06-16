# FE-LLM V2-M1 判定二/三：分层是否值这个 decode_loss 代价

- 综合：**分层未充分证明价值**

## 判定二a 意图分工（z_global vs z_local → action）
- z_global balanced acc：0.1819
- z_local  balanced acc：0.1885
- 通过（z_global 更会分意图）：False

## 判定二b 局部分工（z_local vs z_global → 局部 token）
- z_local token acc：0.7005
- z_global token acc：0.0092
- 通过（z_local 更会分局部 token）：True

## 判定三 分层 surprise 定位（顶层自由能）
- 正常：25.4892
- 词序打乱（表面）：25.4456
- 语义拼接（gist 破坏）：25.4725
- 通过（语义破坏 > 词序打乱）：True
