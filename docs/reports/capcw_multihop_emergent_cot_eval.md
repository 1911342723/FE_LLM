# CAPCW · 纯涌现潜 CoT 2×2 析因（decode-vs-latent × 中间监督-vs-仅末端）

- 判定：**FAIL(中间监督必需): 仅末端监督时 decode→re-embed 也救不活多跳（emergent 0.132 ≈ e2e 0.114，增益 +0.0179<0.10）——CoT **不会自发涌现**，必须显式教 emit 中间符号（中间监督）。诚实负结果，与「多跳要 CoT 且要教」一致。**
- 任务：链 c0→…→cH 查 c0 答 cH；n_sym=20, d=32；随机基线 0.050

| n_hops | e2e(latent·末端) | latent_is(latent·中间) | **emergent(decode·末端)** | cot(decode·中间) |
|---:|---:|---:|---:|---:|
| 1 | 0.288 | 0.288 | **0.288** | 0.288 |
| 2 | 0.104 | 0.445 | **0.125** | 0.471 |
| 3 | 0.123 | 0.302 | **0.138** | 0.354 |

- 多跳(H≥2)均值：e2e 0.114 / latent_is 0.374 / **emergent 0.132** / cot 0.413
- emergent − e2e（decode 本身的作用）= **+0.0179**；cot − emergent（中间监督的额外作用）= **+0.2812**；emergent/cot = **32%**

- 说明：emergent=decode→re-embed + 仅末端监督（不教中间符号）；cot=decode + 中间监督；e2e=latent + 仅末端；latent_is=latent + 中间监督。唯一两变量=读出结构 × 监督，同 seed 初始化。判'纯涌现'看 emergent 这一格。
