# CAPCW Part2 · 穷则变自校准 + 按需动态分配

- 判定：**PASS: 生长准则自校准——grow_m 随 K 增长且精度保持(按需分配 slot)**
- 自校准生长准则：相对边际增益 ≥ 0.15 才继续加 slot；训练 max_slots=8，推理按 K 选 grow_m。

| K（绑定负载） | 自选 grow_m | acc@grow_m | acc@max |
|---:|---:|---:|---:|
| 2 | 3.5 | 0.954 | 0.976 |
| 4 | 6.5 | 0.915 | 0.937 |
| 6 | 7.0 | 0.831 | 0.873 |

- grow_m 随 K 单调增长：True；最大精度损失(max−grow)：0.0425

- 说明：自校准生长=相对边际增益(加 slot 降自由能 <min_rel_gain 即停)；训练 max_slots、推理按 K 用 grow_m。grow_m 随 K 增长=按需分配；acc@grow≈acc@max=省 slot 不掉精度。
