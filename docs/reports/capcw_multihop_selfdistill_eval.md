# CAPCW · 自蒸馏中间步（单跳 teacher 自生成中间监督，免 GT 中间标签）

- 判定：**PASS: **中间监督可自举**——self_distill 0.771 比 final_only 0.576 高 +0.1949，且达 GT 中间监督 0.793 的 97%。**只用单跳标签 + 单跳teacher 自生成中间步即可恢复多跳，无需 GT 中间标签**——中间监督(第 26 节主因)的标注可被省掉。**
- 设置：n_sym=20, n_pairs=5, d=32；随机基线 0.050；读出统一 decode→re-embed

| n_hops | final_only(无中间) | **self_distill(自生成中间)** | gt_intermediate(GT中间·天花板) | teacher 单跳 | teacher 迭代 |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.798 | **0.858** | 0.797 | 0.681 | 0.539 |
| 3 | 0.354 | **0.684** | 0.789 | 0.677 | 0.458 |

- 多跳(H≥2)：self_distill **0.771** vs final_only **0.576**（+0.1949）vs GT 中间 **0.793**（self 达 GT 的 97%）。

- 说明：三臂读出都用 decode→re-embed，唯一变量=中间步来源(无/self/GT)；最终跳一律 GT(现实有最终标签)。teacher=单跳取回(随机 query,只需单跳标签)，自生成中间步=teacher 迭代单跳。承接第 26 节(中间监督是主因)。
