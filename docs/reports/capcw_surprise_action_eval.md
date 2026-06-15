# CAPCW Part3 · surprise→动作 闭环（无动作监督）

- 判定：**PARTIAL: surprise 有区分力但偏弱**
- 任务：模型只学绑定取值（没教 ASK/ANSWER）；推理时仅用 query→slot 路由 surprise 阈值判 ASK/ANSWER。
- K=5, d=32, seeds=2

- **surprise→动作 balanced accuracy（无监督）：0.7260**
- query 匹配度：bound（该答）0.7171 vs unbound（该问）0.4680，分离 +0.2491

- 说明：模型只在绑定取值任务上训练，从未见 ASK/ANSWER 标签；用 query→slot 最大路由权重作匹配度、其补=surprise，单阈值判动作。bound 匹配高(低 surprise=该答)、unbound 匹配低(高 surprise=该问)。
