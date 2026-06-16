# B2b · 真实任务对话(CrossWOZ)领域追踪：belief 的真实 headroom

- 判定：**PARTIAL: belief 在领域追踪上有正向 headroom**
- 数据：`data\crosswoz\train.json.zip`（5012 对话 / 37225 用户轮）
- 领域：['出租', '地铁', '景点', '酒店', '餐馆']；类别分布：{'出租': 747, '地铁': 699, '景点': 12436, '酒店': 11252, '餐馆': 12091}
- 领域未明示子集占比：0.2348（验证集 2181 条）

> 唯一变量=是否加 belief（已出现领域 + 上一活跃领域）；同 MLP/split/train（口径同 teacher_corpus_eval）。

## 总体 balanced accuracy
- 盲（只看句子）：0.9248
- 上下文感知（句子+belief）：0.9776
- delta：0.0528

## 领域未明示子集（跟进句，headroom 关键）
- 盲：0.6638
- 上下文感知：0.8552
- delta：**0.1914**

## 与 B2 互补
- B2（动作类型 offer/nooffer）：belief 无 headroom（−0.02），真实数据动作几乎由 utterance 决定。
- B2b（领域追踪）：见上——belief 在状态/领域追踪环节才是真正决定性的。

- 说明：唯一变量=belief(已出现领域 multi-hot + 上一活跃领域 one-hot)。领域未明示子集=utterance 不含目标领域名（餐馆/酒店/景点/地铁/出租），即跟进句，其领域只能由对话状态决定。与 B2(动作类型 belief 无 headroom) 互补：belief 在真实数据上的价值在状态/领域追踪，不在动作类型。
