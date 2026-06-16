# B2 · 真实任务对话(CrossWOZ)控制层 headroom（上下文感知 vs 盲）

- 判定：**WEAK: 真实任务对话上 belief 对系统动作预测无 headroom（同句多动作≈0，系统动作几乎由当前 utterance 决定）**
- 数据：`data\crosswoz\train.json.zip`（5012 对话 / 36910 usr→sys 实质轮）
- vocab=1868，belief 槽位键=53

> 唯一变量=是否加 belief（决策前累积用户 informed (domain·slot)）；同 MLP/split/train（口径同 teacher_corpus_eval）。
> CrossWOZ 系统从不 Request，故测 offer/nooffer 与 inform/recommend/nooffer 两种动作方案。

## 方案 binary_offer_nooffer
- 类别分布：{'offer': 34410, 'nooffer': 2500}，歧义占比 0.0001
- 总体 balanced_acc：盲 0.8211 → 感知 0.8011（delta **-0.0201**）
- 歧义子集：盲 0.0 → 感知 0.0（delta +0.0000）

## 方案 triple_inform_recommend_nooffer
- 类别分布：{'inform': 31451, 'recommend': 2959, 'nooffer': 2500}，歧义占比 0.0009
- 总体 balanced_acc：盲 0.7763 → 感知 0.7654（delta **-0.0110**）
- 歧义子集：盲 0.5 → 感知 0.5（delta +0.0000）

## 对照锚点
- 教师合成任务语料歧义子集 belief delta：+0.51（强，但属构造特性）
- 真实开放闲聊 LCCC belief：0.655（弱）

- 说明：真实人标任务对话；唯一变量=belief(决策前累积用户 informed (domain·slot) multi-hot)。口径与 teacher_corpus_eval 一致（同 MLP/split/train）。CrossWOZ 系统从不 Request，故测 offer/nooffer（binary）与 inform/recommend/nooffer（triple）。对照：教师合成歧义子集 +0.51、开放闲聊 LCCC 0.655。
