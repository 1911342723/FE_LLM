# FE-LLM 翻译泛化评测（opus-100 zh→en，IntentLM 架构）

在完全未见的验证集上检验：意图弛豫 + 能量递减解码能否做跨语言重建。
字符级 8M 小模型口径，目标是验证架构泛化性，不与 SOTA 翻译对比。

- 未见验证对：200
- 平均 word-F1：0.073（≥0.5 的占 1/200，=0 的占 134/200）
- 平均 char-F1：0.583
- exact match：0/200
- 能量整体下降比例：100%
- hybrid 选字分歧率：8.1%

## 手写探针（训练集外）

| 中文 | 生成英文 |
|---|---|
| 你好 | you are a second. |
| 我爱你 | i don't know what you want. |
| 今天天气很好 | i want to see you again. |
| 我有点累 | i don't know. |
| 谢谢你的帮助 | i don't know what you want. |
| 我们明天见 | i don't know what to do. |

## 验证集样例（前 20 条）

| 中文 | 参考 | 生成 | word-F1 |
|---|---|---|---:|
| 他說他喜欢妈妈买给我的裙 | he liked the dress mommy bought for me. | i don't know what to do. | 0.00 |
| 就这里，快点 | right here. | i don't know what to do. | 0.00 |
| 问题是现在停止了 | that's the point. it stopped. | i was a big deal. | 0.00 |
| 五号车 | car 5 | i don't know what to do. | 0.00 |
| 主人．您的电话 | commander, they're calling you to the phone. | i was a big dead. | 0.00 |
| 你有塑料袋吗? | aah! do you have a plastic bag? | you don't know what that means? | 0.15 |
| 你要把我关起来? | you're gonna lock me up? | you don't know what that means? | 0.00 |
| 豪鬼不会有事的 | you'll find goki outside. | i want to see you again. | 0.00 |
| 神舟七号，就叫「神七」 | we call shenzhou 7 "sheven". | i don't know what to do, don't you? | 0.00 |
| 露西，他耍我们，他骗了我们 | he set us up, lucy. the man set us up. | i don't know what to do. | 0.00 |
| 不 这不会是未决审判 | - no, it's not a mistrial! | international agenda | 0.00 |
| 等一下　請不要再光用相片來決定了 | please don't choose based on photos | i want to see you again. | 0.00 |
| 告诉我 这样更好吗? | tell me, is this...the better way? | what are you doing? | 0.00 |
| 还得那么少 | more than this world gives | i want to see you again. | 0.00 |
| - 来点父辈的智慧，怎么样？ | - some paternal wisdom. how about that? | - what are you doing? | 0.17 |
| 但那是不一樣的 不一樣？ | - but here, it's different? | i was just like that. | 0.00 |
| 给我打得连他妈妈都不认识他 | hit him for me. | i want to see you again. | 0.00 |
| 可是人家不要你，把你赶了出来 | you've been expelled! | i don't know what to do. | 0.00 |
| 也许是因为你先对我好的 | - you were the spider in mywindow. | i want you to be a couple of this time. | 0.12 |
| 你要我怎么做 德拉科 | - what would you have me do? draco. | you don't know what to do, do you? | 0.25 |
