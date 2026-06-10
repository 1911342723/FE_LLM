# FE-LLM M2 判定评测：槽位化意图 vs 单向量意图（opus-100 zh→en）

同数据、同规模、同训练预算，唯一变量是意图表示结构。

- 未见验证对：200
- **mean word-F1：0.083**（单向量基线 0.0735，1.13x）
- 输出多样性：89/200 种（单向量版 67/200）
- word-F1≥0.5：1/200；=0：118/200
- 全局能量下降率：8%；覆盖能量下降率：100%
- **M2 判定：FAIL**

## 手写探针（训练集外）

| 中文 | 生成英文 |
|---|---|
| 你好 | you know what? |
| 我爱你 | i don't know what i want. |
| 今天天气很好 | - the same thing is a second. |
| 我有点累 | i don't know. |
| 谢谢你的帮助 | you can't believe it. |
| 我们明天见 | i don't know what i want. |

## 验证集样例（前 20 条）

| 中文 | 参考 | 生成 | word-F1 |
|---|---|---|---:|
| 他說他喜欢妈妈买给我的裙 | he liked the dress mommy bought for me. | you know what i think i said that i'm sorry. | 0.00 |
| 就这里，快点 | right here. | - yes, sir. | 0.00 |
| 问题是现在停止了 | that's the point. it stopped. | the secretary-general assembly | 0.25 |
| 五号车 | car 5 | - that's a good guy. | 0.00 |
| 主人．您的电话 | commander, they're calling you to the phone. | the secretariat will be all right. | 0.15 |
| 你有塑料袋吗? | aah! do you have a plastic bag? | - what are you doing? | 0.17 |
| 你要把我关起来? | you're gonna lock me up? | what are you doing here? | 0.00 |
| 豪鬼不会有事的 | you'll find goki outside. | the secretariat will be all right. | 0.00 |
| 神舟七号，就叫「神七」 | we call shenzhou 7 "sheven". | the secretariat is all over the country. | 0.00 |
| 露西，他耍我们，他骗了我们 | he set us up, lucy. the man set us up. | i don't know what i said, "so." | 0.00 |
| 不 这不会是未决审判 | - no, it's not a mistrial! | the state of the committee recommends | 0.00 |
| 等一下　請不要再光用相片來決定了 | please don't choose based on photos | the secretariat will be all the country. | 0.00 |
| 告诉我 这样更好吗? | tell me, is this...the better way? | what am i supposed to do with my mother? | 0.00 |
| 还得那么少 | more than this world gives | the committee recommends that the state party: | 0.00 |
| - 来点父辈的智慧，怎么样？ | - some paternal wisdom. how about that? | - what do you think about that? | 0.43 |
| 但那是不一樣的 不一樣？ | - but here, it's different? | - what are you doing here? | 0.18 |
| 给我打得连他妈妈都不认识他 | hit him for me. | i don't know what they were the company. | 0.00 |
| 可是人家不要你，把你赶了出来 | you've been expelled! | you know what the fuck is the same time. | 0.00 |
| 也许是因为你先对我好的 | - you were the spider in mywindow. | you know what i think i do. | 0.15 |
| 你要我怎么做 德拉科 | - what would you have me do? draco. | i don't know what the fuck is the world. | 0.12 |
