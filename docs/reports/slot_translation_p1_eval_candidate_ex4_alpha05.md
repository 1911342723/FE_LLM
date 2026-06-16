# FE-LLM N1 P1 评估：冻结底座 + FE 机制层

- 判定：**FAIL: B below 0.3 word-F1**
- 通过标准：B 组 mean word-F1 ≥ 0.3，且高于 A/C 对照。

## 分组汇总

| 组别 | n | word-F1 | char-F1 | exact | unique | residual↓ | coverage↓ | disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 20 | 0.161 | 0.641 | 0 | 20 | 55% | 95% | 0.0% |
| B | 20 | 0.155 | 0.660 | 0 | 20 | 40% | 95% | 9.2% |
| C | 20 | 0.139 | 0.591 | 0 | 20 | 50% | 100% | 10.4% |

## 样例

| 组别 | 中文 | 参考 | 生成 | word-F1 |
|---|---|---|---|---:|
| A | 他說他喜欢妈妈买给我的裙 | he liked the dress mommy bought for me. | He said he liked my mother's dress. | 0.27 |
| B | 他說他喜欢妈妈买给我的裙 | he liked the dress mommy bought for me. | He said he liked my mother's dress. | 0.27 |
| C | 他說他喜欢妈妈买给我的裙 | he liked the dress mommy bought for me. | He said he liked to buy me a dress from his mother | 0.30 |
| A | 就这里，快点 | right here. | Just here, hurry up | 0.00 |
| B | 就这里，快点 | right here. | Just here, hurry up | 0.00 |
| C | 就这里，快点 | right here. | Just here, hurry up | 0.00 |
| A | 问题是现在停止了 | that's the point. it stopped. | 问题现在停止了 | 0.00 |
| B | 问题是现在停止了 | that's the point. it stopped. | It is now stopped. | 0.22 |
| C | 问题是现在停止了 | that's the point. it stopped. | 问题已经停止了 | 0.00 |
| A | 五号车 | car 5 | 5th car | 0.50 |
| B | 五号车 | car 5 | 5th car | 0.50 |
| C | 五号车 | car 5 | 5th car | 0.50 |
| A | 主人．您的电话 | commander, they're calling you to the phone. | What's your name? | 0.00 |
| B | 主人．您的电话 | commander, they're calling you to the phone. | What's your name? | 0.00 |
| C | 主人．您的电话 | commander, they're calling you to the phone. | What's your name? | 0.00 |
| A | 你有塑料袋吗? | aah! do you have a plastic bag? | Do you have a plastic bag? | 0.77 |
| B | 你有塑料袋吗? | aah! do you have a plastic bag? | Do you have plastic bags? | 0.50 |
| C | 你有塑料袋吗? | aah! do you have a plastic bag? | Do you have plastic bags? | 0.50 |
| A | 你要把我关起来? | you're gonna lock me up? | You want me to be locked up? | 0.33 |
| B | 你要把我关起来? | you're gonna lock me up? | You want me locked up? | 0.40 |
| C | 你要把我关起来? | you're gonna lock me up? | You want me locked up? | 0.40 |
| A | 豪鬼不会有事的 | you'll find goki outside. | There is no danger for the ghost | 0.00 |
| B | 豪鬼不会有事的 | you'll find goki outside. | There is no danger for the ghost | 0.00 |
| C | 豪鬼不会有事的 | you'll find goki outside. | There is no such thing as a ghost in the West | 0.00 |
| A | 神舟七号，就叫「神七」 | we call shenzhou 7 "sheven". | The Shenzhou VII, also called 'Shenzhou VII', was the | 0.00 |
| B | 神舟七号，就叫「神七」 | we call shenzhou 7 "sheven". | The Shenzhou VII was named 'Shenzhou VII' | 0.00 |
| C | 神舟七号，就叫「神七」 | we call shenzhou 7 "sheven". | The Shenzhou VII, called 'Shenzhou VII', was launched on | 0.00 |
| A | 露西，他耍我们，他骗了我们 | he set us up, lucy. the man set us up. | He played tricks on us, he fooled us | 0.25 |
| B | 露西，他耍我们，他骗了我们 | he set us up, lucy. the man set us up. | He played tricks on us, he fooled us | 0.25 |
| C | 露西，他耍我们，他骗了我们 | he set us up, lucy. the man set us up. | He played tricks on us, he fooled us | 0.25 |
