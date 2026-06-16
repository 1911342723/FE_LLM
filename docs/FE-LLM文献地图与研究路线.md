# FE-LLM 文献地图与研究路线

> 当前定位：FE-LLM 不是“下一代 GPT 已成立”，而是一个探索 **能量解释 + 预测编码交互 + 可溯源生成** 的研究型语言模型原型。
> 目标不是先证明 Transformer 错，而是找出：能量视角到底能带来哪些真实、可测、可复现的新能力。

## 1. 对当前项目的稳健表述

建议把项目分成三条线，而不是混成一个大结论：

1. **概念内核线 `kernel/`**
   - 价值：显式自由能、概念吸引子、可打印推理链、主动追问。
   - 边界：目前更像概念级 toy cognitive kernel，不是语言模型主体。
   - 文献定位：predictive coding / active inference / world model / associative memory。

2. **并行坍缩线 `energy_lm` v3**
   - 价值：尝试把“整句从 mask 态坍缩到低能态”落在字符生成上。
   - 边界：当前 checkpoint 生成质量不稳定，非自回归长序列字序是硬问题。
   - 文献定位：BERT as MRF、Mask-Predict、MaskGIT、D3PM、Diffusion-LM、EDLM。

3. **顺序能量线 `seq_*` v4**
   - 价值：当前最能说人话的一支；可以做“能量化、可溯源、带预测编码块的因果语言模型”。
   - 边界：它已经是逐字条件生成，不能再主张“非自回归生成”。
   - 文献定位：autoregressive LM + energy-based reranking/decoding + predictive-coding-inspired block。

一句话：**先承认 v4 是顺序生成，把能量解释做硬；再把 v3 作为并行生成的高风险探索。**

## 2. 文献地图

### A. 能量模型与序列级 EBM

- LeCun et al., 2006, *A Tutorial on Energy-Based Learning*  
  https://yann.lecun.org/exdb/publis/pdf/lecun-06.pdf  
  对 FE-LLM 的意义：给“能量低 = 配置兼容”这个基础语言背书。注意：EBM 的关键难点一直是归一化、负样本、采样和训练稳定性。

- Deng et al., 2020, *Residual Energy-Based Models for Text Generation*  
  https://arxiv.org/abs/2004.11714  
  意义：文本 EBM 不是空白地带。主流可行路线通常不是从零替代 LM，而是在预训练 LM 残差上加 sequence-level energy。

- Bakhtin et al., 2020, *Residual Energy-Based Models for Text*  
  https://arxiv.org/abs/2004.10188  
  意义：把 discriminator/energy 合进生成过程，可改善自回归样本。对 FE-LLM 的提醒：可以先做“能量修正器”，别急着完全推翻 AR。

- Grathwohl et al., 2019, *Your Classifier is Secretly an Energy Based Model*  
  https://arxiv.org/abs/1912.03263  
  意义：`energy = -logits` 是合法视角，但如果训练仍是 CE/softmax，就要诚实说“分类器的 EBM 解释”，不能说完全脱离概率训练。

- Xu et al., 2024, *Energy-Based Diffusion Language Models for Text Generation*  
  https://arxiv.org/abs/2410.21357  
  意义：非常贴近 FE-LLM 的后续方向。它把 EBM 放在离散扩散每一步的整序列能量上，并承认 diffusion LM 与 AR 仍有差距。

### B. 预测编码、自由能与主动推理

- Rao & Ballard, 1999, *Predictive Coding in the Visual Cortex*  
  https://www.cnbc.cmu.edu/~tai/readings/nature/Rao-Ballard-99-NN-nv.pdf  
  意义：预测-误差-修正的经典出处。FE-LLM 的 PER 可以靠近这里，但要避免把“神经科学灵感”说成“已经证明更强”。

- Friston, 2010, *The Free-Energy Principle: A Unified Brain Theory?*  
  https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20A%20unified%20brain%20theory.pdf  
  意义：哲学根基。注意它是认知理论，不自动给出 NLP 训练目标。

- Buckley et al., 2017 / Parr et al. 系列，active inference on discrete state spaces  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/  
  意义：M3“模糊则追问”可以用 expected free energy 来正规化：回答、追问、拒答都是 action。

- Whittington & Bogacz, 2017, predictive coding approximates backprop  
  https://pubmed.ncbi.nlm.nih.gov/28333583/  
  意义：支持“局部预测误差学习”这条线，但也提醒：要给出具体梯度/更新等价或近似条件。

- Scellier & Bengio, 2017, *Equilibrium Propagation*  
  https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full  
  意义：能量最小化与学习规则之间的桥。FE-LLM 可借鉴“自由相/弱夹持相”的实验设计。

- Millidge et al., 2020/2022, predictive coding and backprop unification  
  https://arxiv.org/abs/2006.04182  
  https://arxiv.org/abs/2206.02629  
  意义：如果要把 PER 发展成训练算法，这批文献是必读。

### C. 非自回归、掩码生成与并行迭代

- Devlin et al., 2018/2019, *BERT*  
  https://arxiv.org/abs/1810.04805  
  意义：masked LM 的源头之一。FE-LLM v3 的训练形式与 MLM 很近，应正面承认。

- Wang & Cho, 2019, *BERT as a Markov Random Field Language Model*  
  https://arxiv.org/abs/1902.04094  
  意义：非常关键。它说明 bidirectional masked model 可被看成 MRF/undirected LM；FE-LLM 可把“能量坍缩”接到这条线，而不是孤立发明。

- Gu et al., 2018, *Non-Autoregressive Neural Machine Translation*  
  https://openreview.net/pdf?id=B1l8BtlCb  
  意义：非自回归的老问题：多模态、长度、fertility、字序。FE-LLM 当前 v3 的崩点并不意外。

- Ghazvininejad et al., 2019, *Mask-Predict*  
  https://aclanthology.org/D19-1633/  
  意义：从全 mask/部分 mask 迭代填充目标序列，是 FE-LLM v3 最接近的 NLP 参照。

- Chang et al., 2022, *MaskGIT*  
  https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html  
  意义：并行生成 + 反复 mask 低置信位置在图像上很强；对文本是否成立，要靠实验，而不是直觉。

- Austin et al., 2021, *D3PM*  
  https://arxiv.org/abs/2107.03006  
  意义：离散状态扩散是并行文本生成的重要背景。

- Li et al., 2022, *Diffusion-LM*  
  https://arxiv.org/abs/2205.14217  
  意义：连续 latent 上做文本扩散，优势在可控生成；可作为 FE-LLM “能量轨迹可控”路线的对照。

- Gong et al., 2022, *DiffuSeq*  
  https://arxiv.org/abs/2210.08933  
  意义：seq2seq diffusion 的代表。可以作为 v3/v5 后续的强 baseline。

### D. 注意力、Hopfield 与能量解释

- Vaswani et al., 2017, *Attention Is All You Need*  
  https://arxiv.org/abs/1706.03762  
  意义：不要把 Transformer 塑造成稻草人。它不是“只会统计接龙”，而是高度可并行、可缩放的条件建模架构。

- Ramsauer et al., 2020, *Hopfield Networks is All You Need*  
  https://arxiv.org/abs/2008.02217  
  意义：这篇对 PER 很重要。注意力已经有能量/吸引子/联想记忆解释。若 FE-LLM 说“注意力是 PER 退化特例”，需要和现代 Hopfield 的等价关系做清楚对比。

- Widrich et al., 2020, modern Hopfield attention for immune repertoires  
  https://arxiv.org/abs/2007.13505  
  意义：attention-as-memory 已是成熟线索。PER 的新意应落在“多轮误差弛豫 + 可解释信号 + 动作决策”，而不是“注意力也能写成能量”。

### E. 检索记忆、零重训与知识编辑

- Khandelwal et al., 2020, *kNN-LM*  
  https://arxiv.org/abs/1911.00172  
  意义：外挂记忆改善 LM 是成熟路线。FE-LLM 的 MemoryBank 应定位为 energy-shaped retrieval，而不是“Transformer 做不到”。

- Lewis et al., 2020, *RAG*  
  https://arxiv.org/abs/2005.11401  
  意义：外部知识库 + 生成器是标准范式。FE-LLM 可以强调“检索结果如何进入能量函数”，而不是强调“零重训”本身。

- Borgeaud et al., 2022, *RETRO*  
  https://arxiv.org/abs/2112.04426  
  意义：检索数据库能显著提升语言模型，并带来可干预性。这直接挑战“零重训是 FE-LLM 独有性质”的说法。

- Meng et al., 2022, *ROME*  
  https://arxiv.org/abs/2202.05262  
  意义：Transformer 参数内知识可以被定位和编辑。别说“Transformer 学新知识必须重训”；更准确是“参数内编辑和外挂检索都已有路线，FE-LLM 的不同点是能量场注入形式”。

- MEMIT, *Mass Editing Memory in a Transformer*  
  https://memit.baulab.info/  
  意义：知识编辑可以规模化到多事实。FE-LLM 后续若主张“知识可外科演化”，必须和 MEMIT/ROME 做对照。

### F. JEPA、世界模型与 latent prediction

- LeCun, *A Path Towards Autonomous Machine Intelligence*  
  https://www.rivista.ai/wp-content/uploads/2025/10/10356_a_path_towards_autonomous_mach.pdf  
  意义：把 EBM、世界模型、分层规划放在一个愿景里。FE-LLM 的高层叙事和这条线同源。

- Assran et al., 2023, *I-JEPA*  
  https://arxiv.org/abs/2301.08243  
  意义：预测 latent representation 而不是重建像素/字面细节。FE-LLM 可以考虑从“字符能量”上移到“语义 latent 能量”，再读出文字。

- Ha & Schmidhuber, 2018, *World Models*  
  https://arxiv.org/abs/1803.10122  
  意义：世界模型不是文档库，而是可预测动态的 latent state。`kernel/` 的概念空间可以朝这个方向增强。

### G. 生成质量与评测

- Holtzman et al., 2019, *The Curious Case of Neural Text Degeneration*  
  https://arxiv.org/abs/1904.09751  
  意义：重复、退化、argmax 崩坏是文本生成常见病。FE-LLM 当前 v3 的重复输出应纳入这一类问题研究。

- Zhang et al., 2019, *BERTScore*  
  https://arxiv.org/abs/1904.09675  
  意义：短对话可以用语义相似度补充 exact match。

- Pillutla et al., 2021, *MAUVE*  
  https://arxiv.org/abs/2102.01454  
  意义：如果未来做开放生成，要比较生成分布和人类文本分布，不能只看个例 demo。

## 3. 对 FE-LLM 的研究判断

### 判断一：v3 的失败不是丢人，是研究问题本身

并行去掩码文本生成天然会遇到：

- 多个合理回复的平均化，导致“通用盆地”；
- 位置之间缺少稳定顺序约束；
- 低能 token 局部最优与整句自然性不一致；
- `[EOS]/[PAD]` 稳态会改善长度，但不自动解决语义和字序。

所以 v3 不应写成“已证明连贯生成”，应写成：

> 我们实现了一个并行能量坍缩生成原型，并观察到熟悉输入能量更低；但当前生成质量仍受非自回归字序和多模态平均问题限制。

### 判断二：v4 是最有实用希望的一支

`seq_*` 当前最稳，因为它恢复了内容轴的顺序依赖。建议把它定位为：

> Causal Energy LM with Predictive-Error Relaxation Blocks。

也就是：不是“反自回归”，而是“自回归内容轴 + 能量/预测编码思考轴”。

这个定位更强，因为它可以直接和 tiny Transformer、GRU、普通 causal LM 做公平对照。

### 判断三：PER 的新意要用消融证明，而不是命名证明

下一步 PER 需要回答：

- 去掉 `eta` 弛豫是否显著掉点？
- 去掉 `W_pred` 与 error residual 是否显著掉点？
- 改成标准 attention，在同参数量、同训练步数、同数据上差多少？
- PER 的 trace 是否真的比 attention entropy 更能预测错误/不确定性？

如果答案是肯定的，PER 就有硬价值。

### 判断四：MemoryBank 应改名和改叙事

建议从“零重训成长”改为：

> Energy-shaped external memory / energy-biased retrieval。

评测要对照：

- 无记忆；
- exact-match lookup；
- Jaccard retrieval + 直接输出；
- kNN-LM 风格插值；
- FE-LLM energy floor 注入。

只有超过这些 baseline，才算机制贡献。

## 4. 下一步实验路线

### 实验 1：建立实验登记表

每个 checkpoint 必须记录：

- 数据文件；
- 样本数；
- 字表大小；
- 参数量；
- 训练命令；
- epoch/step；
- loss/accuracy；
- 生成样例；
- 自动评测结果；
- git commit；
- checkpoint 文件名。

否则论文数字会继续“平行宇宙化”。

### 实验 2：v4 公平 baseline

任务：短对话生成。

模型：

- `SeqEnergyNet/PER`;
- 同参数量 causal Transformer;
- 同参数量 GRU/LSTM;
- n-gram 或检索 baseline。

指标：

- next-char accuracy / perplexity；
- exact match；
- BERTScore；
- repetition rate；
- EOS accuracy；
- 能量/确定性是否能预测错误。

关键问题：PER 是否比 attention/GRU 更好，还是只是换了名字。

### 实验 3：v3 并行坍缩专项

不要一开始追求开放对话，先做受控任务：

- 固定长度 copy / reverse / template filling；
- 多个槽位条件生成；
- 同义 prompt 到同一回复；
- 多模态回复集合下是否会平均化。

指标：

- denoising accuracy；
- full-sequence exact match；
- order error；
- repetition rate；
- energy monotonicity；
- number of refinement rounds vs quality。

对照：Mask-Predict / MaskGIT-style confidence remasking / D3PM。

### 实验 4：主动推理

把“模糊则追问”从 demo 变成数据集：

- 清晰问题：应该直接回答；
- 缺槽问题：应该追问；
- 噪声输入：应该拒答/重述请求；
- 多意图冲突：应该澄清。

指标：

- action accuracy：answer / ask / refuse；
- expected free energy 是否与人工不确定性相关；
- 追问后成功率是否提升。

### 实验 5：能量解释是否有用

不要只打印 trace。要测 trace 能不能预测失败：

- 能量高的输出是否更容易错？
- entropy/确定性低的位置是否对应人类标注的坏字？
- energy drop 曲线是否能区分熟悉/陌生/噪声输入？
- attribution 去掉某 token 后，输出语义是否真的变坏？

如果 trace 能预测错误，它才是“可溯源”；否则只是日志。

## 5. 建议的论文标题

当前更稳的标题：

> FE-LLM: A Research Prototype for Energy-Interpretable Language Generation with Predictive-Error Relaxation

中文：

> FE-LLM：一种基于能量解释与预测误差弛豫的可溯源语言生成原型

不要用：

- “下一代大语言模型”；
- “Transformer 结构上做不到”；
- “根除字序问题”；
- “非 softmax”；
- “已证明通用可行”。

## 6. 最短研究路线

1. 先把 v4 做硬：公平 baseline + 自动评测 + trace 预测错误。
2. 再把 v3 作为并行坍缩探索：只在受控任务上证明能量迭代确实优于一次性填空。
3. PER 单独成论文点：和 attention / Hopfield / predictive coding 的关系讲清楚。
4. MemoryBank 降调为 energy-shaped retrieval：和 RAG/kNN-LM/ROME/MEMIT 对齐。
5. 最后再谈“自由能语言系统”的大叙事。

这条路会少一点燃，但硬得多。
