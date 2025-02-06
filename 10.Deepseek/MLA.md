# MHA 

MHA（**M**ulti-**H**ead **A**ttention），也就是多头注意力，是开山之作[《Attention is all you need》](/archives/4765)所提出的一种 Attention 形式，可以说它是当前主流 LLM 的基础工作。在数学上，多头注意力 MHA 等价于多个独立的单头注意力的拼接，假设输入的（行）向量序列为 $\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_l$，其中 $\boldsymbol{x}_i\in\mathbb{R}^d$，那么 MHA 可以形式地记为

$$\begin{equation} \begin{gathered} \boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt] \boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\[15pt] \boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\ \boldsymbol{k}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d\times d_k} \\ \boldsymbol{v}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d\times d_v} \end{gathered} \end{equation}$$

简单起见，这里省略了 Attention 矩阵的缩放因子。实践上，常见的设置是

$d_k = d_v = d / h$

，对于 LLAMA2-7b 有

$d=4096, h=32, d_k = d_v = 128$

，LLAMA2-70b 则是

$d=8192,h=64, d_k = d_v = 128$

由于这里只考虑了主流的自回归 LLM 所用的 Causal Attention，因此在 token by token 递归生成时，新预测出来的第 $t+1$个 token，并不会影响到已经算好的 $\boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}$，因此这部分结果我们可以缓存下来供后续生成调用，避免不必要的重复计算，这就是所谓的 KV Cache。

而后面的 MQA、GQA、MLA，都是围绕 “如何减少 KV Cache 同时尽可能地保证效果” 这个主题发展而来的产物。

瓶颈 [#](#瓶颈)
-----------

一个自然的问题是：为什么降低 KV Cache 的大小如此重要？

众所周知，一般情况下 LLM 的推理都是在 GPU 上进行，单张 GPU 的显存是有限的，一部分我们要用来存放模型的参数和前向计算的激活值，这部分依赖于模型的体量，选定模型后它就是个常数；另外一部分我们要用来存放模型的 KV Cache，这部分不仅依赖于模型的体量，还依赖于模型的输入长度，也就是在推理过程中是动态增长的，当 Context 长度足够长时，它的大小就会占主导地位，可能超出一张卡甚至一台机（8 张卡）的总显存量。

在 GPU 上部署模型的原则是：能一张卡部署的，就不要跨多张卡；能一台机部署的，就不要跨多台机。这是因为 “卡内通信带宽> 卡间通信带宽 > 机间通信带宽”，由于“木桶效应”，模型部署时跨的设备越多，受设备间通信带宽的的“拖累” 就越大，事实上即便是单卡 H100 内 SRAM 与 HBM 的带宽已经达到了 3TB/s，但对于 Short Context 来说这个速度依然还是推理的瓶颈，更不用说更慢的卡间、机间通信了。

所以，减少 KV Cache 的目的就是要实现在更少的设备上推理更长的 Context，或者在相同的 Context 长度下让推理的 batch size 更大，从而实现更快的推理速度或者更大的吞吐总量。当然，最终目的都是为了实现更低的推理成本。

要想更详细地了解这个问题，读者可以进一步阅读[《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》](https://papers.cool/arxiv/2205.14135)、[《A guide to LLM inference and performance》](https://www.baseten.co/blog/llm-transformer-inference-guide/)、[《LLM inference speed of light》](https://zeux.io/2024/03/15/llm-inference-sol/)等文章，这里就不继续展开了（主要是笔者水平也有限，唯恐说多错多）。

MQA [#](#MQA)
-------------

MQA，即 “**M**ulti-**Q**uery **A**ttention”，是减少 KV Cache 的一次非常朴素的尝试，首次提出自[《Fast Transformer Decoding: One Write-Head is All You Need》](https://papers.cool/arxiv/1911.02150)，这已经是 2019 年的论文了，这也意味着早在 LLM 火热之前，减少 KV Cache 就已经是研究人员非常关注的一个课题了。

MQA 的思路很简单，直接让所有 Attention Head 共享同一个 K、V，用公式来说，就是取消 MHA 所有的 $\boldsymbol{k},\boldsymbol{v}$的上标 ${}^{(s)}$：

$$\begin{equation}\require{cancel} \begin{gathered} \boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt] \boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{\color{#ccc}{\smash{\bcancel{(s)}}}} ,\boldsymbol{v}_{\leq t}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)\boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)} \\[15pt] \boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\ \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \boldsymbol{x}_i\boldsymbol{W}_k^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_k} \\ \boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \boldsymbol{x}_i\boldsymbol{W}_v^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_v} \end{gathered} \end{equation}$$

使用 MQA 的模型包括[PaLM](https://arxiv.org/pdf/2204.02311)、[StarCoder](https://papers.cool/arxiv/2305.06161)、[Gemini](https://papers.cool/arxiv/2312.11805)等。很明显，MQA 直接将 KV Cache 减少到了原来的 $1/h$，这是非常可观的，单从节省显存角度看已经是天花板了。

效果方面，目前看来大部分任务的损失都比较有限，且 MQA 的支持者相信这部分损失可以通过进一步训练来弥补回。此外，注意到 MQA 由于共享了 K、V，将会导致 Attention 的参数量减少了将近一半，而为了模型总参数量的不变，通常会相应地增大 FFN/GLU 的规模，这也能弥补一部分效果损失。

GQA
-------------

然而，也有人担心 MQA 对 KV Cache 的压缩太严重，以至于会影响模型的学习效率以及最终效果。为此，一个 MHA 与 MQA 之间的过渡版本 GQA（**G**rouped-**Q**uery **A**ttention）应运而生，出自论文[《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》](https://papers.cool/arxiv/2305.13245)，是去年的工作。

事后看来，GQA 的思想也很朴素，它就是将所有 Head 分为 $g$个组（$g$可以整除 $h$），每组共享同一对 K、V，用数学公式表示为

$$\begin{equation} \begin{gathered} \boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt] \boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{\color{red}{(\lceil sg/h\rceil)}} ,\boldsymbol{v}_{\leq t}^{\color{red}{(\lceil sg/h\rceil)}}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{red}{(\lceil sg/h\rceil)}}{}^{\top}\right)\boldsymbol{v}_i^{\color{red}{(\lceil sg/h\rceil)}}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{red}{(\lceil sg/h\rceil)}}{}^{\top}\right)} \\[15pt] \boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\ \boldsymbol{k}_i^{\color{red}{(\lceil sg/h\rceil)}} = \boldsymbol{x}_i\boldsymbol{W}_k^{\color{red}{(\lceil sg/h\rceil)}}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{\color{red}{(\lceil sg/h\rceil)}}\in\mathbb{R}^{d\times d_k} \\ \boldsymbol{v}_i^{\color{red}{(\lceil sg/h\rceil)}} = \boldsymbol{x}_i\boldsymbol{W}_v^{\color{red}{(\lceil sg/h\rceil)}}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{\color{red}{(\lceil sg/h\rceil)}}\in\mathbb{R}^{d\times d_v} \end{gathered} \end{equation}$$

这里的 $\lceil\cdot\rceil$ 是上取整符号。GQA 提供了 MHA 到 MQA 的自然过渡，当 $g=h$时就是 MHA， $g=1$ 时就是 MQA，当 $1 < g < h$ 时，它只将 KV Cache 压缩到 $g/h$ ，压缩率不如 MQA，但同时也提供了更大的自由度，效果上更有保证。GQA 最知名的使用者，大概是 Meta 开源的[LLAMA2-70B](https://llama.meta.com/llama2/) ，以及 [LLAMA3](https://llama.meta.com/llama3/)全系列，此外使用 GQA 的模型还有 [TigerBot](https://papers.cool/arxiv/2312.08688)、[DeepSeek-V1](https://papers.cool/arxiv/2401.02954)、[StarCoder2](https://papers.cool/arxiv/2402.19173)、[Yi](https://papers.cool/arxiv/2403.04652)、[ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)、[ChatGLM3](https://github.com/THUDM/ChatGLM3)等，相比使用 MQA 的模型更多（ChatGLM 虽然在它的介绍中说自己是 MQA，但实际是 $g=2$ 的 GQA）。

在 llama2/3-70B 中，GQA 的 $g=8$，其他用了 GQA 的同体量模型基本上也保持了这个设置，这并非偶然，而是同样出于推理效率的考虑。我们知道，70B 这个体量的模型，如果不进行极端的量化，那么不可能部署到单卡（A100/H100 80G）上。单卡不行，那么就能单机了，一般情况下一台机可以装 8 张卡，刚才我们说了，Attention 的每个 Head 实际上是独立运算然后拼接起来的，当 $g=8$时，正好可以每张卡负责计算一组 K、V 对应的 Attention Head，这样可以在尽可能保证 K、V 多样性的同时最大程度上减少卡间通信。

MLA
-------------

有了 MHA、MQA、GQA 的铺垫，我们理解 MLA（**M**ulti-head **L**atent **A**ttention）就相对容易一些了。DeepSeek-V2 的技术报告里是从低秩投影的角度引入 MLA 的，以至于有部分读者提出 “为什么 LoRA 提出这么久了，直到 MLA 才提出对 KV Cache 低秩分解的做法” 之类的疑问。

然而，笔者认为低秩投影这个角度并不贴近本质，因为要说低秩投影的话，事实上只要我们将 GQA 的所有 K、V 叠在一起，就会发现 GQA 也相当于在做低秩投影：  

$$\begin{equation}\underbrace{\left[\boldsymbol{k}_i^{(1)},\cdots,\boldsymbol{k}_i^{(g)},\boldsymbol{v}_i^{(1)},\cdots,\boldsymbol{v}_i^{(g)}\right]}_{\boldsymbol{c}_i\in\mathbb{R}^{g(d_k+d_v)}} = \boldsymbol{x}_i \underbrace{\left[\boldsymbol{W}_k^{(1)},\cdots,\boldsymbol{W}_k^{(g)},\boldsymbol{W}_v^{(1)},\cdots,\boldsymbol{W}_v^{(g)}\right]}_{\boldsymbol{W}_c\in\mathbb{R}^{d\times g(d_k+d_v)}}\end{equation}$$

这里我们将所有 $\boldsymbol{k}_i^{(s)},\boldsymbol{v}_i^{(s)}$拼在一起记为 $\boldsymbol{c}_i$，相应的投影矩阵也拼在一起记为 $\boldsymbol{W}_c$，注意到一般都有 $d_c = g(d_k+d_v) < d$ ，所以 $\boldsymbol{x}_i$到 $\boldsymbol{c}_i$ 的变换就是一个低秩投影。所以，MLA 的本质改进不是低秩投影，而是低秩投影之后的工作。

### Part 1 

GQA 在投影之后做了什么呢？首先它将向量对半分为两份分别作为 K、V，然后每一份又均分为 $g$份，每一份复制 $h/g$次，以此来 “凑” 够 $h$个 Attention Head 所需要的 K、V。我们知道分割、复制都是简单的线性变换，所以 MLA 的第一个想法是将这些简单的线性变换换成一般的线性变换，以增强模型的能力：

$$\begin{equation} \begin{gathered} \boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt] \boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\[15pt] \boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\ \boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d_c\times d_k} \\ \boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_c\times d_v} \\[10pt] \boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c} \end{gathered} \end{equation}$$

然而，理论上这样是能增加模型能力，但别忘了 GQA 的主要目的是减少 KV Cache，出于节省计算和通信成本的考虑，我们一般会缓存的是投影后的 $\boldsymbol{k}_i, \boldsymbol{v}_i$ 而不是投影前的 $\boldsymbol{c}_i$ 或 $\boldsymbol{x}_i$ ，而 MLA 的这个做法，通过不同的投影矩阵再次让所有的 K、V Head 都变得各不相同，那么 KV Cache 的大小就恢复成跟 MHA 一样大了，违背了 GQA 的初衷。

对此，MLA 发现，我们可以结合 Dot-Attention 的具体形式，通过一个简单但不失巧妙的恒等变换来规避这个问题。首先，在训练阶段还是照常进行，此时优化空间不大；然后，在推理阶段，我们利用  

$$\begin{equation}\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top} = \left(\boldsymbol{x}_t\boldsymbol{W}_q^{(s)}\right) \left(\boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\right){}^{\top} = \boldsymbol{x}_t\left(\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}\right)\boldsymbol{c}_i^{\top} \end{equation}$$

这意味着推理阶段，我们可以将 $\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$合并起来作为 Q 的投影矩阵，那么 $\boldsymbol{c}_i$则取代了原本的 $\boldsymbol{k}_i$，同理，在 $\boldsymbol{o}_t$后面我们还有一个投影矩阵，于是 $\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}$的 $\boldsymbol{W}_v^{(s)}$ 也可以吸收到后面的投影矩阵中去，于是等效地 $\boldsymbol{v}_i$ 也可以用 $\boldsymbol{c}_i$代替，也就是说此时 KV Cache 只需要存下所有的 $\boldsymbol{c}_i$ 就行，而不至于存下所有的 $\boldsymbol{k}_i^{(s)}$、$\boldsymbol{v}_i^{(s)}$。注意到$\boldsymbol{c}_i$跟${}^{(s)}$无关，也就是说是所有头共享的，即 MLA 在推理阶段它可以恒等变换为一个 MQA。

再次强调，本文的主题是一直都是减少 KV Cache，那到目前为止，MLA 做到了什么呢？答案是通过不同的投影矩阵来增强了 GQA 的能力，并且推理时可以保持同样大小的 KV Cache。那么反过来，如果我们只需要跟 GQA 相近的能力，那么是不是就可以再次减少 KV Cache 了？换言之，$d_c$没必要取 $g(d_k+d_v)$，而是取更小的值（DeepSeek-V2 取了 512），从而进一步压缩 KV Cache，这就是 MLA 的核心思想。

（注：这里有一个细节，就是 $\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$合并成一个矩阵的恒等变换，理论上只有在无限精度下才成立，实际上如果我们使用单精度尤其是 BF16 的话，经过变换后的精度损失往往还是挺明显的，经过多层累积后可能放大到比较可观的程度，这里可能要根据实际误差看要不要做一些后处理。）

### Part 2 

一切似乎都很完美，看上去一个又好又省的理想设计就要出炉了。不过别急，当我们再深入思考一下就会发现，到目前为止的 MLA 有一个难以绕开的缺陷——不兼容 [RoPE（旋转位置编码）](/archives/8265)。

刚才我们说了，MLA 之所以能保持跟 GQA 一样大小的 KV Cache，其关键一步是 “将 $\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$合并成一个（跟位置无关的）矩阵作为 Q 的投影矩阵”，但如果加了 RoPE 的话，这一步就无法实现了。这是因为 RoPE 是一个跟位置相关的、$d_k\times d_k$的分块对角矩阵 $\boldsymbol{\mathcal{R}}_m$，满足 $\boldsymbol{\mathcal{R}}_m\boldsymbol{\mathcal{R}}_n^{\top}=\boldsymbol{\mathcal{R}}_{m-n}$，MLA 加入 RoPE 之后会让 $\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$之间多插入了一项 $\boldsymbol{\mathcal{R}}_{t-i}$：

$$\begin{equation} \boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\quad,\quad\boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i} \\ \boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top} = \left(\boldsymbol{x}_t\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_t}\right) \left(\boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right){}^{\top} = \boldsymbol{x}_t\left(\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_{t-i}}\boldsymbol{W}_k^{(s)}{}^{\top}\right)\boldsymbol{c}_i^{\top} \end{equation}$$

这里的 $\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_{t-i}}\boldsymbol{W}_k^{(s)}{}^{\top}$ 就无法合并为一个固定的投影矩阵了（跟位置差 $t-i$ 相关），从而 MLA 的想法无法结合 RoPE 实现。

前段时间，笔者也很荣幸跟 DeepSeek 团队讨论过这个问题，但这个问题可以说非常本质，所以当时笔者实际上也没能提出什么有效的建议。最简单的方式是放弃 RoPE，换用其他基于 Attention Bias 的位置编码，如 [ALIBI](/archives/9431#ALIBI)，但 DeepSeek 的实验显示它明显不如 RoPE（注意，MLA 不是不能加 RoPE，而是加了 RoPE 之后无法用恒等变换技巧来减少 KV Cache），笔者也提议过换 [Sandwich](/archives/9431#Sandwich)，它不像 ALIBI 单调衰减到负无穷，估计效果会好些，但感觉是治标不治本。还有一个折中的办法是将 $\boldsymbol{q}_i$的输入也改为 $\boldsymbol{c}_i$，然后 RoPE 加在 $\boldsymbol{c}_i$之后，即 $$\begin{equation}\boldsymbol{q}_i^{(s)} = \boldsymbol{c}_i\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\boldsymbol{W}_q^{(s)},\quad\boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\boldsymbol{W}_k^{(s)}\end{equation}$$

这样 $\boldsymbol{\mathcal{R}}_i$ 就可以吸收到 $\boldsymbol{c}_i$ 中去，但这样就没有 $\boldsymbol{\mathcal{R}}_m\boldsymbol{\mathcal{R}}_n^{\top}=\boldsymbol{\mathcal{R}}_{m-n}$ 的运算了，此时的 RoPE 不再是通过绝对位置实现相对位置，而单纯是在 Q、K 上加绝对位置，让模型自己想办法提炼相对位置信息。

最后发布的 MLA，采取了一种混合的方法——每个 Attention Head 的 Q、K 新增 $d_r$个维度用来添加 RoPE，其中 K 新增的维度每个 Head 共享：

$$\begin{equation} \begin{gathered} \boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt] \boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\[15pt] \boldsymbol{q}_i^{(s)} = \left[\boldsymbol{x}_i\boldsymbol{W}_{qc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k + d_r},\quad \boldsymbol{W}_{qc}^{(s)}\in\mathbb{R}^{d\times d_k},\boldsymbol{W}_{qr}^{(s)}\in\mathbb{R}^{d\times d_r}\\ \boldsymbol{k}_i^{(s)} = \left[\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k+d_r},\quad \boldsymbol{W}_{kc}^{(s)}\in\mathbb{R}^{d_c\times d_k}, \boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_r} \\ \boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_c\times d_v} \\[10pt] \boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c} \end{gathered} \end{equation}$$

这样一来，没有 RoPE 的维度就可以重复 “Part 1” 的操作，在推理时 KV Cache 只需要存 $\boldsymbol{c}_i$ ，新增的带 RoPE 的维度就可以用来补充位置信息，并且由于所有 Head 共享，所以也就只有在 K Cache 这里增加了 $d_r$个维度，原论文取了 $d_r = d_k / 2 = 64$，相比原本的 $d_c=512$，增加的幅度不大。

### Part 3 

最后有一个细节，就是 MLA 的最终版本，还将 Q 的输入也改为了低秩投影形式，这与减少 KV Cache 无关，主要是为了减少训练期间参数量和相应的梯度（原论文说的是激活值，个人表示不大理解）所占的显存：  

$$\begin{equation} \begin{gathered} \boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt] \boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\[15pt] \boldsymbol{q}_i^{(s)} = \left[\boldsymbol{c}_i'\boldsymbol{W}_{qc}^{(s)}, \boldsymbol{c}_i'\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k + d_r},\quad \boldsymbol{W}_{qc}^{(s)}\in\mathbb{R}^{d_c'\times d_k},\boldsymbol{W}_{qr}^{(s)}\in\mathbb{R}^{d_c'\times d_r}\\ \boldsymbol{k}_i^{(s)} = \left[\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k+d_r},\quad \boldsymbol{W}_{kc}^{(s)}\in\mathbb{R}^{d_c\times d_k}, \boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_r} \\ \boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_c\times d_v} \\[10pt] \boldsymbol{c}_i' = \boldsymbol{x}_i \boldsymbol{W}_c'\in\mathbb{R}^{d_c'},\quad \boldsymbol{W}_c'\in\mathbb{R}^{d\times d_c'} \\ \boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c} \\ \end{gathered} \end{equation}$$

注意 $\boldsymbol{k}_i^{(s)}$中的第二项，带 RoPE 的部分，其输入还是 $\boldsymbol{x}_i$而不是 $\boldsymbol{c}_i$，这里保持了原论文的设置，不是笔误， $d_c'$ 原论文的取值是 1536，跟 $d_c=512$ 不同。同时，我们把带 RoPE 的 MHA 放在下面，方便大家对比：

$$\begin{equation} \begin{gathered} \boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt] \boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\[15pt] \boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\ \boldsymbol{k}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_k^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d\times d_k} \\ \boldsymbol{v}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d\times d_v} \end{gathered} \end{equation}$$

可以发现，其实在训练阶段，除了多了一步低秩投影以及只在部分维度加 RoPE 外，MLA 与 Q、K 的 Head Size 由 $d_k$换成 $d_k + d_r$ 的 MHA 基本无异。推理阶段的 MLA 则改为  

$$\begin{equation} \begin{gathered} \boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}\boldsymbol{W}_v^{(1)}, \boldsymbol{o}_t^{(2)}\boldsymbol{W}_v^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\boldsymbol{W}_v^{(h)}\right] \\[10pt] \boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{\color{#ccc}{\smash{\bcancel{(s)}}}} ,\boldsymbol{c}_{\leq t}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)\boldsymbol{c}_i}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)} \\[15pt] \boldsymbol{q}_i^{(s)} = \left[\boldsymbol{c}_i'\boldsymbol{W}_{qc}^{(s)}\boldsymbol{W}_{kc}^{(s)}{}^{\top}, \boldsymbol{c}_i'\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_c + d_r}\\ \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \left[\boldsymbol{c}_i, \boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_c+d_r}\\ \boldsymbol{W}_{qc}^{(s)}\in\mathbb{R}^{d_c'\times d_k},\boldsymbol{W}_{kc}^{(s)}\in\mathbb{R}^{d_c\times d_k},\boldsymbol{W}_{qr}^{(s)}\in\mathbb{R}^{d_c'\times d_r},\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_r} \\[10pt] \boldsymbol{c}_i' = \boldsymbol{x}_i \boldsymbol{W}_c'\in\mathbb{R}^{d_c'},\quad \boldsymbol{W}_c'\in\mathbb{R}^{d\times d_c'} \\ \boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c} \\ \end{gathered} \end{equation}$$

此时 Q、K 的 Head Size 变成了 $d_c + d_r$ ，V 的 Head Size 则变成了 $d_c$ ，按照原论文的设置，这是 $d_k$、 $d_v$的 4 倍。所以实际上 MLA 在推理阶段做的这个转换，虽然能有效减少 KV Cache，但其推理的计算量是增加的。

那为什么还能提高推理效率呢？这又回到 “瓶颈” 一节所讨论的问题了，我们可以将 LLM 的推理分两部分：第一个 Token 的生成（Prefill）和后续每个 Token 的生成（Generation），Prefill 阶段涉及到对输入所有 Token 的并行计算，然后把对应的 KV Cache 存下来，这部分对于计算、带宽和显存都是瓶颈，MLA 虽然增大了计算量，但 KV Cache 的减少也降低了显存和带宽的压力，大家半斤八两；但是 Generation 阶段由于每步只计算一个 Token，实际上它更多的是带宽瓶颈和显存瓶颈，因此 MLA 的引入理论上能明显提高 Generation 的速度。

还有一个细节充分体现了这个特性。一般的 LLM 架构参数满足 $h \times d_k = d$，即 num_heads * head_size = hidden_size，但 DeepSeek-V2 不一样，它 $d_k=128,d=5120$，但 $h=128$，是一般设置的 3 倍！这是因为 MLA 的 KV Cache 大小跟 $h$无关，增大 $h$只会增加计算量和提升模型能力，但不会增加 KV Cache，所以不会带来速度瓶颈。

