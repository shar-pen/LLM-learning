## 背景介绍

-------

        我们曾在《[通用大模型架构分类及技术统一化](https://blog.csdn.net/weixin_65514978/article/details/141595911 "通用大模型架构分类及技术统一化")》一文中，引出过对于 [GLM](https://so.csdn.net/so/search?q=GLM&spm=1001.2101.3001.7020)（General Language Model）【1】模型结构的讨论。该基于自回归式空白填充的通用语言模型，是为了能同时应对自然语言理解（NLU）、无条件生成和条件生成。GLM 通过引入二维位置编码并允许以任意顺序预测片段，对空白填充预训练进行了改进。同时，GLM 可以通过调整空白的数量和长度，适应不同类型的任务。

![](https://i-blog.csdnimg.cn/direct/11c46397940e4180a44c376558ffbaf9.png)

        上图为 GLM 示意图，将文本片段（绿色部分，连续的 token 片段（类似于自编码思想））置为空白，并以自回归方式生成这些片段。

        事实上，空白填充已被 T5【2】用于文本到文本的预训练，不过 GLM 提出了两项改进：**片段乱序（span shuffling）和二维位置编码（2D positional encoding）**。另外，受到 Pattern-Exploiting Training（PET）【3】的启发，GLM 将 NLU 任务重新表述为手工设计的 cloze 问题，以模仿人类语言的形式，GLM 能够通过自回归式空白填充自然处理包含多个 token 的答案。更进一步，通过调整空白片段的数量和长度，展示了自回归式空白填充目标可以预训练[语言模型](https://so.csdn.net/so/search?q=%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020)以支持条件生成和无条件生成。通过对多种预训练目标的多任务学习，一个单一的 GLM 能够在 NLU 和（条件生成与无条件生成）文本生成任务中都表现出色。实验表明，与单一任务基线相比，采用多任务预训练的 GLM 在 NLU、条件文本生成和语言建模任务上均有全面提升，并通过参数共享进一步增强了性能表现。

        关于 PET，这里举一个例子帮助理解：

![](https://i-blog.csdnimg.cn/direct/b497659ce0d84209b85dba6258b69e9f.png)

 PET 基本思想为填空式任务重新表述。PET 通过将传统的 NLP 任务（如文本分类）重新表述为填空问题来匹配预训练语言模型的能力。具体步骤如下：

> *   **设计模式（Pattern）**：为每个任务定义一个模板，将任务转化为类似人类语言的问题。例如：
>     *   文本分类任务：给定一个句子，判断其情感：
>         *   原始输入：`The movie was fantastic!`
>         *   模板：`The movie was fantastic! It was [MASK].`
>         *   标签映射：`great` 对应积极情感，`terrible` 对应消极情感。
> *   **填空预测**：利用预训练语言模型预测 `[MASK]` 的值。
> *   **标签映射（Verbalizer）**：将模型输出的 token 映射到具体的任务标签。例如，`great` -> 积极，`terrible` -> 消极。

        这种方法充分利用了语言模型的上下文建模能力，使模型更容易从少量数据中捕捉到语义信息。

        有了上述背景信息，我们再来看下 glm 模型系列的发展时间线【4】。

![](https://i-blog.csdnimg.cn/direct/1f58d20b630849e1848fed699d77a4ca.png)

## GLM 预训练框架

------------

        GLM 是基于自回归空白填充目标的通用预训练框架。GLM 将自然语言理解（NLU）任务形式化为包含任务描述的填空问题，这些问题可以通过自回归生成进行回答。

### 2.1 预训练目标

#### 2.1.1 自回归空白填充

        GLM 通过优化自回归空白填充目标进行训练。给定输入文本 $x = [x_1, \dots, x_n]$，随机抽取多个文本片段$\{s_1, \dots, s_m\}$，其中每个片段 $s_i$对应于$x$中的一系列连续的 token $[s_{i,1}, \dots, s_{i,l_i}]$。每个片段被替换为单个 [MASK] token，形成一个损坏文本$x_{corrupt}$。模型通过自回归方式预测缺失的 token，在预测一个片段中的缺失 token 时，模型可以访问损坏文本以及之前预测的片段。为了充分捕捉不同片段之间的相互依赖关系，随机打乱片段的顺序，这类似于排列语言模型【5】。形式化地，设 $Z_m$ 为长度为 m 的索引序列 $[1, 2, \dots, m]$ 的所有可能排列集合，$s_{z<i} $  为 ，$[s_{z_1}, \dots, s_{z_{i-1}}]$ 将预训练目标定义为：

$$
\max_\theta \mathbb{E}_{z \sim Z_m} \left[ \sum_{i=1}^{m} \log p_\theta(s_{z_i} | x_{\text{corrupt}}, s_{z<i}) \right]
$$
        按照从左到右的顺序生成每个空白中的 token，即生成片段 $s_i$ 的概率可以分解为：

$$
p_\theta(s_i | x_{\text{corrupt}}, s_{z<i}) = \prod_{j=1}^{l_i} p(s_{i,j} | x_{\text{corrupt}}, s_{z<i}, s_i,_{<j})
$$
        通过以下描述实现自回归空白填充目标。输入 ![](https://latex.csdn.net/eq?x) 被分为两部分：A 部分是损坏文本 ![](https://latex.csdn.net/eq?x_%7B%5Ctext%7Bcorrupt%7D%7D)，B 部分包含被掩码的片段。A 部分的 token 可以相互注意，但不能关注 B 部分中的任何 token；B 部分的 token 可以关注 A 和 B 部分中的前置 token，但不能关注 B 中的后续 token。为了启用自回归生成，每个片段会通过特殊的 [START] 和 [END] token 填充，分别用于输入和输出。通过这种方式，模型在统一模型中自动学习到双向编码器（用于部分 A）和单向解码器（用于部分 B）。GLM 的实现如下图所示。        ![](https://i-blog.csdnimg.cn/direct/f5e3a760aa624ae1a7a3a9884cecd26f.png)

        从泊松分布中随机抽取长度为![](https://latex.csdn.net/eq?%5Clambda%20%3D%203)的片段长度。反复抽取新片段，直到至少 15% 的原始 token 被掩码，15% 的比例对下游 NLU 任务表现至关重要。

        这里关于几个技术点展开说一下：

**（1）排列语言模型**

        排列语言模型（Permutation Language Model）是一种改进的自回归语言模型，通过对输入文本进行排列来增强模型的表达能力，尤其是在处理语言建模和生成任务时。该模型的核心思想是，**传统的自回归语言模型通常按顺序预测每个单词或标记，而排列语言模型通过对标记的顺序进行随机化排列，来捕捉输入序列中不同部分之间的依赖关系**。

> *   **排列顺序**：排列语言模型的一个核心特征是，通过对输入序列中的标记进行随机排列，来生成不同的序列顺序。每次训练时，输入的标记序列的顺序都被打乱，从而模拟各种可能的语境，避免模型仅仅捕捉到线性顺序的依赖关系。
>     
> *   **增强依赖关系捕获**：传统的语言模型（如 GPT）倾向于捕捉文本中的顺序依赖性，而排列语言模型通过不同顺序的排列，增强模型对非线性依赖关系的理解。这样做能够提高模型的泛化能力。
>     
> *   **适应不同任务**：排列语言模型可以适应多种任务，包括文本生成、语言理解、分类等。这是因为在训练过程中，模型学会了在不同的输入顺序下进行有效预测，而不是局限于固定的顺序。
>     

        但是，由于需要处理所有可能的排列，排列语言模型通常比传统的[自回归模型](https://so.csdn.net/so/search?q=%E8%87%AA%E5%9B%9E%E5%BD%92%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020)需要更多的计算资源，尤其在训练阶段。此外尽管排列语言模型增强了对上下文的捕捉，但它可能在某些任务上没有像 GPT 那样自然地捕捉序列的顺序信息。

**（2）泊松分布采样**

        在 GLM 中，采样是基于泊松分布的，主要与其目标 “有效地训练和建模文本中空缺部分的填充（blank infilling）任务” 相关。这里的核心思想是通过泊松分布来控制被遮掩（masked）的文本片段的长度，而不是直接给定固定长度的片段。

        泊松分布描述的是在一个固定时间间隔或空间区域内，某个事件发生的次数的概率分布。它的概率质量函数（PMF）为：

![](https://latex.csdn.net/eq?P%28X%3Dk%29%20%3D%20%5Cfrac%7B%5Clambda%5Ek%20e%5E%7B-%5Clambda%7D%7D%7Bk%21%7D%2C%20%5Cquad%20k%20%3D%200%2C%201%2C%202%2C%20%5Cdots)

        其中，![](https://latex.csdn.net/eq?%5Clambda)是事件的平均发生次数（期望值），k 是实际观察到的事件发生次数。

        在文本处理中，文本片段（如句子或短语）的长度具有自然的变化和随机性。使用泊松分布采样可以在不同的训练实例中产生不同长度的遮掩片段，从而使模型在训练时能够应对不同长度的缺失部分，提高泛化能力。泊松分布的参数![](https://latex.csdn.net/eq?%5Clambda)（均值）控制了每个采样的片段的平均长度。在 GLM 中，λ=3 作为一个超参数来定义，意味着大约 3 个单词的片段会被遮掩。这种分布在遮掩的长度上引入了一定的随机性，使得模型不仅学习如何填补固定长度的空白，还能适应不同长度的缺失部分。泊松分布通常不产生极端长的片段，而是更多地产生中等长度的片段。这有助于避免训练过程中出现过长或过短的片段，既能保持足够的上下文信息，又能避免遮掩的内容太少，导致模型无法有效地学习。

        假设：泊松分布参数为 λ=3，另假设我们需要知道片段长度为 0 到 5 的概率。

        计算各个长度（0 到 5）的概率：

> **长度为 0（即没有遮掩）**：
> 
> ![](https://latex.csdn.net/eq?P%28X%20%3D%200%29%20%3D%20%5Cfrac%7B3%5E0%20e%5E%7B-3%7D%7D%7B0%21%7D%20%3D%20%5Cfrac%7B1%20%5Ctimes%20e%5E%7B-3%7D%7D%7B1%7D%20%3D%20e%5E%7B-3%7D%20%5Capprox%200.0498)
> 
> **长度为 1**：
> 
> ![](https://latex.csdn.net/eq?P%28X%20%3D%201%29%20%3D%20%5Cfrac%7B3%5E1%20e%5E%7B-3%7D%7D%7B1%21%7D%20%3D%20%5Cfrac%7B3%20%5Ctimes%20e%5E%7B-3%7D%7D%7B1%7D%20%3D%203%20%5Ctimes%20e%5E%7B-3%7D%20%5Capprox%200.1494)
> 
> **长度为 2**：
> 
> ![](https://latex.csdn.net/eq?P%28X%20%3D%202%29%20%3D%20%5Cfrac%7B3%5E2%20e%5E%7B-3%7D%7D%7B2%21%7D%20%3D%20%5Cfrac%7B9%20%5Ctimes%20e%5E%7B-3%7D%7D%7B2%7D%20%3D%204.5%20%5Ctimes%20e%5E%7B-3%7D%20%5Capprox%200.2240)
> 
> **长度为 3**：
> 
> ![](https://latex.csdn.net/eq?P%28X%20%3D%203%29%20%3D%20%5Cfrac%7B3%5E3%20e%5E%7B-3%7D%7D%7B3%21%7D%20%3D%20%5Cfrac%7B27%20%5Ctimes%20e%5E%7B-3%7D%7D%7B6%7D%20%3D%204.5%20%5Ctimes%20e%5E%7B-3%7D%20%5Capprox%200.2240)
> 
> **长度为 4**：
> 
> ![](https://latex.csdn.net/eq?P%28X%20%3D%204%29%20%3D%20%5Cfrac%7B3%5E4%20e%5E%7B-3%7D%7D%7B4%21%7D%20%3D%20%5Cfrac%7B81%20%5Ctimes%20e%5E%7B-3%7D%7D%7B24%7D%20%5Capprox%200.1680)
> 
> **长度为 5**：
> 
> ![](https://latex.csdn.net/eq?P%28X%20%3D%205%29%20%3D%20%5Cfrac%7B3%5E5%20e%5E%7B-3%7D%7D%7B5%21%7D%20%3D%20%5Cfrac%7B243%20%5Ctimes%20e%5E%7B-3%7D%7D%7B120%7D%20%5Capprox%200.1008)

        另外提到概率质量函数，这里顺便说一下几个名词的区别：

#### ![](https://i-blog.csdnimg.cn/direct/bccd72821ce54c5f9b58a04e22c72410.png)2.1.2 多任务预训练

        GLM 对短片段进行掩码，适合用于 NLU 任务。然而希望预训练一个可以同时处理 NLU 和文本生成任务的模型。因此引入一种多任务预训练方案，其中生成更长文本的第二个目标与空白填充目标共同优化。考虑以下两种目标：

> *   **文档级别**：抽取一个单一片段，片段长度从原始长度的 50%–100% 之间的均匀分布中抽取。该目标旨在生成较长文本。
> *   **句子级别**：限制掩码片段必须是完整的句子。多个片段（句子）被抽取以覆盖原始文本的 15%。该目标旨在处理 seq2seq 任务，这些任务的预测通常是完整的句子或段落。

        这两个新目标与 2.1.1 中原始目标的定义相同。唯一的不同是片段的数量和长度。

### 2.2 模型架构

        GLM 使用单一 Transformer，并对架构进行了几项修改：

**(1) 重新安排了层归一化和残差连接的顺序，避免大模型发生数值错误【6】。**

        在 GLM 中，重新安排层归一化（Layer Normalization）和残差连接（Residual Connection）的顺序，是为了避免在训练大规模语言模型时出现数值不稳定的问题，也被【6】证明是解决训练过程中数值错误的一种有效方法。

![](https://i-blog.csdnimg.cn/direct/25775a75327e4c9cbd069e343cdcbb72.png)

        层归一化是一种标准化技术，通常用于神经网络的每一层，用来调整输入数据的分布，使得数据的均值为 0，方差为 1。有助于加速模型的收敛，并减少训练过程中的数值不稳定性。在标准化之后，网络的每一层将更稳定地进行训练，避免梯度爆炸或梯度消失的问题。

        残差连接指的是在神经网络的每一层中，将输入直接加到输出上。能够缓解深层网络的训练困难，避免梯度消失问题。残差连接让信息在网络中更直接地流动，有助于深层网络的训练。

        在标准的 Transformer 模型中，层归一化通常放在残差连接之前。即：

![](https://i-blog.csdnimg.cn/direct/8e4afbf88b7c43969836102e2ef27234.png)

> *   输入通过残差连接直接加到输出。
> *   然后对结果进行 层归一化。

        然而，对于大模型，尤其是在训练时使用的参数非常多（比如亿级甚至百亿级参数时），传统的顺序（残差连接 → 层归一化）会在训练过程中引发数值不稳定或梯度消失 / 爆炸问题。因为大模型可能会在深层计算时导致数值的精度丢失或溢出，特别是在反向传播梯度的传递过程中。

        在 GLM 中，层归一化和残差连接的顺序被反过来排列，即先进行层归一化，然后再应用残差连接。这一调整的目的是通过将标准化步骤提前来保持数值的稳定性。

**（2）使用一个线性层进行输出 token 的预测；**

**（3）将 ReLU 激活函数替换为 GeLU 激活函数。**

#### 2.2.1 2D 位置编码

        自回归空白填充任务的一个挑战是如何编码位置信息。Transformer 依赖位置编码来注入 token 的绝对和相对位置。【1】提出了 2D 位置编码来解决这个问题。具体来说，每个 token 使用两个位置 id 进行编码。第一个位置 id 表示在损坏文本![](https://latex.csdn.net/eq?x_%7B%5Ctext%7Bcorrupt%7D%7D)​ 中的位置。对于被掩码的片段，它是对应的 [MASK] token 的位置。第二个位置 id 表示片段内的位置。对于 A 部分的 token，其第二个位置 id 为 0；对于 B 部分的 token，其第二个位置 id 范围从 1 到片段的长度。两个位置 id 通过可学习的嵌入表映射到两个向量，这两个向量被加到输入 token 的嵌入中。编码方法确保模型在重建时不会意识到掩码片段的长度，设计适应下游任务，因为生成文本的长度通常事先是未知的。

### 2.3 GLM 微调

        对于下游的 NLU 任务，线性分类器将预训练模型生成的序列或 token 表示作为输入，并预测正确的标签。这与生成预训练任务不同，导致预训练和微调之间存在不一致。因此将 NLU 分类任务重新表述为空白填充的生成任务，参考了 PET【3】。具体来说，给定一个标注示例 (x,y)，通过包含单个掩码 token 的模式将输入文本 x 转换为填空问题 c(x)。该模式以自然语言编写，以表示任务的语义。例如，情感分类任务可以表述为：“{SENTENCE}。它真的很 [MASK]”。候选标签 y∈Y 也被映射为填空问题的答案，称为 verbalizer v(y)。在情感分类中，标签 “positive” 和“negative”分别映射为 “good” 和“bad”。给定 x 的条件概率为：

![](https://latex.csdn.net/eq?p%28y%20%7C%20x%29%20%3D%20%5Cfrac%7Bp%28v%28y%29%20%7C%20c%28x%29%29%7D%7B%5Csum_%7By%27%20%5Cin%20Y%7D%20p%28v%28y%27%29%20%7C%20c%28x%29%29%7D)

        其中 Y 是标签集合。因此，句子为正面或负面的概率与预测 “good” 或“bad”填入空白的概率成正比。

![](https://i-blog.csdnimg.cn/direct/3ba042bdbe234500af56c4c9cd993d47.png)

        然后，使用交叉熵损失对 GLM 进行微调。对于文本生成任务，给定的上下文构成输入的 A 部分，最后附加一个掩码 token。模型自回归地生成 B 部分的文本。可以直接将预训练的 GLM 应用于无条件生成，或在下游条件生成任务上对其进行微调。

### 2.4 GLM-4 All Tools 模型能力

        GLM-4 All Tools 模型能更好地理解用户意图，并能够自动选择最合适的工具来完成任务【4】。如可以通过网页浏览器以多轮方式访问在线信息，使用 Python 解释器解决数学问题，利用文本到图像模型生成图像，并调用用户定义的函数。下图展示了一个示例，说明 GLM-4 All Tools 如何通过网页浏览器和 Python 解释器来解决用户的查询 “搜索 2000 到 2023 年的全球人口数据，然后计算年均增长率”。

![](https://i-blog.csdnimg.cn/direct/02c7c7a526634bae9d4adab55daff57c.png)

![](https://i-blog.csdnimg.cn/direct/1d7439d8964c4a04b03cafa1c26b2e6d.png)
--------------------------------------------------------------------------

3. GLM 以及 ChatGLM 的技术演进
-----------------------

        首先来看 GLM 系列与 ChatGLM 的对应关系，以及能力提升点。

![](https://i-blog.csdnimg.cn/direct/e498e0ddfb654c88abf6b433f66fa583.png)

### 3.1 GLM-4 关键点

#### 3.1.1 预训练数据

        预训练语料库由多语言（主要是英语和中文）文档组成，来源包括网页、维基百科、书籍、代码和研究论文。数据处理流程主要包括三个阶段：去重、过滤和分词。去重阶段通过去除重复或相似文档来提高数据多样性，包括精确去重和模糊去重。网页过滤阶段通过去除包含攻击性语言、占位符文本、源代码等的噪声文档来提高数据质量。分词阶段将文本转换为标记序列，以便进一步处理。预训练数据中的 token 数量直接影响模型的训练速度。采用了字节对编码（BPE）算法在字节级别上分别学习中文和多语言标记，并将其与 tiktoken 中的 cl100k_base 分词器的标记合并，形成一个大小为 150,000 的统一词汇表。在最终的训练集里，对不同的数据来源进行了重新加权，以增加高质量和教育性资源（如书籍和维基百科）的重要性。预训练语料库包含了约十万亿个标记。在 ChatGLM4 中，数据质量和多样性有助于构建有效的大模型。

#### **3.1.2 架构**

        GLM 系列 LLM 是基于 Transformer 构建的。GLM-130B 采用了 DeepNorm 作为层归一化策略，使用旋转位置编码（RoPE）以及带 GeLU 激活函数的门控线性单元（GLU），并在前馈网络（FFN）中使用它们。探索过程中研究了不同的策略以提升模型性能和推理效率。最近的 GLM-4 模型采用了以下架构设计选择：

> *   **除 QKV 外不使用偏置项**：为了提高训练速度，移除了除了注意力层中查询（Q）、键（K）和值（V）矩阵的偏置项之外的所有偏置项。这样做发现在长度推断方面有轻微的改进。
> *   **RMSNorm 和 SwiGLU**：采用了 RMSNorm 替代了 LayerNorm，采用 SwiGLU 替代了 ReLU，这两种策略提高了模型性能。
> *   **旋转位置嵌入（RoPE）**：将 RoPE 扩展为二维形式，以适应 GLM 中的 2D 位置编码。
> *   **群体查询注意力（GQA）**：用群体查询注意力（GQA）替代了多头注意力（MHA），从而减少了推理过程中的 KV 缓存大小。由于 GQA 使用的参数比 MHA 少，增加了 FFN 参数数量以保持相同的模型大小。

        模型的上下文长度从 2K（ChatGLM），到 32K（ChatGLM2 和 ChatGLM3），再到 128K 和 1M（GLM-4）。这些扩展不仅通过上下文扩展—位置编码扩展和长文本的持续训练来实现，还通过长上下文对齐，使 GLM-4 能够有效处理非常长的上下文【7】。

![](https://i-blog.csdnimg.cn/direct/91bf3a5d3c8443c8b3c296e52d2f4930.png)

#### 3.1.3 对齐

        预训练为 LLM 奠定了基础，而后训练则进一步优化这些模型，使其与人类偏好对齐，例如理解人类意图、遵循指令并支持多轮对话。对于 GLM-4，主要通过监督微调（SFT）和强化学习人类反馈（RLHF）来实现对齐。在 SFT 中，真实的人工提示和交互，而非基于模板或模型生成的响应，对于对齐质量有帮助。虽然 SFT 在很大程度上使基础模型与人类偏好对齐，但 RLHF 可以进一步帮助解决响应拒绝、安全性、生成的双语标记混合以及多轮一致性等问题。对于第一代模型（ChatGLM-6B 和 ChatGLM-130B），提示 - 响应对大多由模型开发人员注释。对于后续的模型，对齐数据则是内部注释和通过第三方获得的专有数据的结合，且这些数据经过严格的质量控制。

**扩展阅读：**

《[全方位解读大模型](https://blog.csdn.net/weixin_65514978/article/details/143043450 "全方位解读大模型")》

4. 参考材料
-------

【1】[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/pdf/2103.10360 "GLM: General Language Model Pretraining with Autoregressive Blank Infilling")

【2】[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683 "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer")

【3】[Exploiting Cloze Questions for Few Shot Text Classification and Natural Language](https://arxiv.org/pdf/2001.07676 "Exploiting Cloze Questions for Few Shot Text Classification and Natural Language") Inference

【4】[ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools](https://arxiv.org/pdf/2406.12793 "ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools")

【5】[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://papers.nips.cc/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf "XLNet: Generalized Autoregressive Pretraining for Language Understanding")

【6】[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053 "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism")

【7】[LongAlign: A Recipe for Long Context Alignment of Large Language Models](https://arxiv.org/pdf/2401.18058 "LongAlign: A Recipe for Long Context Alignment of Large Language Models")