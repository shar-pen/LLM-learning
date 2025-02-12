# GRPO算法

## 为什么需要关注强化学习与策略优化？

在正式开始介绍 GRPO 之前，我们先谈谈一个较为根本的问题：**为什么需要策略优化？又为什么要在意强化学习？** 其实，无论是做推荐系统、对话系统，还是在数学推理、大语言模型对齐（alignment）场景里，最终我们都希望模型能输出 “更优” 或“更符合某些偏好”的序列。**深度强化学习**（DRL）借用 “奖励”（reward）来衡量我们希望的目标，从而对生成的过程进行引导。策略优化（Policy Optimization）则是其中一个关键方法论。

在语言模型的应用中，比如要让模型解出数学题、满足人类对话偏好（例如避免不良输出，或给出更详细解释），我们往往先用大规模的无监督或自监督训练打下基础，然后通过一些 “监督微调”（SFT）再进一步让模型学会初步符合需求。然而，SFT 有时难以将人类或某些高层目标的偏好显式地整合进去。**这时，“强化学习微调” 就登场了**。PPO 是其中的代表性算法，但它同样有自己的痛点，比如要维护额外的大价值网络，对内存与计算的需求在大模型场景中不容忽视。GRPO 正是在此背景下闪亮登场。

## 强化学习中的基本概念

### 智能体、环境与交互

在传统的强化学习框架中，我们通常有一个“智能体”（Agent）和一个 “环境”（Environment）。智能体每一步会基于自身策略 $\pi(s)$去决定一个动作 $a$，然后环境会根据这个动作给出新的状态和一个奖励 $r$，智能体收集这个奖励并继续下一步。这种循环往复构成了一个时间序列过程，直到到达终止条件（如达成目标或超时等）。

不过在语言模型（尤其是大型语言模型，LLM）当中，我们也可以把一个“问题”（例如一段文本提示 prompt）当作环境给的状态，然后模型（智能体）产出下一 token（动作），再不断重复，直到生成一段完整的回答；人类或额外的奖励模型再给予一个整段回答的质量分，或在每个 token（或步骤）时刻给出一个局部奖励。虽然大语言模型看似和传统强化学习中的“马尔可夫决策过程(MDP)”有一些差别，但本质上也可以抽象为状态—动作—奖励—状态—动作的机制。

### 状态、动作、奖励、策略

- **状态**$s$：对于语言模型来说，可以把已经生成的 token 序列（以及当前问题）视为一种压缩后的状态；在传统 RL 里则是环境观测到的一些向量或特征。
- **动作**$a$：在语言模型生成场景，动作可以是 “在词表 vocabulary 里选出下一个 token”；在机器人或游戏环境中就是“移动、旋转、跳跃” 等操作。
- **奖励**$r$：衡量好坏程度的指标。在语言模型对齐中，常见做法是训练一个奖励模型来打分；或者直接用规则判断回答是否正确等。
- **策略**$\pi$：智能体在状态$s$ 下如何选动作$a$的概率分布函数$\pi(s)$。在语言模型里，这就是产生每个 token 的条件分布。

### 价值函数与优势函数：为什么需要它们

在 PPO 等典型策略梯度方法中，我们通常还会引入一个**价值函数（Value Function）**，它大致表示在当前状态下，未来能期望得到多少奖励；或者更进一步，我们可以在每个动作之后去看 “优势函数（Advantage Function）”，衡量 “这个动作比平均水平好多少”。为什么要搞价值函数或优势函数？因为在训练时，如果只有奖励的直接指引，每个样本都可能方差很大，收敛缓慢。价值函数的引入可以**降低训练方差**，提升训练效率。



## 从传统方法到近端策略优化（PPO）的发展脉络

### 策略梯度与 Actor-Critic 范式

**策略梯度方法（Policy Gradient）** 是强化学习中一种比较直接的做法：我们直接对策略函数$\pi_{\theta}(a \mid s)$

进行建模，计算相应的梯度来最大化期望回报。它不用像价值迭代一样枚举所有状态-动作组合，也不用像 Q-learning 那样先学Q再做贪心决策。策略梯度可以很好地适应高维连续动作空间，以及更灵活的策略表示。

不过，如果单纯用 **REINFORCE **等策略梯度方法，每一步更新都可能有很大方差，甚至出现不稳定现象。为此，研究者们提出了 **Actor-Critic** 框架：将 “策略” 叫做 **Actor**，将 “价值函数” 叫做 **Critic**，两者共同训练，让 **Critic **起到估计价值、降低方差的作用。

### PPO 的核心思路：clip 与优势函数

后来又有了**近端策略优化（PPO）**，它是在 **Actor-Critic **的基础上，为了避免策略更新太猛导致训练不稳定，引入了一个**剪切 (clip)** 技巧，即把

$$
\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)}
$$

这个概率比率给夹在$[1-\varepsilon, 1+\varepsilon] $ 区间内。这样就能防止每次更新过度，从而保持相对稳定。但要在实践中实现 PPO，需要在每个时间步都有一个价值网络去估计优势函数

$$
A_t = r_t + \gamma V_{\psi}(s_{t+1}) - V_{\psi}(s_t) 
$$

或者更常用的是广义优势估计（GAE），来让更新时的方差更小。可问题在于，当我们的模型规模急剧增加——如在数十亿甚至千亿参数的语言模型上搞PPO，就会发现**训练资源消耗巨大**。因为这个价值网络本身通常要和策略网络 “同样大” 或近似大，并且需要在每个 token 都计算价值，从而带来可观的内存占用与计算代价。

### PPO 的局限性：模型规模与价值网络的负担

小模型时代，这也许还好，但是在当代的 LLM 背景下，我们需要**极度节省**训练内存与计算资源。尤其当你要做 RLHF（Reinforcement Learning from Human Feedback）或者别的对齐强化学习时，还要搭建奖励模型Reward Model、价值网络Critic Model，再加上本身的策略模型 Actor Model，算力负担往往让人头痛。

这就是 GRPO 的问题背景：**如何在保证 PPO 那样的收益（稳定、可控等）前提下，减少对昂贵价值网络的依赖？** 这背后的核心思路就是：用 “分组输出相互比较” 的方式来估计基线（Baseline），从而免去对价值网络的需求。



## GRPO（分组相对策略优化）

### GRPO 提出的动机：为何需要它

基于上节的对 PPO 的简要回顾，可以了解到PPO在大模型时代的痛点。要不就牺牲训练速度和成本，要不就需要想其他方法来绕过价值网络的全程参与。而 GRPO（全称 Group Relative Policy Optimization）正是对这一问题做出了一种解答。

**核心动机**：在许多实际应用中，奖励只有在序列末端才给一个分数（称之为 Result/Oucome Supervision），或在每一步给一些局部分数（Process Supervision）。不管怎么样，这个奖励本身往往是离散且比较稀疏的，要让价值网络去学习每个 token 的价值，可能并不划算。而如果我们在同一个问题 q 上**采样多份输出** $o_1, o_2, \ldots, o_G$​，对它们进行奖励对比，就能更好地推断哪些输出更好。**由此，就能对每个输出的所有 token 做相对评分**，无须明确地学到一个价值函数。

在数理推理、数学解题等场景，这个技巧尤其管用，因为常常会基于同一个题目 q 生成多个候选输出，有对有错，或者优劣程度不同。那就把它们的奖励进行一个分组内的比较，以获取相对差异，然后把相对优势视为更新策略的依据。

### GRPO 的关键点一：分组采样与相对奖励

GRPO 中，“分组”非常关键：我们会在一个问题 q 上，采样 GRPO 份输出$o_1, o_2, \ldots, o_G$​。然后把这组输出一起送进奖励模型（或规则），得到奖励分 $r_1, r_2, \ldots, r_G$。下一步干嘛呢？我们并不是单纯地对每个输出和一个固定基线比较，而是先把$\mathbf{r} = \{r_1, r_2, \ldots, r_G\}$ 做一个归一化（如减去平均值再除以标准差），从而得出分组内的相对水平。这样就形成了相对奖励 $\tilde{r}_i$。最后我们会把这个相对奖励赋给该输出对应的所有 token 的优势函数。

简单来说：**多生成几份答案，一起比较，再根据排名或分数差更新**，能更直接、简洁地反映同一问题下的优劣关系，而不需要用一个显式的价值网络去学习所有中间时刻的估计。

### GRPO 的关键点二：无需价值网络的高效策略优化

因为不再需要在每个 token 上拟合一个价值函数，我们就能**大幅节省内存**——不必再维护和 **Actor **同样大的 **Critic **模型。这不仅是存储层面的解放，也是训练过程中的显著加速。

当然，GRPO 也会引入一些新的代价：我们要为每个问题采样一组输出（不止一条），意味着推理时要多花点算力去生成候选答案。这种方法和 “自洽性采样（Self-consistency）” 思路也有点类似，如果你了解一些数学题多候选合并判断的做法，就能感受到其中的相通之处。

## GRPO 的原理

![](./assets/GRPO-pipeline.png)

PPO 和 GRPO 的对比。 GRPO 放弃了价值模型，从分组得分中估计，显著减少了训练资源

先让我们写下一个 PPO 的核心目标函数回顾一下：在 PPO 的简化推导里，假设一次只更新一步，那么

$$
\mathcal{J}^{\mathrm{PPO}}(\theta) = \mathbb{E}_{[q \sim P(Q),\, o \sim \pi_{\theta_{\mathrm{old}}}(O \mid q)]} \Biggl[ \frac{1}{\|o\|} \sum_{t=1}^{\|o\|} \frac{\pi_{\theta}(o_t \mid q, o_{<t})}{\pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})} \, A_t \Biggr]
$$



- $q$是从一个训练集问题分布$P(Q)$中采样来的问题；
- $o$是在旧策略 $\pi_{\theta_{\mathrm{old}}}$下生成的输出序列；
- $\|o\|$是输出序列的长度（token 数）；
- $A_t$是优势函数，需要一个单独价值网络 $V_\psi$来估计。

而 GRPO 做的事情则是：同样从问题分布中取到$q$，但这一次我们会针对同一个$q$采样出一组输出 $\{o_1, \ldots, o_G\}$。对每个输出$o_i$做奖励打分 $r_i$。然后相对化后，将它当作对各 token 的优势函数。最后也类似 PPO 的做法去最大化一个带有 **ratio **的目标，只不过 “价值函数” 被分组相对奖励给替代了。用更直观的话说：

$$
\mathcal{J}^{\mathrm{GRPO}}(\theta) = \mathbb{E} \Biggl[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\|o_i\|} \sum_{t=1}^{\|o_i\|} \min\bigl[ r_{\mathrm{ratio}},\, \operatorname{clip}(r_{\mathrm{ratio}},\, 1-\varepsilon,\, 1+\varepsilon) \bigr] \cdot \hat{A}_{i,t} \biggr] - \text{(KL 正则项)}
$$

其中

- $r_{\mathrm{ratio}} = \frac{\pi_{\theta}(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q, o_{i,<t})} $，
- $\hat{A}_{i,t}$是分组相对意义上的 “优势”，我们下节会具体解释它是怎么来的；
- KL 正则用来限制策略和一个参考策略（通常是初始 SFT 模型或当前 $\theta_{\mathrm{old}}$之间不要差异过大，以防训练崩坏。

### 分组得分与基线估计

那么$ \hat{A}_{i,t}$到底怎么来？就是**分组相对奖励**：我们先把每个$o_i$的奖励$r_i$做如下归一化

$$
\tilde{r}_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})} 
$$

然后令

$$
\hat{A}_{i,t} = \tilde{r}_i
$$

也就是说，输出$o_i$的所有 token 共享同一个分数$\tilde{r}_i$。它们的好坏相对于该分组内的平均水平来衡量，而不依赖外部价值网络去“拆分”或“插值”。这样我们就得到了一个无价值网络的优势函数，核心思路就是**基于相互间的比较与排序**。

如果用的是过程监督（process supervision），即在推理过程中的每个关键步骤都打分，那么就会略有不同。那时每个步骤都有一个局部奖励，就可以把它依时间序列累加或折算成与 token 对应的优势，这在后文示例里我们会详细展示。

### 一步步理解损失函数

让我们把 PPO/GRPO 都视为一种 “Actor 优化” 过程，每个 token 的梯度大致长这样：

$$
\nabla_{\theta} \mathcal{J}(\theta) = \mathbb{E}\bigl[ (\text{gradient coefficient}) \cdot \nabla_{\theta} \log \pi_{\theta}(o_t \mid q, o_{<t}) \bigr]
$$

在 PPO 里，gradient coefficient 里往往含有优势$A_t$以及 ratio 等信息；而在GRPO里，gradient coefficient 变成了以分组奖励为基础的一些值。之所以说GRPO是PPO的一个变体，是因为它同样维持了 ratio 的范式，只不过优势函数来自 “分组内相对奖励”，而非价值网络。

### 惩罚项与 KL 正则

PPO 中常见的 KL 惩罚手段或者 clipping 手段，在 GRPO 中都可以**保留**，以避免训练过程中的策略分布出现暴走。当然，也有一些更精细的做法，比如把 per-token KL正则直接加到损失中，而不是只在奖励函数 $r$里扣一个$\beta \cdot \log \frac{\pi_\theta}{\pi_{\mathrm{ref}}}$。这在各家实现时略有不同，但思路都类似。

## 用 GRPO 来解决一个简单问题

有了上文的理论基础后，可以通过一个简化的实例，帮助你把 GRPO 的实施逻辑走一遍。我们会从最基本的样本生成到分组打分再到反向传播。

### 实验场景与环境：示例说明

假设有一个文本对话场景：系统给定一个问题$q$，模型需要给出回答$o$。我们有一个**奖励模型**来判断回答的好坏（比如回答是否准确、是否违反某些安全规范等），返回一个数值分$r$。为简单起见，就不考虑过程监督，先考虑结果监督（Outcome Supervision）的情境。

在这个设定下，每个问题$q$提供的 “回合” 只有一次——即输出一段文本$o$，即可拿到一个终端奖励$r$。要做GRPO，我们至少要对同一个$q$生成GRPO条回复$o_1, o_2, ..., o_G $。

### 过程监督 VS 结果监督：过程奖励与末端奖励的对比

-  **结果监督（Outcome Supervision）**：只有输出序列结束才打一个奖励，如回答对 / 错、得分多少。GRPO 则把这个$r$ 同样分配给序列里每个 token。
- **过程监督（Process Supervision）**：对中间推理步骤也有打分（比如计算正确一步就 + 1，错误一步就 - 1）。那就得收集多个时刻的奖励，然后累加到每个 token 或步骤上，再做分组相对化。

在绝大多数简单场景下，初学者往往更容易先实现结果监督的版本，这也正好方便讲解 GRPO 的主干思路。

### 分组采样的实现：batch 内如何分组？

在实际操作中，我们往往会在一个 batch 中包含若干个问题$q$，对每个问题生成GRPO个答案。也就是说 batch 大小 = $B $，每个问题生成$GRPO$个候选，那么一次前向推理要生成$B*GRPO $条候选。然后，每个候选都送奖励模型$\mathrm{RM}$得到分数$r_i$。注意这样做推理开销不小，如果$GRPO$较大，会显著地增加生成次数，但换来的好处是，我们不再需要价值网络了。

### 实际伪代码示例

我们以**结果监督**为例，先给出一个简化版的伪代码，帮助你更好理解 GRPO 的操作流程。假设$\pi_\theta$是当前策略模型，$ \pi_{\text{ref}}$是参考模型（一般初始可设为和$\pi_\theta$同一个拷贝，用于算KL 正则），$\mathrm{RM}$是奖励模型。

```Python
# 请注意这只是简化的示例，忽略了各种超参数细节
# GPRO 伪代码 (结果监督)

for iteration in range(N_iterations):
    # 1) 设置参考模型 pi_ref <- pi_theta
    pi_ref = clone(pi_theta)
    
    for step in range(M_steps_per_iter):
        # 2) 从训练集中取一批问题 D_b
        D_b = sample_batch(train_dataset, batch_size=B)
        
        # 3) 让旧策略 pi_theta 生成 G 个输出
        #    o_i 表示第 i 个候选答案
        batch_outs = []
        for q in D_b:
            outs_for_q = []
            for i in range(G):
                o_i = sample(pi_theta, q)
                outs_for_q.append(o_i)
            batch_outs.append(outs_for_q)
        
        # 4) 对每个输出用奖励模型 RM 打分
        #    r_i = RM(q, o_i)
        #    同时做分组归一化
        #    r_i_tilde = (r_i - mean(r)) / std(r)
        #    赋值给 A_i (整条序列的优势)
        
        # 这里只是一种写法：对 batch 内每个 q 都做
        for outs_for_q in batch_outs:
            # outs_for_q 大小是 G
            r_list = [RM(q, o_i) for o_i in outs_for_q]
            mean_r = mean(r_list)
            std_r = std(r_list)
            if std_r == 0: std_r = 1e-8  # 避免除0
            
            for i, o_i in enumerate(outs_for_q):
                r_tilde = (r_list[i] - mean_r) / std_r
                # 把这个 r_tilde 记为 A(o_i) 用于后续计算
                # 也可以存在某个 data structure 里

        # 5) 根据 GPRO 目标函数做梯度更新
        #    关键是每个 token 的优势都用 A(o_i)
        #    并加上 KL 正则
        loss = compute_gpro_loss(pi_theta, pi_ref, batch_outs, r_tilde_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

在这个伪代码里，我们可以看到最关键的部分就是**每个问题都采样**$GRPO$**个输出**，分别打分，然后在该分组里做归一化。每个输出$o_i $的所有 token 共享一个相同的优势值 $\hat{A}_{i,t} = \tilde{r}_i$。然后再像 PPO 那样做 ratio + clip 的梯度更新。

这便完成了结果监督版本的 GRPO 训练循环。相比 PPO，差别在于：**不再需要一个大型的价值网络**来估计优劣，而是由分组对比来获得相对优势。



### 源码分析

我们再次分析下GRPO的损失

$$
\begin{array}{c}\mathcal{L}_{\mathrm{GRPO}}(\theta)=-\frac{1}{G} \sum_{i=1}^{G} \frac{1}{\left|o_{i}\right|} \\ \sum_{t=1}^{\left|o_{i}\right|}\left[\min \left(\frac{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)} \hat{A}_{i, t}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_{i, t}\right)\right. \\ -\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_{\theta} \| \pi_{\mathrm{ref}}\right]\end{array}
$$

GRPO loss看起来复杂，实际上，仅包含三部分：

1. 第一个连加的$G$为一个样本的采样数量，第二个$|o_i|$是第$i$条输出的采样长度
2. 在$min(⋅,⋅)$里，与标准PPO差异不大，这里的advantage需要提前计算$\hat{A}_{i, t}=\frac{r_{i}-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}$，在一条采样回答数据中对于不同的$t$优势值都一样的。另外这里的ratio对比的是新旧策略。 这个式子是token-level的。
3. KL项$\beta$因子控制约束力度，KL计算的是新模型和参考模型。



KL通常用于衡量两个概率分布的差异情况。标准的KL项为：

$$
\mathbb{D}_{K L}\left[\pi_{\theta}| | \pi_{r e f}\right]=-\log \frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}
$$

GRPO采用以下形式的KL

$$
\mathbb{D}_{K L}\left[\pi_{\theta} \| \pi_{r e f}\right]=\frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}-\log \frac{\pi_{r e f}\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta}\left(o_{i, t} \mid q, o_{i,<t}\right)}-1,
$$

**trl**库里面的**grpo_trainer.py**文件用于实现GRPO的训练流程，下面我们分析一下该文件的代码。

`class GRPOTrainer`该类为GRPO的类

为了尽量简单的解释清楚，所以只解释重要的代码

下面针对一些参数的来历做简单解释，不做过多讲解，更细致的信息请自行查看源码

`ref_model`：参考模型$ \pi_{\text{ref}}$

`model`：训练模型

`ref_per_token_logps`：参考模型输出的每个tokens的log概率（有$G$个回答）

`per_token_logps`：训练模型输出的每个tokens的log概率（有$G$个回答）

得到`ref_per_token_logps`和`per_token_logps`后可以计算KL散度

```Python
# 计算每个token的KL散度（正则化项）
per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1




```

然后获得相应的奖励

奖励函数可以为模型也可以为函数，R1中奖励模型为规则函数，因此我们主要看下规则函数部分的代码

```Python
 # 如果奖励函数不是模型，假设它是一个普通的函数，直接计算奖励
reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
for key in reward_kwargs:
    for example in inputs:
        reward_kwargs[key].extend([example[key]] * self.num_generations)
output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

 # 计算总奖励（将所有奖励函数的结果相加）
rewards = rewards_per_func.sum(dim=1)
# 计算按组平均的奖励
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
# 计算按组奖励的标准差
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
# 将奖励标准化，用于计算优势函数
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)  # 防止除以零


```

`reward_func`为定义的奖励函数，`prompts`为模型的输出，`completions`为ground truth

通过上面的代码，我们可以得到优势函数$\hat{A}_{i, t}=\frac{r_{i}-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}$

整体的损失函数如下所示：

```Python
# 计算每个token的损失，使用优势函数进行加权
per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
# 计算最终的损失，考虑KL散度正则化项
per_token_loss = -(per_token_loss - self.beta * per_token_kl)
# 计算每个token的损失并进行掩码操作，以忽略EOS后的token
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

HuggingFace实现的源码中并没有涉及到剪切（clip）操作，它通过**KL 散度惩罚**来避免策略更新过于剧烈，达到与公式中**clip**部分类似的效果。这是 GRPO 中的一个设计差异，虽然方法不同，但目的基本一致——保证训练的稳定性并防止过大的策略更新。

### 不同项目中奖励函数的设计

OpenR1项目中的GRPO算法详见：**`grpo.py`**

定义了两个奖励函数：**accuracy**和**format**

**accuracy**：用于评估答案正确性，回答正确奖励1，错误奖励为0

**format**：用于奖励格式，格式为`"^<think>.`*`?</think><answer>.`*`?</answer>$"`正确奖励1，错误奖励为0



下面这两个项目大致是一样的，英文版和中文版的区别，设计了五种奖励函数

**正确性奖励**，模型回答正确奖励2

**数字奖励**，模型推理出数字则奖励0.5

**硬格式奖励**，模型推理严格按照给定格式输出则奖励0.5

**软格式奖励**，模型推理按照给定格式输出则奖励0.5

**固定标签奖励**，模型推理输出包含给定的固定标签时则奖励0.125

[llm_related/deepseek_learn/deepseek_r1_train/deepseek_r1_train.py at main · wyf3/llm_related](https://github.com/wyf3/llm_related/blob/main/deepseek_learn/deepseek_r1_train/deepseek_r1_train.py)

[GRPO Llama-1B](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)



### 参考

[【DeepSeek】一文详解GRPO算法——为什么能减少大模型训练资源？_group relative policy optimization-CSDN博客](https://blog.csdn.net/qq_38961840/article/details/145384852)

[https://zhuanlan.zhihu.com/p/20812786520](https://zhuanlan.zhihu.com/p/20812786520)