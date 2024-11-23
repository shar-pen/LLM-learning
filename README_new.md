# LLM-learning

Here's the content from the image converted into Markdown:

---

### 1. 手撕Transformer

- Transformer模型架构
- 输入position Encoding
- 注意力Attention原理和实现
- Encoder-Decoder实现
- 掩码(Masked)层
- 实践：英文文本翻译 https://github.com/dt-3t/Transformer-en-to-cn.git https://blog.csdn.net/qq_36396406/article/details/132384993

### 2. 手撕GPT

- GPT1/2/3/3.5/4论文解析
- GPT模型架构
- BPE编码原理
- Generate阶段
- FlashAttention加速
- 实践：GPT预测处理单词



https://github.com/karpathy/nanoGPT/tree/master



### 4. 手撕LLaMA

- LLaMA论文解析
- LLaMA模型架构
- RMS_NORM标准化
- SwiGLU激活函数
- KV Cache推理加速、MOA、GQA
- 实践：LLaMA预测训练

### 5. 手撕Alpaca

- Instruction Finetune原理
- Alpaca模型
- self-instruct实现细节
- Prompt/Prefix/Adapter微调原理
- finetune/adapter/lora效率对比
- CoT/ToT
- 实践：Alpaca Instruction微调

### 6. 手撕LoRA

- LoRA/QLoRA论文解析
- LoRA算法推导
- LoRA实现和细节讲解
- QLoRA NF4量化双重化
- 实践：LLaMA2+QLoRA微调

### 7. 手撕Chinese-LLaMA2

- Chinese-LLaMA2论文讲解
- 中文Tokenizer讲解及扩展
- 中文模型预训练
- 二次预训练+SFT
- baichuan2论文解读
- 实践：中文模型微调

### 8. 手撕chatLLaMA-Agent

- 多对话场景意图分类
- Agent原理
- RAG背景原理
- 文本处理及任务链传递
- 双向多轮对话意图推测
- 实践：多对话场景话题生成

### 9. 手撕RL-PPO

- 强化深度学习技术综述
- MDP/QV/计算
- DQN算法
- Policy Gradient
- Actor-Critic
- PPO算法
- PyTorch实现DQN、PG、A2C、PPO

### 10. 手撕RLHF

- LLaMA2/GPT3.5论文解析
- SFT
- Rewards-model
- MARL参数共享
- RLHF-PPO详细算法流程
- RLHF-PPO Loss计算
- FTRL+RHF训练

### 实践：垂直大模型Chatbot

- 医疗大模型
- 中文Tokenizer讲解
- 二次预训练
- SFT QLoRA微调
- Reward Model基准训练
- RLHF-PPO
- Langchain
- 长对话+RAG多轮生成
