import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class LLaMaConfig:
    embed_dim:int=4096
    intermediate_size:int=14336
    max_position_embeddings:int=8192
    num_attention_heads:int=32
    num_blocks:int=32
    num_key_value_heads:int=8
    rms_norm_eps:float=1e-5
    vocab_size:int=128000 # this param should be equal to tokenizer.vocab_size

    num_attention_heads:int=32
    num_key_value_heads:int=8
    prob_dropout:float=0.1


class RMS_Norm(nn.Module):
    """
    RMS归一化,应当在先归一化再过子层
    """
    def __init__(self, config:LLaMaConfig) -> None:
        super(RMS_Norm, self).__init__()
        
        self.refactor = nn.Parameter(torch.ones(config.embed_dim))
        self.rms_norm_eps = config.rms_norm_eps
        
    def __norm(self, x):
        # 加了eps确保不会出现除以0的情况，注意这里rsqrt是取根号的倒数
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
        return x * rms
    
    def forward(self, x):
        x = self.__norm(x)
        x = x * self.refactor
        return x
    

class FeedForwardNet(nn.Module):
    """
    与GPT的FFN的不同在于把中间的relu换成了SwiGLU
    """
    def __init__(self, config:LLaMaConfig) -> None:
        super(FeedForwardNet, self).__init__()

        self.w_1 = nn.Linear(config.embed_dim, config.intermediate_size)
        self.w_2 = nn.Linear(config.embed_dim, config.intermediate_size)
        self.w_3 = nn.Linear(config.intermediate_size, config.embed_dim)

    def forward(self, x):

        y = self.w_1(x)
        g = F.silu(self.w_2(x))
        y = y*g
        y = self.w_3(y)
        return y


class GroupQueryAttention(nn.Module):
    """
    分组查询注意力,与多头注意力的区别在于，多个query共享1对key和value
    """
    def __init__(self, config:LLaMaConfig) -> None:
        super(GroupQueryAttention, self).__init__()

        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.embed_dim % config.num_attention_heads == 0 

        # query的head最多，所以单头的dim由query来决定，实际上kv的参数量少了
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.embed_dim // config.num_attention_heads


        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, self.head_dim*config.num_key_value_heads)
        self.v_proj = nn.Linear(config.embed_dim, self.head_dim*config.num_key_value_heads)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.prob_dropout)

    def forward(self, 
                query:torch.tensor, 
                key:torch.tensor, 
                value:torch.tensor, 
                mask:torch.tensor=None
                ):
        assert query.shape == key.shape == value.shape
        batch_size, seq_len, num_emb = query.shape

        # 映射
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        # print(Q.shape, K.shape, V.shape)

        # 拆分出头
        Q = torch.reshape(Q, (batch_size, seq_len, self.num_q_heads, self.head_dim))
        K = torch.reshape(K, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        V = torch.reshape(V, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        # print(Q.shape, K.shape, V.shape)

        # 翻转 head和seq两个维度
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        # print(Q.shape, K.shape, V.shape)

        # 复制k和v的头，用repeat_interleave而不是repeat
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        # print(Q.shape, K.shape, V.shape)

        # 之后就和正常的MHA一样
        atten_score = Q @ K.transpose(-1,-2) / math.sqrt(self.head_dim)
        if mask is not None:
            atten_score = atten_score.masked_fill(mask==0, float('-inf'))
        atten_weight = torch.softmax(atten_score, dim=-1)
        # print(atten_weight.shape)

        y = atten_weight @ V
        y = y.transpose(1,2).contiguous().reshape((batch_size, seq_len, -1))

        y = self.o_proj(y)
        y = self.dropout(y)

        return y 
    

class LLaMaBlock(nn.Module):
    """
    先RMS归一化，再过子层，再残差相加
    """
    def __init__(self, config:LLaMaConfig) -> None:
        super(LLaMaBlock, self).__init__()

        self.attention = GroupQueryAttention(config)
        self.norm_1 = RMS_Norm(config)
        self.ffn = FeedForwardNet(config)
        self.norm_2 = RMS_Norm(config)

    def forward(self, x):

        _x = x
        x = self.norm_1(x)
        x = self.attention(x, x, x)
        x = _x + x

        _x = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = _x + x

        return x


class LLaMaHeadNet(nn.Module):
    """
    head
    """
    def __init__(self, config:LLaMaConfig) -> None:
        super(FeedForwardNet, self).__init__()

        self.net = nn.Sequential(
            RMS_Norm(config),
            nn.Linear(config.embed_dim, config.vocab_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class LLaMa(nn.Module):

    def __init__(self, config:LLaMaConfig) -> None:
        super(FeedForwardNet, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = [LLaMaBlock(config) for _ in range(config.num_blocks)]
        self.head = LLaMaHeadNet(config)

    def forward(self, x):
        x = self.tok_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x
        

# class 

if __name__ == '__main__':

    batch_size = 2
    config = LLaMaConfig()

    x = torch.ones((batch_size, 10, config.embed_dim), dtype=torch.float32)
    print(x.shape)
    # logger.debug(f'input shape: {x.shape}')
    # model = GroupQueryAttention(config)
    model = LLaMaBlock(config)
    y = model(x)
    print(y.shape)