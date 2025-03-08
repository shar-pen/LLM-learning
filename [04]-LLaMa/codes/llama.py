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
    num_key_value_heads:int=8
    num_blocks:int=32
    rms_norm_eps:float=1e-5
    vocab_size:int=128000 # this param should be equal to tokenizer.vocab_size

    num_attention_heads:int=32
    num_key_value_heads:int=8
    prob_dropout:float=0.1


# 旋转位置编码

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # dim应该等同于head dim
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    angel_split_num = dim // 2
    angel_splits = torch.arange(0, angel_split_num).float() / angel_split_num
    freqs = 1.0 / (theta ** angel_splits)
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    # 根据偏转角度转换为复数
    # 假设 freqs = [x, y] 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # # freqs_cis is (seq_len, head_dim/2, 2), e.g. (8, 64, 2)
    # freqs_cis_real_and_imag = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    freqs_cis_real = freqs.cos()
    freqs_cis_imag = freqs.sin()
    freqs_cis_real_and_imag = torch.stack([freqs_cis_real, freqs_cis_imag], dim=-1)
    return freqs_cis_real_and_imag


def apply_rope(x, freqs_cis_real_and_imag):
    # QKV shape (batch_size, num_q_heads, seq_len, head_dim)
    # QKV shape (batch_size, num_q_heads, seq_len, head_dim/2, 2)
    x = torch.reshape(x, (*x.shape[:-1], -1, 2))
    
    # freqs_cis is (max_len, head_dim/2, 2)
    # truncate to support variable sizes
    T = x.size(-3)
    freqs_cis_real_and_imag = freqs_cis_real_and_imag[:T]
    

    y = torch.stack(
        [
            # 这是 q_0 * cos - q_1 * sin
            x[..., 0] * freqs_cis_real_and_imag[..., 0] - x[..., 1] * freqs_cis_real_and_imag[..., 1],
            # 这是 q_1 * cos + q_0 * sin
            x[..., 1] * freqs_cis_real_and_imag[..., 0] + x[..., 0] * freqs_cis_real_and_imag[..., 1],
        ],
        -1,
    )
    y = y.flatten(start_dim=3)
    return y



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
                freqs_cis_real_and_imag:torch.tensor, 
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
        # Q shape (batch_size, seq_len, num_q_heads, head_dim)
        # KV shape (batch_size, seq_len, num_kv_heads, head_dim)
        Q = torch.reshape(Q, (batch_size, seq_len, self.num_q_heads, self.head_dim))
        K = torch.reshape(K, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        V = torch.reshape(V, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        # print(Q.shape, K.shape, V.shape)

        # 翻转 head和seq两个维度
        # Q shape (batch_size, num_q_heads, seq_len, head_dim)
        # KV shape (batch_size, num_kv_heads, seq_len, head_dim)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        # print(Q.shape, K.shape, V.shape)

        # 复制k和v的头，用repeat_interleave而不是repeat
        # QKV shape (batch_size, num_q_heads, seq_len, head_dim)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        # print(Q.shape, K.shape, V.shape)
        
        # 应用旋转位置编码
        Q = apply_rope(Q, freqs_cis_real_and_imag)
        K = apply_rope(K, freqs_cis_real_and_imag)

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

        self.norm_pre_attention = RMS_Norm(config)
        self.attention = GroupQueryAttention(config)
        self.norm_pre_ffn = RMS_Norm(config)
        self.ffn = FeedForwardNet(config)

    def forward(self, x, precompute_freqs_cis, mask):

        _x = x
        x = self.norm_pre_attention(x)
        x = self.attention(x, x, x, precompute_freqs_cis, mask)
        x = _x + x

        _x = x
        x = self.norm_pre_ffn(x)
        x = self.ffn(x)
        x = _x + x

        return x


class LLaMaHeadNet(nn.Module):
    """
    head
    """
    def __init__(self, config:LLaMaConfig) -> None:
        super(LLaMaHeadNet, self).__init__()

        self.net = nn.Sequential(
            RMS_Norm(config),
            nn.Linear(config.embed_dim, config.vocab_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class LLaMa(nn.Module):

    def __init__(self, config:LLaMaConfig) -> None:
        super(LLaMa, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.precompute_freqs_cis = precompute_freqs_cis(config.embed_dim//config.num_attention_heads, config.max_position_embeddings)
        self.blocks = nn.ModuleList(LLaMaBlock(config) for _ in range(config.num_blocks))
        self.head = LLaMaHeadNet(config)

    def forward(self, x:torch.tensor, mask:torch.tensor=None):
        
        causal_mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]), dtype=int)
        causal_mask = torch.tril(causal_mask).to(x.device)
        if mask is not None:
            assert x.shape[:2] == mask.shape
            padding_mask = mask.unsqueeze(dim=1)
            padding_mask = padding_mask.repeat(1,mask.shape[-1],1)
            causal_mask = causal_mask & padding_mask
        # 扩展维度对应head那一维
        causal_mask = causal_mask.unsqueeze(dim=1)
        # print(causal_mask)
        x = self.tok_emb(x)
        for block in self.blocks:
            x = block(x, self.precompute_freqs_cis.to(x.device), causal_mask)
        x = self.head(x)
        return x
        

if __name__ == '__main__':

    batch_size = 2
    config = LLaMaConfig(
        embed_dim=1024,
        intermediate_size=1024*4,
        max_position_embeddings=1024,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_blocks=2,
        vocab_size=5000,  # verify the tokenizer has the same vocab szie
    )
    print(config)

    x = torch.ones((batch_size, 10), dtype=torch.int).to('cuda')
    mask = torch.tensor([[1]*9+[0], [1]*5+[0]*5]).to('cuda')
    print(x.shape)
    model = LLaMa(config).to('cuda')
    y = model(x, mask)
    print(y.shape)