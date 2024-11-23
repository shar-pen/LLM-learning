import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    
    vocab_size: int=1000
    n_emb: int=256
    n_head: int=8
    prob_dropout: float=0.1
    n_hidden: int=1024
    n_block: int=4
    max_len: int=100
    eps: float=1e-6
    device: str='cuda'
    

class AttentionMasker:
    """
    store mask matrix in one class for all block to share.
    """
    def __init__(self, config) -> None:
        
        self.causal_mask = torch.tril(torch.ones(config.max_len, config.max_len)).unsqueeze(0).unsqueeze(0)
        
    def get_casual_mask(self, x):
        bs, sl, ne = x.shape
        
        return self.causal_mask[:,:,:sl,:sl]


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, config) -> None:
        super(CausalSelfAttention, self).__init__()
        
        assert config.n_emb % config.n_head == 0
        
        self.n_emb = config.n_emb
        self.n_head = config.n_head
        self.n_emb_per_head = config.n_emb // config.n_head
        
        self.qkv_proj = nn.Linear(config.n_emb, 3*config.n_emb)
        self.dropout = nn.Dropout(config.prob_dropout)
        self.o_proj = nn.Linear(config.n_emb, config.n_emb)
        
    def forward(self, x:torch.tensor, mask:torch.tensor=None):
        bs, sl, ne = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, self.n_emb, dim=-1)
        
        # (bs, nh, sl, neph)
        q = torch.reshape(q, (bs, sl, self.n_head, self.n_emb_per_head)).transpose(1,2)
        k = torch.reshape(k, (bs, sl, self.n_head, self.n_emb_per_head)).transpose(1,2)
        v = torch.reshape(v, (bs, sl, self.n_head, self.n_emb_per_head)).transpose(1,2)
        
        # (bs, nh, sl, sl)
        # causal self-attention; Self-attend: (bs, nh, sl, neph) x (bs, nh, neph, sl) -> (bs, nh, sl, sl)
        att_score = q @ k.transpose(-2,-1) / math.sqrt(self.n_emb_per_head)
        if mask is not None:
            att_score = att_score.masked_fill(mask==0, float('-inf'))
        att_weight = F.softmax(att_score, dim=-1)
        
        # (bs, nh, sl, neph) => (bs, sl, ne)
        y = att_weight @ v
        y = y.transpose(1,2).contiguous().reshape((bs, sl, ne))
        
        # output projection and dropout
        y = self.o_proj(y)
        y = self.dropout(y)
        
        return y
        

class LayerNorm(nn.Module):
    
    def __init__(self, config) -> None:
        super(LayerNorm, self).__init__()
        self.eps = config.eps
        self.refactor_a = nn.Parameter(torch.ones(config.n_emb))
        self.refactor_b = nn.Parameter(torch.zeros(config.n_emb))
        
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x-mean) / (std + self.eps)
        x = x * self.refactor_a + self.refactor_b
        return x
    
    
class FeedForwardNet(nn.Module):
    
    def __init__(self, config) -> None:
        super(FeedForwardNet, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_emb, config.n_hidden),
            nn.ReLU(),
            nn.Linear(config.n_hidden, config.n_emb)
        )
    
    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    
    def __init__(self, config) -> None:
        super(Block, self).__init__()
        
        self.att = CausalSelfAttention(config)
        self.norm1 = LayerNorm(config)
        self.ffn = FeedForwardNet(config)
        self.norm2 = LayerNorm(config)

    def forward(self, x, mask):
        _x = x
        x = self.att(x, mask)
        x = x + _x
        x = self.norm1(x)
        
        _x = x
        x = self.ffn(x)
        x = x + _x
        x = self.norm2(x)
        
        return x
         
    
class TokenEmbeddding(nn.Embedding):
    
    def __init__(self, config) -> None:
        super().__init__(config.vocab_size, config.n_emb, config.padding_idx)
    

class PositionalEmbedding(nn.Module):
    
    def __init__(self, config) -> None:
        # This class takes device due to it being just a matrix, rather than torch.nn layers
        super(PositionalEmbedding, self).__init__()
        
        self.encoding = torch.zeros(config.max_len, config.d_model, device=config.device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, config.max_len, device=config.device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, config.n_emb, step=2, device=config.device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / config.n_emb)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / config.n_emb)))
        
    def forward(self, input):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = input.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]


class GPT(nn.Module):
    
    def __init__(self, config) -> None:
        super(GPT, self).__init__()
        
        self.wte = nn.Embedding(config.vocab_size, config.n_emb)
        self.wpe = nn.Embedding(config.vocab_size, config.n_emb)
        self.masker = AttentionMasker(config)
        self.blocks = [Block(config) for _ in range(config.n_block)]
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size)
        
    def forward(self, x):
        x = self.wte(x) + self.wpe(x)
        mask = self.masker.get_casual_mask(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.lm_head(x)
        return x
        
        
        
if __name__ == '__main__':
    
    config = ModelConfig()
    print(config)
    
    model = GPT(config)
    x = torch.ones((2,10),dtype=int)
    print(model(x).shape)
    