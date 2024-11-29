import torch.nn as nn
import torch
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    hidden_size:int=4096
    intermediate_size:int=14336
    max_position_embeddings:int=8192
    num_attention_heads:int=32
    num_hidden_layers:int=32
    num_key_value_heads:int=8
    rms_norm_eps:float=1e-5
    vocab_size:int=128000 # this param should be equal to tokenizer.vocab_size



# class 

if __name__ == '__main__':

    batch_size = 2
    config = ModelConfig()

    x = torch.ones((batch_size, config.hidden_size), dtype=int)
    logger.debug(f'input shape: {x.shape}')
    tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    x = tok_emb(x)
    logger.debug(f'embedding shape: {x.shape}')