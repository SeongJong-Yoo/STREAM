import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from .positional_encoder import PositionalEncoder, PositionalEncoderLearned, RotaryEmbedding

def build_position_encoding(dim, position_embedding='learned'):
    if position_embedding == 'learned':
        return PositionalEncoderLearned(dim, dropout=0.1)
    elif position_embedding == 'sine':
        return PositionalEncoder(dim, dropout=0.1)
    elif position_embedding == 'rotary':
        return RotaryEmbedding(dim)
    else:
        return


def get_clone(module):
    return copy.deepcopy(module)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "swish":
        return F.silu
    raise RuntimeError(F"activation should be relu/gelu/swish, not {activation}.")    



def lengths_to_mask(lengths, device, max_len=None):
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    