from PIL.Image import module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from einops import rearrange

import numpy as np
from typing import Optional

from .utility import get_clones, get_activation_fn


def modulate(x, shift, scale):
    size = scale.shape[0]
    if size != x.shape[0]:
        diff = x.shape[0] - size
        return torch.cat((x[:diff], x[diff:] * (1 + scale) + shift)) # This is for VAE
    return x * (1 + scale) + shift
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 encoder_layer, 
                 num_layers, 
                 norm=None):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        if norm is not None:
            self.norm = get_clones(norm, num_layers)
        else:
            self.norm = None

    def forward(self, 
                tgt, 
                cond=None,
                mask=None, 
                tgt_key_padding_mask=None, 
                pos=None,
                position_index=None):
        output = tgt

        for i, layer in enumerate(self.layers):
            output = layer(output, 
                           cond=cond,
                           tgt_mask=mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask, 
                           pos=pos,
                           position_index=position_index)

            if self.norm is not None:
                output = self.norm[i](output)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 decoder_layer, 
                 num_layers, 
                 norm=None):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)   
        self.num_layers = num_layers
        self.norm = get_clones(norm, num_layers)

    def forward(self, 
                tgt, 
                memory, 
                cond=None,
                tgt_mask=None, 
                memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, 
                pos=None, 
                query_pos=None,
                position_index=None):
        
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output, 
                           memory, 
                           cond=cond,
                           tgt_mask=tgt_mask, 
                           memory_mask=memory_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask, 
                           memory_key_padding_mask=memory_key_padding_mask, 
                           pos=pos, 
                           query_pos=query_pos,
                           position_index=position_index)

            if self.norm is not None:
                output = self.norm[i](output)
            
            intermediate.append(output)
        
        return output #, intermediate[-3:-1]

class HierarchyTransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None):
        super().__init__()
        self.num_layers = num_layers

        self.motion_layer = get_clones(encoder_layer, num_layers)
        self.acc_layer = get_clones(encoder_layer, num_layers)
        self.norm_motion = get_clones(norm, num_layers)
        self.norm_acc = get_clones(norm, num_layers) 
    
    def forward(self, 
                motion,
                acc, 
                cond,
                mask=None, 
                tgt_key_padding_mask=None, 
                pos=None):
        for i in range(self.num_layers):
            output_acc = self.acc_layer[i](acc,
                                           cond=cond,
                                           tgt_mask=mask, 
                                           tgt_key_padding_mask=tgt_key_padding_mask, 
                                           pos=pos)
            acc = self.norm_acc[i](output_acc)
            output_motion = self.motion_layer[i](motion,
                                                 cond=cond,
                                                 tgt_mask=mask, 
                                                 tgt_key_padding_mask=tgt_key_padding_mask, 
                                                 pos=pos)
            motion = self.norm_motion[i](output_motion + acc)

        return motion, acc


class HierarchyTransformerDecoder(nn.Module):
    def __init__(self, 
                 decoder_layer, 
                 num_layers, 
                 norm=None):
        super().__init__()
        self.num_layers = num_layers

        self.motion_layer = get_clones(decoder_layer, num_layers)   
        self.acc_layer = get_clones(decoder_layer, num_layers)
        self.norm_motion = get_clones(norm, num_layers)
        self.norm_acc = get_clones(norm, num_layers)

    def forward(self, 
                tgt, 
                memory, 
                cond,
                tgt_mask=None, 
                memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, 
                pos=None, 
                query_pos=None):
        
        for i in range(self.num_layers):
            output_acc = self.acc_layer[i](tgt, 
                                       memory, 
                                       cond=cond,
                                       tgt_mask=tgt_mask, 
                                       memory_mask=memory_mask, 
                                       tgt_key_padding_mask=tgt_key_padding_mask, 
                                       memory_key_padding_mask=memory_key_padding_mask, 
                                       pos=pos, 
                                       query_pos=query_pos)
            acc = self.norm_acc[i](output_acc)
            output_motion = self.motion_layer[i](tgt, 
                                       memory, 
                                       cond=cond,
                                       tgt_mask=tgt_mask, 
                                       memory_mask=memory_mask, 
                                       tgt_key_padding_mask=tgt_key_padding_mask, 
                                       memory_key_padding_mask=memory_key_padding_mask, 
                                       pos=pos, 
                                       query_pos=query_pos)
            motion = self.norm_motion[i](output_motion + acc)

        return motion, acc

class TransformerDualDecoder(nn.Module):
    def __init__(self, 
                 decoder_layer, 
                 num_layers,
                 latent_dim, 
                 norm):
        super().__init__()
        self.layers_1 = get_clones(decoder_layer, num_layers)   
        self.layers_2 = get_clones(decoder_layer, num_layers)   
        self.num_layers = num_layers
        self.norm_1 = get_clones(norm, num_layers)
        self.norm_2 = get_clones(norm, num_layers)

    def forward(self, 
                tgt, 
                memory, 
                cond=None,
                tgt_mask=None, 
                memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, 
                pos=None, 
                query_pos=None):

        tgt_1 = tgt
        tgt_2 = memory
        for i in range(self.num_layers):
            output_1 = self.layers_1[i](tgt_1, 
                                      tgt_2, 
                                      cond=cond,
                                      tgt_mask=tgt_mask, 
                                      memory_mask=memory_mask, 
                                      tgt_key_padding_mask=tgt_key_padding_mask, 
                                      memory_key_padding_mask=memory_key_padding_mask, 
                                      pos=pos, 
                                      query_pos=query_pos)
            output_2 = self.layers_2[i](tgt_2, 
                                        tgt_1, 
                                        cond=cond,
                                        tgt_mask=tgt_mask, 
                                        memory_mask=memory_mask, 
                                        tgt_key_padding_mask=tgt_key_padding_mask, 
                                        memory_key_padding_mask=memory_key_padding_mask, 
                                        pos=pos, 
                                        query_pos=query_pos)

            tgt_1 = self.norm_1[i](output_1)
            tgt_2 = self.norm_2[i](output_2)
            
        output = torch.cat((tgt_1, tgt_2), dim=-1)
        return output

class DeepDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dim_ratio=2,
                 dropout=0.1,
                 activation='relu',
                 modulator='film',
                 conv=False,
                 rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_ratio = dim_ratio
        self.dropout = dropout
        self.activation = activation
        self.modulator = modulator
        self.conv = conv

        if self.modulator == 'film':
            self.encoder_film_layer = DenseFiLM(d_model)
            self.decoder_film_layer = DenseFiLM(d_model)
            self.last_film_layer = DenseFiLM(d_model)
        elif self.modulator == 'AdaLN':
            self.encoder_AdaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )
            self.decoder_AdaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )
            self.last_AdaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )

        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)

        if self.conv:
            self.linear1 = nn.Conv1d(d_model, self.dim_ratio * d_model, kernel_size=3, stride=1, padding=1)
            self.linear2 = nn.Conv1d(self.dim_ratio * d_model, d_model, kernel_size=3, stride=1, padding=1)
        else:
            self.linear1 = nn.Linear(d_model, self.dim_ratio * d_model)
            self.linear2 = nn.Linear(self.dim_ratio * d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.initialize_weights()

    def initialize_weights(self):
        if self.modulator == 'AdaLN':
            nn.init.constant_(self.encoder_AdaLN[-1].weight, 0)
            nn.init.constant_(self.encoder_AdaLN[-1].bias, 0)
            nn.init.constant_(self.decoder_AdaLN[-1].weight, 0)
            nn.init.constant_(self.decoder_AdaLN[-1].bias, 0)
            nn.init.constant_(self.last_AdaLN[-1].weight, 0)
            nn.init.constant_(self.last_AdaLN[-1].bias, 0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        tensor = pos.rotate_queries_or_keys(tensor)
        return tensor

    def forward(self, tgt, memory, cond, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                pos=None, query_pos=None, position_index=None):
        
        # Self model
        tgt = self.norm1(tgt)
        tgt = self.with_pos_embed(tgt, pos)
        if self.modulator == 'film':
            tgt = featurewise_affine(tgt, self.encoder_film_layer(cond))
        elif self.modulator == 'AdaLN':
            # latent_size = tgt.shape[0]
            # window_size = cond.shape[0] // latent_size
            # modulated_cond = torch.stack([torch.mean(cond[i*window_size:(i+1)*window_size], dim=0) for i in range(latent_size)])
            shift_msa, scale_msa, gate_msa = self.encoder_AdaLN(cond).chunk(3, dim=-1)
            tgt = modulate(tgt, shift_msa, scale_msa)

        tgt2 = self.self_attn(tgt, tgt, tgt,
                            attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        if self.modulator == 'AdaLN':
            if gate_msa.shape[0] != tgt2.shape[0]:
                diff = tgt2.shape[0] - gate_msa.shape[0]
                tgt2 = torch.cat((tgt2[:diff], gate_msa * tgt2[diff:]), dim=0)
            else:
                tgt2 = gate_msa * tgt2
            
        tgt = tgt + self.dropout1(tgt2)  # Proper residual

        # Global model
        tgt = self.norm2(tgt)
        if self.modulator == 'film':
            tgt = featurewise_affine(tgt, self.decoder_film_layer(cond))
        elif self.modulator == 'AdaLN':
            shift_mlp, scale_mlp, gate_mlp = self.decoder_AdaLN(cond).chunk(3, dim=-1)
            tgt = modulate(tgt, shift_mlp, scale_mlp)

        # Cross-attention with proper residual
        tgt = self.with_pos_embed(tgt, query_pos)
        memory = self.with_pos_embed(memory, pos)
        if memory_mask is not None:
            if memory_mask.ndim == 3 and memory_mask.shape[0] != tgt.shape[1] * self.num_head:
                memory_mask = torch.tile(memory_mask.unsqueeze(1), (1, self.num_head, 1, 1)).reshape(-1, memory_mask.shape[1], memory_mask.shape[2])
        tgt2 = self.cross_attn(tgt, memory, memory,
                            attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)[0]
        if self.modulator == 'AdaLN':
            if gate_mlp.shape[0] != tgt2.shape[0]:
                diff = tgt2.shape[0] - gate_mlp.shape[0]
                tgt2 = torch.cat((tgt2[:diff], gate_mlp * tgt2[diff:]), dim=0)
            else:
                tgt2 = gate_mlp * tgt2

        tgt = tgt + self.dropout2(tgt2)  # Proper residual
        tgt_residual = tgt
        
        # FFN with proper residual
        tgt = self.norm3(tgt)
        if self.modulator == 'film':
            tgt = featurewise_affine(tgt, self.last_film_layer(cond))
        elif self.modulator == 'AdaLN':
            shift_mlp, scale_mlp, gate_mlp = self.last_AdaLN(cond).chunk(3, dim=-1)
            tgt = modulate(tgt, shift_mlp, scale_mlp)  # Use norm3 here

        if self.conv:
            tgt = tgt.permute(1, 2, 0)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        if self.conv:
            tgt2 = tgt2.permute(2, 0, 1)
        if self.modulator == 'AdaLN':
            if gate_mlp.shape[0] != tgt2.shape[0]:
                diff = tgt2.shape[0] - gate_mlp.shape[0]
                tgt2 = torch.cat((tgt2[:diff], gate_mlp * tgt2[diff:]), dim=0)
            else:
                tgt2 = gate_mlp * tgt2
        tgt = tgt_residual + self.dropout3(tgt2)  # Proper residual
        
        return tgt

class DeepEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dim_ratio=2,
                 dropout=0.1,
                 activation='relu',
                 modulator='film',
                 rope=None,
                 conv=False):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_ratio = dim_ratio
        self.dropout = dropout
        self.activation = activation
        self.modulator = modulator
        self.conv = conv

        if self.modulator == 'film':
            self.encoder_film_layer = DenseFiLM(d_model)
            self.last_film_layer = DenseFiLM(d_model)
        elif self.modulator == 'AdaLN':
            self.encoder_AdaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )
            self.last_AdaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )

        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)

        if self.conv:
            self.linear1 = nn.Conv1d(d_model, self.dim_ratio * d_model, kernel_size=3, stride=1, padding=1)
            self.linear2 = nn.Conv1d(self.dim_ratio * d_model, d_model, kernel_size=3, stride=1, padding=1)
        else:
            self.linear1 = nn.Linear(d_model, self.dim_ratio * d_model)
            self.linear2 = nn.Linear(self.dim_ratio * d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.initialize_weights()

    def initialize_weights(self):
        if self.modulator == 'AdaLN':
            nn.init.constant_(self.encoder_AdaLN[-1].weight, 0)
            nn.init.constant_(self.encoder_AdaLN[-1].bias, 0)
            nn.init.constant_(self.last_AdaLN[-1].weight, 0)
            nn.init.constant_(self.last_AdaLN[-1].bias, 0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        tensor = pos.rotate_queries_or_keys(tensor)
        return tensor

    def forward(self, tgt, cond,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                pos=None,
                query_pos=None,
                position_index=None):
        
        # Self model
        tgt = self.norm1(tgt)
        tgt = self.with_pos_embed(tgt, pos)
        if self.modulator == 'film':
            tgt = featurewise_affine(tgt, self.encoder_film_layer(cond))
        elif self.modulator == 'AdaLN':
            shift_msa, scale_msa, gate_msa = self.encoder_AdaLN(cond).chunk(3, dim=-1)
            tgt = modulate(tgt, shift_msa, scale_msa)

        tgt2 = self.self_attn(tgt, tgt, tgt,
                            attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        if self.modulator == 'AdaLN':
            if gate_msa.shape[0] != tgt2.shape[0]:
                diff = tgt2.shape[0] - gate_msa.shape[0]
                tgt2 = torch.cat((tgt2[:diff], gate_msa * tgt2[diff:]), dim=0)
            else:
                tgt2 = gate_msa * tgt2
            
        tgt = tgt + self.dropout1(tgt2)  
        tgt_residual = tgt
        
        # FFN with proper residual
        tgt = self.norm2(tgt)
        if self.modulator == 'film':
            tgt = featurewise_affine(tgt, self.last_film_layer(cond))
        elif self.modulator == 'AdaLN':
            shift_mlp, scale_mlp, gate_mlp = self.last_AdaLN(cond).chunk(3, dim=-1)
            tgt = modulate(tgt, shift_mlp, scale_mlp) 
        
        if self.conv:
            tgt = tgt.permute(1, 2, 0)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        if self.conv:
            tgt2 = tgt2.permute(2, 0, 1)
        if self.modulator == 'AdaLN':
            if gate_mlp.shape[0] != tgt2.shape[0]:
                diff = tgt2.shape[0] - gate_mlp.shape[0]
                tgt2 = torch.cat((tgt2[:diff], gate_mlp * tgt2[diff:]), dim=0)
            else:
                tgt2 = gate_mlp * tgt2
        tgt = tgt_residual + self.dropout2(tgt2) 
        
        return tgt


class MixDecoder(nn.Module):
    def __init__(self, 
                 decoder_layer, 
                 num_layers, 
                 norm=None):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)   
        self.num_layers = num_layers
        self.norm = get_clones(norm, num_layers)

    def forward(self, 
                tgt, 
                memory, 
                tgt_mask=None, 
                memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, 
                pos=None, 
                query_pos=None):
        
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output, 
                           memory, 
                           tgt_mask=tgt_mask, 
                           memory_mask=memory_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask, 
                           memory_key_padding_mask=memory_key_padding_mask, 
                           pos=pos, 
                           query_pos=query_pos)

            if self.norm is not None:
                output = self.norm[i](output)
            
            intermediate.append(output)
        
        return output, intermediate[-3:-1]
    
class AdaLNTransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu'):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.AdaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 6, bias=True)
        )

        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(self, 
                src, 
                cond,
                tgt_mask=None,
                memory_mask=None, 
                memory_key_padding_mask=None, 
                tgt_key_padding_mask=None,
                pos=None,
                query_pos=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.AdaLN(cond).chunk(6, dim=-1)
        modulated_src = modulate(self.norm1(src), shift_msa, scale_msa)
        src2 = self.self_attn(modulated_src, modulated_src , modulated_src, 
                              attn_mask=memory_mask, 
                              key_padding_mask=memory_key_padding_mask)[0]
        src2 = gate_msa.unsqueeze(1) * src2
        src = src + self.dropout1(src2)

        modulated_src = modulate(self.norm2(src), shift_mlp, scale_mlp)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(modulated_src))))
        src2 = gate_mlp.unsqueeze(1) * src2
        src = src + self.dropout2(src2)

        return src


class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_head, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 activation="relu",
                 rope=None,
                 conv=False):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.rope = rope
        self.conv = conv

        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)

        if self.conv:
            self.linear1 = nn.Conv1d(d_model, dim_feedforward, kernel_size=3, stride=1, padding=1)
            self.linear2 = nn.Conv1d(dim_feedforward, d_model, kernel_size=3, stride=1, padding=1)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        tensor = pos.rotate_queries_or_keys(tensor)
        return tensor

    def apply_rope(self, q, k, position_index):
        T, B, D = q.shape
        q = q.reshape(T, B, self.num_head, D//self.num_head)
        k = k.reshape(T, B, self.num_head, D//self.num_head)
        q, k = self.rope(q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), position_index)
        q = q.permute(2, 0, 1, 3).reshape(T, B, D)
        k = k.permute(2, 0, 1, 3).reshape(T, B, D)
        return q, k

    def forward(self, 
                src, 
                cond=None,
                tgt_mask=None, 
                tgt_key_padding_mask=None, 
                pos=None,
                position_index=None):
        q = k = self.with_pos_embed(src, pos)
        if self.rope is not None:
            q, k = self.apply_rope(q, k, position_index)
            
        src2 = self.self_attn(q, k , value=src, 
                              attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.conv:
            src = src.permute(1, 2, 0)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        if self.conv:
            src = src.permute(2, 0, 1)
            src2 = src2.permute(2, 0, 1)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_head, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 activation='relu',
                 rope=None,
                 conv=False):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.rope = rope
        self.conv = conv
        
        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)

        if self.conv:
            self.linear1 = nn.Conv1d(d_model, dim_feedforward, kernel_size=3, stride=1, padding=1)
            self.linear2 = nn.Conv1d(dim_feedforward, d_model, kernel_size=3, stride=1, padding=1)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        tensor = pos.rotate_queries_or_keys(tensor)
        return tensor

    def apply_rope(self, q, k, position_index):
        T, B, D = q.shape
        q = q.reshape(T, B, self.num_head, D//self.num_head)
        k = k.reshape(T, B, self.num_head, D//self.num_head)
        q, k = self.rope(q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), position_index)
        q = q.permute(2, 0, 1, 3).reshape(T, B, D)
        k = k.permute(2, 0, 1, 3).reshape(T, B, D)
        return q, k

    def forward(self, 
                tgt, 
                memory, 
                cond=None,
                tgt_mask=None, 
                memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, 
                pos=None, 
                query_pos=None,
                position_index=None):
        
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.rope is not None:
            q, k = self.apply_rope(q, k, position_index)
        tgt2 = self.self_attn(q, k, value=tgt, 
                              attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if memory_mask is not None:
            if memory_mask.ndim == 3:
                if memory_mask.shape[0] != tgt.shape[1] * self.num_head:
                    memory_mask = torch.tile(memory_mask.unsqueeze(1), (1, self.num_head, 1, 1)).reshape(-1, memory_mask.shape[1], memory_mask.shape[2])
        tgt = self.with_pos_embed(tgt, query_pos)
        memory = self.with_pos_embed(memory, pos)
        key = memory
        if self.rope is not None:
            tgt, key = self.apply_rope(tgt, memory, position_index)
        tgt2 = self.cross_attn(query=tgt,
                               key=key,
                               value=memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        if self.conv:
            tgt2 = tgt2.permute(1, 2, 0)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        if self.conv:
            tgt = tgt.permute(2, 0, 1)
            tgt2 = tgt2.permute(2, 0, 1)
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu'):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward
        self.activation = activation

        self.cross_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)

        self.norms = get_clones(nn.LayerNorm(d_model), 2)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropouts = get_clones(nn.Dropout(dropout), 2)
        self.activation = get_activation_fn(activation)

    def with_pose_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                memory,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        
        tgt = self.norms[0](tgt)
        tgt2, att_weights = self.cross_attn(query=self.with_pose_embed(tgt, query_pos),
                               key=self.with_pose_embed(memory, pos),
                               value=memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropouts[0](tgt2)
        tgt2 = self.norms[1](tgt)
        tgt2 = self.linear2(self.dropouts[0](self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropouts[1]

        return tgt, att_weights

        

class DenseFiLM(nn.Module):
    """
    DenseFiLM layer from EDGE: https://github.com/Stanford-TML/EDGE?tab=readme-ov-file
    """
    def __init__(self, dim, batch_first=False):
        super().__init__()
        self.dim = dim
        self.block = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, dim * 2)
        )
        self.norm = nn.LayerNorm(dim*2)
        self.batch_first = batch_first

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = self.norm(pos_encoding)
        if self.batch_first:
            if pos_encoding.ndim == 2:
                pos_encoding = rearrange(pos_encoding, 'b d -> b 1 d')
        else:
            if pos_encoding.ndim == 2:
                pos_encoding = rearrange(pos_encoding, 'b d -> 1 b d')
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift
    
def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class ResNetBlock(nn.Module):
    """
    ResNet block from diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim,
                 dropout=0.1,
                 groups=32,
                 activation='swish'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout
        self.activation = activation

        self.norm1 = nn.GroupNorm(groups, in_channels, affine=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = nn.Linear(time_emb_dim, time_emb_dim)

        self.norm2 = nn.GroupNorm(groups, out_channels, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = get_activation_fn(activation)


    def forward(self, input, time_emb):
        batch_size, f, d = input.shape 
        x = self.activation(self.norm1(input)) # [batch_size, f, d]
        x = self.conv1(x) # [batch_size, f, d]
        if time_emb.ndim ==2:
            time_emb.unsqueeze(1).repeat(1, f, 1)
        x = x + self.time_emb_proj(time_emb) # [batch_size, f, d]
        x = self.activation(self.norm2(x))
        x = self.conv2(self.dropout(x))
        x = x + input
        return x
        
        
class GenmoLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dim_ratio=2,
                 dropout=0.1,
                 activation='relu',
                 modulator='film'):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_ratio = dim_ratio
        self.dropout = dropout
        self.activation = activation
        self.modulator = modulator

        if self.modulator == 'AdaLN':
            self.encoder_AdaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )
            self.decoder_AdaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )
            self.last_AdaLN = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )

        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)

        self.linear1 = nn.Linear(d_model, self.dim_ratio * d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.dim_ratio * d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.initialize_weights()

    def initialize_weights(self):
        if self.modulator == 'AdaLN':
            nn.init.constant_(self.encoder_AdaLN[-1].weight, 0)
            nn.init.constant_(self.encoder_AdaLN[-1].bias, 0)
            nn.init.constant_(self.decoder_AdaLN[-1].weight, 0)
            nn.init.constant_(self.decoder_AdaLN[-1].bias, 0)
            nn.init.constant_(self.last_AdaLN[-1].weight, 0)
            nn.init.constant_(self.last_AdaLN[-1].bias, 0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        tensor = pos.rotate_queries_or_keys(tensor)
        return tensor


    def forward(self, tgt, memory, cond, memory_mask, tgt_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                pos=None, query_pos=None):
        """
        memory_mask_list: list of masks for each batch 
        """
        L, B, D = tgt.shape
        # Text conditioning model
        tgt = self.norm1(tgt)
        if self.modulator == 'AdaLN':
            shift_mlp, scale_mlp, gate_mlp = self.decoder_AdaLN(cond).chunk(3, dim=-1)
            tgt = modulate(tgt, shift_mlp, scale_mlp)

        # Cross-attention with proper residual
        tgt = self.with_pos_embed(tgt, query_pos)
        memory = self.with_pos_embed(memory, pos)
        memory_mask = torch.tile(memory_mask.unsqueeze(1), (1, self.num_head, 1, 1)).reshape(-1, L, L)
        tgt2 = self.cross_attn(tgt, memory, memory,
                            attn_mask=memory_mask)[0]

        if self.modulator == 'AdaLN':
            tgt2 = gate_mlp * tgt2

        tgt = tgt + self.dropout1(tgt2)  # Proper residual

        # Self model
        tgt = self.norm2(tgt)
        tgt = self.with_pos_embed(tgt, pos)
        if self.modulator == 'AdaLN':
            shift_msa, scale_msa, gate_msa = self.encoder_AdaLN(cond).chunk(3, dim=-1)
            tgt = modulate(tgt, shift_msa, scale_msa)

        tgt2 = self.self_attn(tgt, tgt, tgt,
                            attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        if self.modulator == 'AdaLN':
            tgt2 = gate_msa * tgt2
            
        tgt = tgt + self.dropout2(tgt2)  # Proper residual
        tgt_residual = tgt
        
        # FFN with proper residual
        tgt = self.norm3(tgt)
        if self.modulator == 'AdaLN':
            shift_mlp, scale_mlp, gate_mlp = self.last_AdaLN(cond).chunk(3, dim=-1)
            tgt = modulate(tgt, shift_mlp, scale_mlp)  # Use norm3 here
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        if self.modulator == 'AdaLN':
            tgt2 = gate_mlp * tgt2
        tgt = tgt_residual + self.dropout3(tgt2)  # Proper residual
        
        return tgt


class Genmo(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = get_clones(norm, num_layers)
    
    def forward(self,
                tgt,
                memory,
                memory_mask,
                cond=None,
                pos=None):
        
        for i in range(self.num_layers):
            tgt = self.layers[i](tgt,
                                   memory,
                                   cond=cond,
                                   memory_mask=memory_mask,
                                   pos=pos)
            tgt = self.norm[i](tgt)

        return tgt

class LateFusionDecoder(nn.Module):
    def __init__(self, num_layers, norm, cond_dim, latent_dim, num_heads, hidden_dim_ratio, dropout, activation):
        super().__init__()
        self.num_separate_layers = 2 * (num_layers // 3)
        self.num_combined_layers = num_layers - self.num_separate_layers
        self.cond1_layers = get_clones(DeepEncoderLayer(d_model=latent_dim,
                                                      num_head=num_heads,
                                                      dim_ratio=hidden_dim_ratio,
                                                      dropout=dropout,
                                                      activation=activation,
                                                      modulator='AdaLN'), self.num_separate_layers)
        self.cond2_layers = get_clones(DeepEncoderLayer(d_model=latent_dim,
                                                      num_head=num_heads,
                                                      dim_ratio=hidden_dim_ratio,
                                                      dropout=dropout,
                                                      activation=activation,
                                                      modulator='AdaLN'), self.num_separate_layers)
        self.projector = nn.Linear(2 * latent_dim, latent_dim)
        self.cond_proj = nn.Linear(2 * cond_dim, latent_dim)
        self.combined_layers = get_clones(DeepEncoderLayer(d_model=latent_dim,
                                                         num_head=num_heads,
                                                         dim_ratio=hidden_dim_ratio,
                                                         dropout=dropout,
                                                         modulator='AdaLN',
                                                         activation=activation), self.num_combined_layers)
        self.norm = get_clones(norm, self.num_combined_layers)
        self.separate1_norm = get_clones(norm, self.num_separate_layers)
        self.separate2_norm = get_clones(norm, self.num_separate_layers)

    def forward(self, tgt, cond1, cond2, pos=None):
        for i in range(self.num_separate_layers):
            if i == 0:
                tgt_1 = self.cond1_layers[i](tgt, cond1, pos=pos)
                tgt_2 = self.cond2_layers[i](tgt, cond2, pos=pos)
            else:
                tgt_1 = self.cond1_layers[i](tgt_1, cond1, pos=pos)
                tgt_2 = self.cond2_layers[i](tgt_2, cond2, pos=pos)
            tgt_1 = self.separate1_norm[i](tgt_1)
            tgt_2 = self.separate2_norm[i](tgt_2)
        tgt = torch.cat((tgt_1, tgt_2), dim=-1)
        tgt = self.projector(tgt)
        cond = self.cond_proj(torch.cat((cond1, cond2), dim=-1))
        for i in range(self.num_combined_layers):
            tgt = self.combined_layers[i](tgt, cond, pos=pos)
            tgt = self.norm[i](tgt)
        return tgt
