"""
Energy Based Cross Attention: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f0878b7efa656b3bbd407c9248d13751-Abstract-Conference.html
Many codes are borrowed from: https://github.com/EnergyAttention/Energy-Based-CrossAttention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseline import DeepEncoderLayer, DeepDecoderLayer, TransformerEncoderLayer, TransformerDecoderLayer, DenseFiLM, featurewise_affine, ResNetBlock, modulate
from .utility import get_clone, get_clones, get_activation_fn
import math
from einops import rearrange


def get_gaussian_kernel1d(kernel_size=5, sigma=1.0):
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1) # (1, 1, kernel_size)
    
class EnergyCrossAttnLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dropout=0.1,
                 rope=None,
                 modulator=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head
        self.dropout = dropout
        self.rope = rope
        self.modulator = modulator

        self.norm = nn.LayerNorm(d_model)
        self.qkv_proj = get_clones(nn.Linear(d_model, d_model), 3)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

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
            self.initialize_weights()

    def with_pos_embed(self, tensor, pos):
        if pos is None:
            return tensor
        tensor = pos.rotate_queries_or_keys(tensor)
        return tensor

    def attend(self, 
               input_state, 
               condition, 
               attention_mask=None,
               pos=None,
               position_index=None):
        batch_size = input_state.shape[0]
        input_state = self.with_pos_embed(input_state, pos)
        condition = self.with_pos_embed(condition, pos)
        query = self.qkv_proj[0](input_state).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        key = self.qkv_proj[1](condition).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2) 
        value = self.qkv_proj[2](condition).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)

        if self.rope is not None:
            query, key = self.rope(query, key, position_index)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attention_mask is not None:
            if attention_mask.ndim != scores.ndim:
                attention_mask = attention_mask.reshape(scores.shape)
            scores = scores.masked_fill(attention_mask==1, -1e9)
        
        probs = self.attn_dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(probs, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        output = self.dropout(self.output_linear(output))

        return output, query, key, scores

    def bayesian_context_update(self, 
                                scores: torch.Tensor, 
                                query: torch.Tensor, 
                                key: torch.Tensor, 
                                gamma_attn: float, 
                                gamma_norm: float):
        batch_size = query.shape[0]
        probs = scores.mT.softmax(dim=-1)
        probs = probs.to(query.dtype).view(-1, probs.shape[-2], probs.shape[-1])

        # Attention: nabla_K E_QK
        query = rearrange(query, 'b h t d -> (b h) t d')
        E_QK = gamma_attn * torch.bmm(probs, query) 
        E_QK = rearrange(E_QK, '(b h) t d -> b t (h d)', h=self.num_head)
        # E_QK = E_QK.view(batch_size, -1, self.num_head, self.d_k)

        # Regularization: nabla_K E(K) 
        k  = rearrange(key, 'b h t d -> (b h) t d')
        weight = torch.diagonal(torch.bmm(k, k.mT), dim1=-2, dim2=-1)
        E_K = -k * weight.softmax(dim=-1).unsqueeze(-1) * gamma_norm
        E_K = rearrange(E_K, '(b h) t d -> b t (h d)', h=self.num_head)

        C = E_QK + E_K
        C = torch.matmul(C, self.qkv_proj[1].weight.detach().to(query.device))

        return C
        
    def initialize_weights(self):
        if self.modulator == 'AdaLN':
            nn.init.constant_(self.encoder_AdaLN[-1].weight, 0)
            nn.init.constant_(self.encoder_AdaLN[-1].bias, 0)
            nn.init.constant_(self.last_AdaLN[-1].weight, 0)
            nn.init.constant_(self.last_AdaLN[-1].bias, 0)

    def composition_and_context_update(self, 
                                      input_state,
                                      condition,
                                      condition_weights,
                                      gamma_attn,
                                      gamma_norm,
                                      attention_mask=None,
                                      pos=None,
                                      position_index=None):
        if isinstance(condition, list):
            num_cond = len(condition)
            if condition_weights is None:
                condition_weights = [1.0 / num_cond] * num_cond

        output_condition_list = []
        output = torch.zeros_like(input_state)
        for i in range(num_cond):
            output_i, query, key, scores = self.attend(input_state, condition[i], attention_mask, pos, position_index)
            cond_i = self.bayesian_context_update(scores, query, key, gamma_attn, gamma_norm)
            
            output_condition_list.append(cond_i)
            output = output + output_i * condition_weights[i]

        return output, output_condition_list


    def forward(self,
                input_state,
                condition,
                gamma_attn: float,
                gamma_norm: float,
                modulator=None,
                condition_weights=None,
                attention_mask=None,
                pos=None,
                position_index=None):
        
        input_state = self.norm(input_state)

        if self.modulator == 'film':
            input_state = featurewise_affine(input_state, self.encoder_film_layer(modulator))
        
        if isinstance(condition, list):
            output, C_main = self.composition_and_context_update(input_state, condition, condition_weights, gamma_attn, gamma_norm, attention_mask, pos, position_index)
            scores = None
        else:
            output, query, key, scores = self.attend(input_state, condition, attention_mask, pos, position_index)
            C_main = self.bayesian_context_update(scores, query, key, gamma_attn, gamma_norm)

        return output, C_main, scores

class STREAM(nn.Module):
    """
    STREAM: Structural-Temporal Regularized Energy-based Attention for Motion

    Implements the update rule:
    Q_new = Q + eta * ( Attention(Q, K_t, V_t) + lambda * StructureAlign(Q, K_m) )

    Where:
    - Attention(Q, K_t, V_t): Standard semantic guidance from text.
    - StructureAlign(Q, K_m): Forces motion self-similarity to match music self-similarity.
    """
    def __init__(self,
                 d_model,
                 num_head,
                 gamma_music=0.1,
                 gamma_music_attn=None,
                 gamma_music_drift=None,
                 downsample_factor=None,
                 dropout=0.1,
                 apply_gaussian_blur=False,
                 music_only=False,
                 proj=False,
                 rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head
        self.dropout = dropout
        self.rope = rope
        self.music_only = music_only
        self.proj = proj
        # Separate gammas: if not provided, fall back to shared gamma_music
        self.gamma_music_attn = gamma_music_attn if gamma_music_attn is not None else gamma_music
        self.gamma_music_drift = gamma_music_drift if gamma_music_drift is not None else gamma_music
        # if self.proj:
            # self.drift_projection = nn.Linear(self.d_k, self.d_k)

        self.gaussian_blur = apply_gaussian_blur
        if self.gaussian_blur:
            kernel = get_gaussian_kernel1d(kernel_size=3, sigma=0.5)
            self.register_buffer('kernel', kernel)

        self.norm = nn.LayerNorm(d_model)
        self.qkv_proj = get_clones(nn.Linear(d_model, d_model), 3)
        self.k_proj = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if downsample_factor is not None:
            self.downsample_factor = downsample_factor
            self.downsample_conv = nn.Conv1d(self.d_k, self.d_k, kernel_size=downsample_factor, stride=downsample_factor)

    def compute_ssm(self, k, visualize=False):
        """
        Compute Self-Similarity Matrix
        k: (B, H, L, D)
        Returns: (B, H, L, L)
        """
        k_norm = F.normalize(k, p=2, dim=-1)
        if self.downsample_factor is not None:
            b, h, l, d = k_norm.shape
            k_norm = rearrange(k_norm, 'b h l d -> (b h) d l')
            k_norm = self.downsample_conv(k_norm)
            k_norm = rearrange(k_norm, '(b h) d l -> b h l d', h=h)
        # S[i,j] = cos_sim(k_i, k_j)
        S = torch.matmul(k_norm, k_norm.transpose(-2, -1))
        if visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(S[2, 0].detach().cpu().numpy(), aspect='auto', cmap='viridis')
            ax.set_title("Jazz, BPM: 75", fontsize=21)
            ax.set_xlabel("Frame", fontsize=21)
            ax.set_ylabel("Frame", fontsize=21)
            ax.tick_params(axis='both', labelsize=18)
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=18)
            plt.tight_layout()
            plt.savefig('images/ssm.png')
            plt.close(fig)

        mask = torch.eye(S.shape[-1]).to(S.device).bool()
        S = S.masked_fill(mask, 0.0)
        return S

    def with_pos_embed(self, tensor, pos):
        if pos is None:
            return tensor
        tensor = pos.rotate_queries_or_keys(tensor)
        return tensor

    def bayesian_context_update(self, 
                                scores: torch.Tensor, 
                                query: torch.Tensor, 
                                key: torch.Tensor, 
                                gamma_attn: float, 
                                gamma_norm: float):
        batch_size = query.shape[0]
        probs = scores.mT.softmax(dim=-1)
        probs = probs.to(query.dtype).view(-1, probs.shape[-2], probs.shape[-1])

        # Attention: nabla_K E_QK
        query = rearrange(query, 'b h t d -> (b h) t d')
        E_QK = gamma_attn * torch.bmm(probs, query) 
        E_QK = rearrange(E_QK, '(b h) t d -> b t (h d)', h=self.num_head)
        # E_QK = E_QK.view(batch_size, -1, self.num_head, self.d_k)

        # Regularization: nabla_K E(K) 
        k  = rearrange(key, 'b h t d -> (b h) t d')
        weight = torch.diagonal(torch.bmm(k, k.mT), dim1=-2, dim2=-1)
        E_K = -k * weight.softmax(dim=-1).unsqueeze(-1) * gamma_norm
        E_K = rearrange(E_K, '(b h) t d -> b t (h d)', h=self.num_head)

        C = E_QK + E_K
        C = torch.matmul(C, self.qkv_proj[1].weight.detach().to(query.device))

        return C


    def compute_music_context_update(self, query, key_m, scores, gamma_attn, gamma_norm):
        """
        Music context update via MAP estimation:
        c_m = c_m - gamma * (nabla_Km E(Q,c_t,c_m) + nabla_Km E(K_m))

        Returns the descent direction (negative gradient), to be added to the music context.

        Args:
            query:   (B, H, T, D)
            key_m:   (B, H, T, D)
            scores:  (B, H, T, T) joint attention scores (pre-softmax logits)
            gamma_attn: float
            gamma_norm: float
        """
        # 1. Correct Probabilistic Normalization
        # Softmax must be applied over the key dimension (last dim) BEFORE transposing.
        probs = scores.softmax(dim=-1).mT
        probs = probs.to(query.dtype).view(-1, probs.shape[-2], probs.shape[-1])

        # Reshape to combine batch and head dimensions: (B*H, T, D)
        q = rearrange(query, 'b h t d -> (b h) t d')
        k_m = rearrange(key_m, 'b h t d -> (b h) t d')

        # --- Block 1: Attention Gradient ---
        # Descent direction: -(-A^T Q) = A^T Q
        E_attn = gamma_attn * torch.bmm(probs, q)

        # --- Block 2: Structure Alignment Regularizer ---
        # Descent direction: + 2 * Lambda^{-1} * (Q(Q^T K_hat_m) - diag(...)K_hat_m)
        # L2 normalization with safe epsilon
        k_norm = torch.linalg.norm(k_m, dim=-1, keepdim=True) + 1e-6
        k_hat = k_m / k_norm

        # O(T d^2) associative matrix multiplication
        # QT_K shape: (BH, D, D)
        QT_K = torch.bmm(q.mT, k_hat)
        # Q_QT_K shape: (BH, T, D)
        Q_QT_K = torch.bmm(q, QT_K)

        # Extract the diagonal components via row-wise dot product
        # alignment shape: (BH, T, 1)
        alignment = (Q_QT_K * k_hat).sum(dim=-1, keepdim=True)

        # Apply the Jacobian scaling (2.0 / ||k||) to map back to unnormalized space
        E_reg = gamma_norm * (2.0 / k_norm) * (Q_QT_K - alignment * k_hat)

        # --- Block 3: Self-Regularization ---
        # Descent direction: -D(softmax(0.5 * diag(K_m K_m^T))) K_m
        # Optimized diagonal extraction: squared L2 norm of rows avoids O(T^2) bmm
        weight = (k_m * k_m).sum(dim=-1) 
        E_K = -gamma_norm * k_m * (0.5 * weight).softmax(dim=-1).unsqueeze(-1)

        # --- Block 4: Final Assembly and Projection ---
        # Combine descent directions
        C = E_attn + E_reg + E_K
        
        # Reshape back to separated batch and head dimensions: (B, T, H*D)
        C = rearrange(C, '(b h) t d -> b t (h d)', h=self.num_head)
        
        # Project back using the linear layer's weights. 
        # F.linear correctly handles the weight transposition (W^T) intrinsically.
        out_device = C.device
        C = F.linear(C, self.k_proj.weight.detach().to(out_device))

        return C

    def apply_gaussian_blur(self, x):
        if self.apply_gaussian_blur:
            x_reshaped = x.view(-1, 1, x.shape[-1])
            x_reshaped = F.conv1d(x_reshaped, self.kernel.to(x.device), padding=1)
            x = x_reshaped.view(x.shape)
        return x

    def attend(self,
               input_state, 
               music,
               attention_mask=None,
               pos=None,
               position_index=None):
        batch_size = input_state.shape[0]
        input_state = self.with_pos_embed(input_state, pos)
        query = self.qkv_proj[0](input_state).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        key = self.qkv_proj[1](input_state).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        value = self.qkv_proj[2](input_state).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)

        if self.rope is not None:
            query, key = self.rope(query, key, position_index)

        key_m = self.k_proj(music).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        scores_music = torch.matmul(query, key_m.transpose(-2, -1)) / math.sqrt(self.d_k)
        S_music = self.compute_ssm(key_m)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        E_drift = torch.matmul(S_music, query)

        if attention_mask is not None:
            # === Decomposed Energy: E = E_text(Q,K_t;M) + λ E_music(Q,K_m) + λ R(Q,c_m) ===
            # With unit_mask, decompose into separate softmaxes so masking text
            # doesn't suppress global music attention.
            if attention_mask.ndim != scores.ndim:
                attention_mask = attention_mask.reshape(scores.shape)
            scores_masked = scores.masked_fill(attention_mask==1, -1e9)

            # Text energy: local per segment (masked)
            probs = F.softmax(scores_masked, dim=-1)
            output = torch.matmul(probs, value)

            # Music energy: global across all frames (unmasked)
            probs_music = F.softmax(scores_music, dim=-1)
            output_music = torch.matmul(probs_music, value)

            output = output + self.gamma_music_attn * output_music + self.gamma_music_drift * E_drift
            # Return masked text scores for Bayesian context update
            scores_for_update = scores_masked
        else:
            # === Joint Energy: E(Q, K_t, K_m) with single softmax ===
            # Without unit_mask, use the original joint formulation where text
            # and music reinforce each other inside a shared softmax.
            scores_joint = scores + self.gamma_music_attn * scores_music
        
            probs = F.softmax(scores_joint, dim=-1)
            output = torch.matmul(probs, value)

            # Music alignment drift
            output = output + self.gamma_music_drift * E_drift
            scores_for_update = scores_joint

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        output = self.dropout(self.output_linear(output))

        return output


    def dual_attend(self,
               input_state,
               condition,
               music,
               attention_mask=None,
               pos=None,
               position_index=None,
               visualize=False):
        batch_size = input_state.shape[0]
        input_state = self.with_pos_embed(input_state, pos)
        condition = self.with_pos_embed(condition, pos)
        query = self.qkv_proj[0](input_state).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        key_c = self.qkv_proj[1](condition).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        value_c = self.qkv_proj[2](condition).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)

        if self.rope is not None:
            query, key_c = self.rope(query, key_c, position_index)

        key_m = self.k_proj(music).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        scores_music = torch.matmul(query, key_m.transpose(-2, -1)) / math.sqrt(self.d_k)
        S_music = self.compute_ssm(key_m, visualize)
        scores_text = torch.matmul(query, key_c.transpose(-2, -1)) / math.sqrt(self.d_k)
        E_drift = torch.matmul(S_music, query)
        if self.gaussian_blur:
            scores_music = self.apply_gaussian_blur(scores_music)
            scores_text = self.apply_gaussian_blur(scores_text)
            E_drift = self.apply_gaussian_blur(E_drift)

        if attention_mask is not None:
            # === Decomposed Energy: E = E_text(Q,K_t;M) + λ E_music(Q,K_m) + λ R(Q,c_m) ===
            # With unit_mask, decompose into separate softmaxes so masking text
            # doesn't suppress global music attention.
            if attention_mask.ndim != scores_text.ndim:
                attention_mask = attention_mask.reshape(scores_text.shape)
            scores_text_masked = scores_text.masked_fill(attention_mask==1, -1e9)

            # Text energy: local per segment (masked)
            probs_text = F.softmax(scores_text_masked, dim=-1)
            output_text = torch.matmul(probs_text, value_c)

            # Music energy: global across all frames (unmasked)
            probs_music = F.softmax(scores_music, dim=-1)
            output_music = torch.matmul(probs_music, value_c)

            output = output_text + self.gamma_music_attn * output_music + self.gamma_music_drift * E_drift
            # Return masked text scores for Bayesian context update
            scores_for_update = scores_text_masked
        else:
            # === Joint Energy: E(Q, K_t, K_m) with single softmax ===
            # Without unit_mask, use the original joint formulation where text
            # and music reinforce each other inside a shared softmax.
            scores_joint = scores_text + self.gamma_music_attn * scores_music
        
            probs = F.softmax(scores_joint, dim=-1)
            output = torch.matmul(probs, value_c)

            # Music alignment drift
            if self.proj:
                # output = output + self.gamma_music_drift * self.drift_projection(E_drift)
                output = output + self.gamma_music_drift * F.normalize(E_drift, p=2, dim=-1)
            else:
                output = output + self.gamma_music_drift * E_drift
            scores_for_update = scores_joint

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        output = self.dropout(self.output_linear(output))

        return output, query, key_c, key_m, scores_for_update, S_music
    
    def forward(self,
                input_state,
                condition,
                gamma_attn: float,
                gamma_norm: float,
                gamma_music_attn=0.0,
                gamma_music_drift=0.0,
                music=None, 
                condition_weights=None,
                attention_mask=None,
                pos=None,
                position_index=None):
        
        input_state = self.norm(input_state)

        # if self.music_only:
            # output = self.attend(input_state, music, attention_mask, pos, position_index)
            # C_main = torch.zeros_like(music)
        # else:
        output, query, key_c, key_m, scores, S_music = self.dual_attend(input_state, condition, music, attention_mask, pos, position_index)
        if self.music_only:
            C_main = torch.zeros_like(music)
        else:
            C_main = self.bayesian_context_update(scores, query, key_c, gamma_attn, gamma_norm)
            # C_music = self.compute_music_context_update(query, key_m, scores, 0.001, 0.001)

        return output, C_main, torch.zeros_like(music)

class EnergyBasedDecoderModule(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 norm,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 rope=None,
                 use_deep_modulator=False,
                 conv=False):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward
        self.activation = activation

        self.norm_self = get_clone(norm)
        self.norm_cross = get_clone(norm)
        self.norm_ff = get_clone(norm)
        self.rope = rope

        self.linear = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = get_activation_fn(activation)

        if use_deep_modulator:
            self.self_layer = DeepEncoderLayer(
                d_model=d_model,
                num_head=num_head,
                dim_ratio=dim_feedforward//d_model,
                dropout=dropout,
                activation=activation,
                rope=rope,
                conv=conv)
        else:
            self.self_layer = TransformerEncoderLayer(
                d_model=d_model,
                num_head=num_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                rope=rope,
                conv=conv
            )

        self.cross_layer = EnergyCrossAttnLayer(
            d_model=d_model,
            num_head=num_head,
            dropout=dropout,
            rope=rope
        )

    def forward(self,
                tgt,
                conditions,
                gamma_attn,
                gamma_norm,
                modulator=None,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                pos=None,
                position_index=None):

        tgt = tgt.permute(1, 0, 2) # [batch_size, num_frames, latent_dim] -> [num_frames, batch_size, latent_dim]

        # 1. Self-attention
        self_out = self.self_layer(tgt,
                            cond=modulator,
                            tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            pos=pos,
                            position_index=position_index)

        tgt = self.dropout1(self_out) + tgt
        tgt = self.norm_self(tgt)

        tgt = tgt.permute(1, 0, 2) # [num_frames, batch_size, latent_dim] -> [batch_size, num_frames, latent_dim]
        # 2. Energy-based cross-attention
        cross_out, update_c, _ = self.cross_layer(tgt,
                                                conditions,
                                                gamma_attn,
                                                gamma_norm,
                                                attention_mask=tgt_mask,
                                                pos=pos,
                                                position_index=position_index)

        tgt = self.dropout2(cross_out) + tgt
        tgt = self.norm_cross(tgt)

        # 4. Update the context
        update_c = conditions + update_c

        # 5. Feed-forward
        output = self.linear2(self.dropout3(self.activation(self.linear(tgt))))

        output = output + tgt
        output = self.norm_ff(output)

        return output, update_c
    
class EnergyBasedStageModule(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 norm,
                 concat,
                 conditioned=False,
                 gamma_music=0.1,
                 gamma_music_attn=None,
                 gamma_music_drift=None,
                 downsample_factor=None,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 rope=None,
                 conv=False,
                 non_energy=False,
                 apply_gaussian_blur=False,
                 AdaLN=True,
                 music_only=False,
                 proj=False):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.norm_cross = get_clone(norm)
        self.norm_ff = get_clone(norm)
        self.conditioned = conditioned
        self.AdaLN = AdaLN
        self.music_only = music_only

        self.activation = get_activation_fn(activation)        
        if AdaLN:
            self.AdaLN_xattn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )
            self.AdaLN_ff = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 3, bias=True)
            )
            self.initialize_weights()
        self.non_energy = non_energy
        self.concat = concat

        if concat:
            self.linear = nn.Linear(d_model * 2, d_model)

        self.encoder_layer = DeepEncoderLayer(
            d_model=d_model,
            num_head=num_head,
            dim_ratio=dim_feedforward//d_model,
            dropout=dropout,
            activation=activation,
            rope=rope,
            modulator='film',
            conv=conv)

        if non_energy:
            self.cross_layer = nn.MultiheadAttention(d_model, num_head, dropout=dropout, batch_first=True)
        else:
            self.cross_layer = STREAM(
                d_model=d_model,
                num_head=num_head,
                gamma_music=gamma_music,
                gamma_music_attn=gamma_music_attn,
                gamma_music_drift=gamma_music_drift,
                downsample_factor=downsample_factor,
                dropout=dropout,
                apply_gaussian_blur=apply_gaussian_blur,
                music_only=music_only,
                proj=proj,
                rope=rope)

        self.proj1 = nn.Linear(d_model, dim_feedforward)
        self.proj2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def initialize_weights(self):
        nn.init.constant_(self.AdaLN_xattn[-1].weight, 0)
        nn.init.constant_(self.AdaLN_xattn[-1].bias, 0)
        nn.init.constant_(self.AdaLN_ff[-1].weight, 0)
        nn.init.constant_(self.AdaLN_ff[-1].bias, 0)

    def forward(self,
                tgt,
                conditions,
                gamma_attn,
                gamma_norm,
                music,
                condition_weights=None,
                modulator=None,
                pos=None,
                stage_mask=None,
                attention_mask=None,
                tgt_key_padding_mask=None,
                position_index=None):
        # if isinstance(conditions, list):
        #     num_cond = len(conditions)
        #     if condition_weights is None:
        #         condition_weights = [1.0 / num_cond] * num_cond

        if self.concat:
            tgt = self.linear(tgt)

        encoder_out = self.encoder_layer(tgt.permute(1, 0, 2),
                                cond = modulator,
                                tgt_mask=attention_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                pos=pos,
                                position_index=position_index)
    
        tgt = encoder_out.permute(1, 0, 2)

        if self.AdaLN:
            shift_xa, scale_xa, gate_xa = self.AdaLN_xattn(conditions).chunk(3, dim=-1)
        residual = tgt
        if self.AdaLN:
            tgt = modulate(tgt, shift_xa, scale_xa)

        cross_out, update_c, update_m = self.cross_layer(tgt,
                                                conditions,
                                                gamma_attn,
                                                gamma_norm,
                                                music=music,
                                                condition_weights=condition_weights,
                                                attention_mask=stage_mask,
                                                pos=pos,
                                                position_index=position_index)
        if self.AdaLN:
            cross_out = gate_xa * cross_out
        tgt = residual + cross_out
        tgt = self.norm_cross(tgt)

        update_c = conditions + update_c
        
        if update_m is not None and music is not None:
            update_m = music + update_m

        if self.AdaLN:
            shift_ff, scale_ff, gate_ff = self.AdaLN_ff(update_c).chunk(3, dim=-1)
        residual = tgt
        if self.AdaLN:
            tgt = modulate(tgt, shift_ff, scale_ff)
        output = self.proj2(self.dropout_ff(self.activation(self.proj1(tgt))))
        if self.AdaLN:
            output = gate_ff * output
        output = output + residual
        output = self.norm_ff(output)

        return output, update_c, update_m

class EnergyBasedStageDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_head,
                 norm,
                 gamma_music=0.1,
                 gamma_music_attn=None,
                 gamma_music_drift=None,
                 stage_cond=False,
                 num_frame=150,
                 num_units=10,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 rope=None,
                 stack=False,
                 conv=False,
                 non_energy=False,
                 downsample_factor=None,
                 apply_gaussian_blur=False,
                 music_only=False,
                 proj=False,
                 AdaLN=True):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.stack = stack

        self.norm_self = norm
        self.stage_cond = stage_cond
        self.rope = rope

        self.num_units = num_units
        self.num_frame = num_frame
        self.unit_mask = None
        self.music_only = music_only

        self.num_blocks = (num_layers - 1) // 2

        if self.num_blocks * 2 != num_layers - 1 :
            raise ValueError("num_layers must be odd")

        _music_kwargs = dict(gamma_music=gamma_music,
                             gamma_music_attn=gamma_music_attn,
                             gamma_music_drift=gamma_music_drift)

        self.self_layers = get_clones(EnergyBasedStageModule(d_model=d_model,
                                                             num_head=num_head,
                                                             norm=norm,
                                                             concat=False,
                                                             conditioned=self.stage_cond,
                                                             dim_feedforward=dim_feedforward,
                                                             dropout=dropout,
                                                             activation=activation,
                                                             rope=rope,
                                                             conv=conv,
                                                             non_energy=non_energy,
                                                             downsample_factor=downsample_factor,
                                                             apply_gaussian_blur=apply_gaussian_blur,
                                                             music_only=music_only,
                                                             AdaLN=AdaLN,
                                                             proj=proj,
                                                             **_music_kwargs), self.num_blocks)
        self.middle_layer = EnergyBasedStageModule(d_model=d_model,
                                                   num_head=num_head,
                                                   norm=norm,
                                                   concat=False,
                                                   conditioned=self.stage_cond,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   rope=rope,
                                                   conv=conv,
                                                   non_energy=non_energy,
                                                   downsample_factor=downsample_factor,
                                                   apply_gaussian_blur=apply_gaussian_blur,
                                                   music_only=music_only,
                                                   AdaLN=AdaLN,
                                                   proj=proj,
                                                   **_music_kwargs)
        self.global_layers = get_clones(EnergyBasedStageModule(d_model=d_model,
                                                               num_head=num_head,
                                                               norm=norm,
                                                               concat=False,
                                                               conditioned=self.stage_cond,
                                                               dim_feedforward=dim_feedforward,
                                                               dropout=dropout,
                                                               activation=activation,
                                                               rope=rope,
                                                               conv=conv,
                                                               non_energy=non_energy,
                                                               downsample_factor=downsample_factor,
                                                               apply_gaussian_blur=apply_gaussian_blur,
                                                               music_only=music_only,
                                                               AdaLN=AdaLN,
                                                               proj=proj,
                                                               **_music_kwargs), self.num_blocks)

    def _construct_unit_mask(self, num_frame, num_units):
        self.unit_mask = torch.ones((num_frame, num_frame), dtype=torch.bool)
        if num_frame > num_units:
            unit_size = num_frame // num_units
            for i in range(num_units):
                start = i * unit_size
                # start_col = i
                self.unit_mask[start:start+unit_size, start:start+unit_size] = False
        elif num_frame < num_units:
            self.unit_mask = torch.ones((num_frame, num_units), dtype=torch.bool)
            unit_size = num_units // num_frame
            for i in range(num_frame):
                start_col = i * unit_size
                start_row = i 
                self.unit_mask[start_row, start_col:start_col+unit_size] = False

    def forward(self,
                tgt,
                conditions,
                gamma_attn,
                gamma_norm,
                music=None,
                pos=None,
                unit_mask=None,
                time_emb=None,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                position_index=None,
                condition_weights=None):
        """
        Input is batch first
        """
        update_c = conditions
        update_m = music

        self_out = []
        if unit_mask is not None:
            if unit_mask.ndim == 3:
                unit_mask = torch.tile(unit_mask.unsqueeze(1), (1, self.num_head, 1, 1)).reshape(-1, unit_mask.shape[1], unit_mask.shape[2])
            self.unit_mask = unit_mask.to(tgt.device)
        for i in range(self.num_blocks):
            tgt, update_c, update_m = self.self_layers[i](tgt=tgt,
                                                conditions=update_c,
                                                modulator=time_emb,
                                                gamma_attn=gamma_attn,
                                                gamma_norm=gamma_norm,
                                                music=update_m,
                                                condition_weights=condition_weights,
                                                stage_mask=self.unit_mask,
                                                attention_mask=tgt_mask,
                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                pos=pos,
                                                position_index=position_index)
            self_out.append(tgt)

        tgt = self.norm_self(tgt)
        tgt, update_c, update_m = self.middle_layer(tgt=tgt,
                                          conditions=update_c,
                                          gamma_attn=gamma_attn,
                                          gamma_norm=gamma_norm,
                                          music=update_m,
                                          condition_weights=condition_weights,
                                          stage_mask=self.unit_mask,
                                          modulator=time_emb,
                                          attention_mask=tgt_mask,
                                          pos=pos,
                                          position_index=position_index)

        for i in range(self.num_blocks):
            if not self.stack:
                tgt = tgt + self_out.pop()
            tgt, update_c, update_m = self.global_layers[i](tgt=tgt,
                                                  conditions=update_c,
                                                  gamma_attn=gamma_attn,
                                                  gamma_norm=gamma_norm,
                                                  music=update_m,
                                                  condition_weights=condition_weights,
                                                  stage_mask=self.unit_mask,
                                                  modulator=time_emb,
                                                  attention_mask=tgt_mask,
                                                  pos=pos,
                                                  position_index=position_index)

        return tgt
