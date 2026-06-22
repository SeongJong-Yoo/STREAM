# This file is taken from signjoey repository
import math

import torch
from torch import Tensor, nn
from .baseline import TransformerEncoder, TransformerEncoderLayer


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Positional Embedding from EDGE: https://github.com/Stanford-TML/EDGE
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def get_timestep_embedding(timesteps: Tensor, 
                           embedding_dim: int, 
                           flip_sin_to_cos: bool=False,
                           downscale_freq_shift: float=1,
                           scale: float=1,
                           max_period: int=10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps must be a 1-D Tensor"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # Scale embeddings
    emb = scale * emb

    # Concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # Flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # Zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, 
                 channel: int, 
                 time_embed_dim: int,
                 act_fn: str='silu'):
        super().__init__()

        self.linear_1 = nn.Linear(channel, time_embed_dim)
        self.act = None
        if act_fn.lower() == 'silu':
            self.act = nn.SiLU()
        elif act_fn.lower() == 'gelu':
            self.act = nn.GELU()
        elif act_fn.lower() == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f'{act_fn} is not supported as an activation function.')
        
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        sample = self.linear_1(x)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, 
                 flip_sin_to_cos: bool,
                 downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: Tensor) -> Tensor:
        time_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift
        )
        return time_emb
            

class EmbedAtt(nn.Module):
    def __init__(self, 
                 config,
                 num_attributes: int,
                 latent_dim: int,
                 guidance_scale: float,
                 guidance_uncond_prob: float,
                 seq_len: int,
                 force_mask: bool=False):
        super().__init__()

        self.config = config
        self.num_attributes = num_attributes
        self.latent_dim = latent_dim
        self.guidance_scale = guidance_scale
        self.guidance_uncond_prob = guidance_uncond_prob

        # self.attribute_embedding = nn.Parameter(
        #     torch.randn(self.num_attributes, self.latent_dim)
        # )
        
        # Attribute condition encoder
        if self.config.data.label_converter == 'one-hot':
            self.attribute_embedding = nn.Sequential(
                nn.Embedding(self.num_attributes, self.config.model.Diffusion.att_embed_dim),
                nn.Linear(self.config.model.Diffusion.att_embed_dim, self.config.model.Diffusion.att_embed_dim),
                nn.SiLU(),
                nn.Linear(self.config.model.Diffusion.att_embed_dim, self.config.model.Diffusion.att_embed_dim),
                nn.LayerNorm(self.config.model.Diffusion.att_embed_dim)
            )
        elif self.config.data.label_converter == 'gemini':
            self.attribute_embedding = nn.Sequential(
                nn.Linear(768, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim)
            )
        self.null_embedding = nn.Parameter(torch.randn(1, seq_len, self.latent_dim))
        att_encoder_layer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            num_head=4,
            dim_feedforward=self.latent_dim * 2,
            dropout=0.1,
            activation='gelu'
        )
        att_encoder_norm = nn.LayerNorm(self.latent_dim)
        self.att_encoder = TransformerEncoder(att_encoder_layer, 3, att_encoder_norm)
        # self.null_embedding = torch.zeros(1, seq_len, self.latent_dim)

        self.force_mask = force_mask
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def mask_cond(self, output, mode, force=False):
        b, t, d = output.shape
        random_mask = self.training
        if mode == 'val':
            random_mask=True

        # Classifier Guidance & classifier-free guidance
        if self.force_mask or force:
            # return torch.zeros_like(output)
            return torch.tile(self.null_embedding, (b, 1, 1))
        elif random_mask and self.guidance_uncond_prob > 0:
            # mask = torch.bernoulli( # 1 -> use null condition, 0 -> use real condition
            #     torch.ones(b, device=output.device) * self.guidance_uncond_prob).repeat(d, t, 1).permute(2, 1, 0)
            mask = (torch.rand(b, device=output.device) < self.guidance_uncond_prob)
            mask = mask.view(b, 1, 1).expand(-1, t, d)
            # return output * (1. - mask)
            # null_embedding = torch.zeros(1, t, d).to(output.device)
            return torch.where(mask==0, output, self.null_embedding)
        else:
            return output
        
    def forward(self, input, mode):
        if self.config.data.label_converter == 'one-hot':
            idx = input.to(torch.long) 
            output = self.attribute_embedding(idx)
        elif self.config.data.label_converter == 'gemini':
            output = self.attribute_embedding(input)

        output = self.att_encoder(output)
        # if mode =='vae':
        #     return output.permute(1, 0, 2)
        
        if not self.training and self.guidance_scale > 1.0:
            if output.shape[0] % 2 == 0:
                uncond, output = output.chunk(2)
                uncond_out = self.mask_cond(uncond, mode, force=True)
                out = self.mask_cond(output, mode)
                output = torch.cat((uncond_out, out))

        output = self.mask_cond(output, mode)

        return output.permute(1, 0, 2)
            
        