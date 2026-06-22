import torch
import torch.nn as nn
from .baseline import (DeepDecoderLayer, DeepEncoderLayer, TransformerEncoderLayer, TransformerDecoderLayer, 
                       TransformerEncoder, TransformerDecoder, DenseFiLM, LateFusionDecoder,
                       featurewise_affine, ResNetBlock)
from .energy_based_cross_attention import EnergyBasedStageDecoder
from .embeddings import Timesteps, TimestepEmbedding, SinusoidalPosEmb
from .utility import build_position_encoding, get_clones
from einops import rearrange

class CrossEnergyDenoiserJoint(nn.Module):
    def __init__(self, config, 
                 flip_sin_to_cos: bool=True,
                 freq_shift: int=0):
        super().__init__()
        
        self.config = config
        self.latent_dim = config.model.Diffusion.latent_dim
        self.dropout = config.model.dropout
        self.activation = config.model.Diffusion.activation

        self.num_layers = config.model.Diffusion.num_layers
        self.num_enc_layers = config.model.Diffusion.ablation.num_enc_layers
        self.num_heads = config.model.Diffusion.num_heads
        self.hidden_dim_ratio = config.model.Diffusion.hidden_dim_ratio
        self.pe_type = config.model.Diffusion.ablation.pe_type
        self.arch = config.model.Diffusion.ablation.arch
        self.arch_enc = config.model.Diffusion.ablation.arch_enc
        self.cond_combine = config.model.Diffusion.ablation.cond_combine
        self.film = config.model.Diffusion.ablation.film
        self.conformer = config.model.Diffusion.ablation.conformer        
        self.non_energy = config.model.Diffusion.ablation.non_energy
        
        if self.cond_combine == 'concat' or self.arch == 'concat' or self.arch == 'simple':
            self.cond_dim = self.latent_dim // 2
        else:
            self.cond_dim = self.latent_dim

        null_dim = self.cond_dim
        self.stage_cond = config.model.Diffusion.ablation.stage_cond
        # self.frame_motion = config.data.motion_fps * config.data.chunk_time
        self.frame_motion = config.data.motion_fps * config.data.chunk_time
        self.frame_cond = config.data.audio_fps * config.data.chunk_time
        self.body_features = self.config.data.body_features
        self.num_joints = self.config.data.num_joints
        self.output_dim = self.body_features * self.num_joints
        self.input_dim = self.body_features * self.num_joints
        self.data_type = self.config.data.type
        self.absolute = config.data.absolute
        self.frame_indexing = config.model.frame_indexing
        self.neutralization_ratio = 0
        
        self.downsample_factor = self.frame_cond // self.frame_motion
        # Direct output to motion 
        if 'relative' in self.config.data.motion_preprocessing:
            self.output_dim = self.output_dim + 9 # 3 for translation, 6 for rotation
            self.input_dim = self.input_dim + 9 # 3 for translation, 6 for rotation

        if self.config.data.motion_preprocessing in ['absolute', 'face_forward']:
            self.input_dim = self.input_dim + 3 # 3 for root position
            self.output_dim = self.output_dim + 3 # 3 for root position

        if self.absolute:
            self.input_dim = self.body_features
            self.output_dim = self.num_joints * self.body_features

        self.num_attributes = config.data.num_attributes

        self.guidance_scale = config.model.Diffusion.guidance_scale
        self.guidance_uncond_prob = config.model.Diffusion.guidance_uncond_prob
        self.predict_epsilon = config.model.Diffusion.predict_epsilon

        # Structured joint dropout probabilities for multi-condition CFG
        self._cfg_p_uncond = getattr(config.model.Diffusion, 'cfg_p_uncond', 0.10)
        self._cfg_p_text_drop = getattr(config.model.Diffusion, 'cfg_p_text_drop', 0.10)
        self._cfg_p_music_drop = getattr(config.model.Diffusion, 'cfg_p_music_drop', 0.10)
        self.music_only = getattr(config.model.Diffusion.ablation, 'music_only', False)
        apply_gaussian_blur = getattr(config.model.Diffusion.ablation, 'apply_gaussian_blur', False)

        # Ablation
        self.arch = self.config.model.Diffusion.ablation.arch.split('_')[0]
        modulator = self.config.model.Diffusion.ablation.arch.split('_')
        if len(modulator) > 1 and modulator[1] == 'deep':
            self.use_deep_modulator = True
        else:
            self.use_deep_modulator = False
        if self.arch == 'stage' or self.arch_enc == 'stage':
            self.att_avg = config.model.Diffusion.ablation.att_avg

        # Positional Encoder
        self.query_pe = nn.Identity()
        self.pe_encoder = nn.Identity()
        self.cond_pe = nn.Identity()
        self.att_pe = nn.Identity()
        self.desc_pe = nn.Identity()
        self.music_pe = nn.Identity()
        self.rotary_query = None
        self.rotary_pe_encoder = None
        self.rotary_cond_pe = None
        self.rotary_att = None
        self.rotary_desc = None
        self.rotary_music = None
        self.rotary_additional_cond_pe = None
        if self.cond_combine == 'compose':
            self.additional_cond_pe = nn.Identity()
        elif self.cond_combine == 'concat' or self.cond_combine == 'modulate':
            self.cond_pe = nn.Identity()

        self.text_cond_dim = self.cond_dim# // 2 
        self.rope=None
        self.rope_cond=None
        if self.pe_type in ['sine', 'learned']:
            self.query_pe = build_position_encoding(self.latent_dim, position_embedding=self.pe_type)
            self.pe_encoder = build_position_encoding(self.latent_dim, position_embedding=self.pe_type)
            self.cond_pe = build_position_encoding(self.cond_dim, position_embedding=self.pe_type)
            self.att_pe = build_position_encoding(self.text_cond_dim, position_embedding=self.pe_type)
            self.desc_pe = build_position_encoding(self.text_cond_dim, position_embedding=self.pe_type)
            self.music_pe = build_position_encoding(self.cond_dim, position_embedding=self.pe_type)
            if self.cond_combine == 'compose':
                self.additional_cond_pe = build_position_encoding(self.cond_dim, position_embedding=self.pe_type)
            elif self.cond_combine == 'concat':
                self.cond_pe = build_position_encoding(self.latent_dim, position_embedding=self.pe_type)
        elif self.pe_type == 'rotary':
            self.rotary_pe_encoder = build_position_encoding(self.latent_dim, position_embedding=self.pe_type)
            self.rotary_cond_pe = build_position_encoding(self.cond_dim, position_embedding=self.pe_type)
            self.rotary_att = build_position_encoding(self.text_cond_dim, position_embedding=self.pe_type)
            self.rotary_desc = build_position_encoding(self.text_cond_dim, position_embedding=self.pe_type)
            self.rotary_music = build_position_encoding(self.cond_dim, position_embedding=self.pe_type)
            if self.cond_combine == 'compose':
                self.rotary_additional_cond_pe = build_position_encoding(self.cond_dim, position_embedding=self.pe_type)
            elif self.cond_combine == 'concat':
                self.rotary_cond_pe = build_position_encoding(self.latent_dim, position_embedding=self.pe_type)
        else:
            raise ValueError(f"Unknown positional embedding type: {self.pe_type}")
            
        # Music condition
        self.music_cond = True
        if self.music_cond:
            self.music_dim = config.data.audio_features

            # Music condition encoder
            self.music_proj = nn.Sequential(
                nn.Linear(self.music_dim, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.LayerNorm(self.cond_dim)
            )
            if self.cond_combine == 'modulate':
                self.music_film = DenseFiLM(self.latent_dim)
        
        # Time embedding
        self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim)

        if self.film and not 'stage' in self.arch:
            self.film_layers = get_clones(DenseFiLM(self.latent_dim), self.num_layers)
        elif not self.film and not self.arch=='stage' and not self.arch=='separate':
            self.resnet_layers = get_clones(ResNetBlock(self.frame_motion, self.frame_motion, self.latent_dim, groups=30), self.num_layers)

        # Attribute condition encoder
        if self.config.data.label_converter == 'one-hot':
            self.num_attributes = config.data.num_attributes
            self.att_proj = nn.Sequential(
                nn.Embedding(self.num_attributes, self.cond_dim),
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.LayerNorm(self.cond_dim)
            )
        elif self.config.data.label_converter == 'gemini':
            self.num_attributes = 768
            self.att_proj = nn.Sequential(
                nn.Linear(self.num_attributes, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
                nn.LayerNorm(self.cond_dim)
            )
        elif self.config.data.label_converter == 'clip':
            self.num_attributes = 512
            self.att_proj = nn.Sequential(
                nn.Linear(self.num_attributes, self.text_cond_dim),
                nn.SiLU(),
                nn.Linear(self.text_cond_dim, self.text_cond_dim),
                nn.LayerNorm(self.text_cond_dim)
            )
            self.desc_proj = nn.Sequential(
                nn.Linear(self.num_attributes, self.text_cond_dim),
                nn.SiLU(),
                nn.Linear(self.text_cond_dim, self.text_cond_dim),
                nn.LayerNorm(self.text_cond_dim)
            )
        elif self.config.data.label_converter == 't5':
            self.num_attributes = 768
            self.att_proj = nn.Sequential(
                nn.Linear(self.num_attributes, self.text_cond_dim),
                nn.SiLU(),
                nn.Linear(self.text_cond_dim, self.text_cond_dim),
                nn.LayerNorm(self.text_cond_dim)
            )
            self.desc_proj = nn.Sequential(
                nn.Linear(self.num_attributes, self.text_cond_dim),
                nn.SiLU(),
                nn.Linear(self.text_cond_dim, self.text_cond_dim),
                nn.LayerNorm(self.text_cond_dim)
            )
        self.null_description = nn.Parameter(torch.randn(1, self.num_attributes))
        self.null_embedding = nn.Parameter(torch.randn(1, 1, null_dim))
        self.music_null_embedding = nn.Parameter(torch.randn(1, 1, self.cond_dim))
        if self.frame_indexing:
            # Strength of label_index-based attention bias (larger => stronger locality in index space).
            # Assumes label_index is float in [0, 1].
            self.label_index_bias_scale = 5.0
            att_encoder_layer = TransformerDecoderLayer(
                d_model=self.text_cond_dim,
                num_head=self.num_heads,
                dim_feedforward=self.text_cond_dim * self.hidden_dim_ratio,
                dropout=self.dropout,
                activation=self.activation,
            )
            desc_encoder_layer = TransformerDecoderLayer(
                d_model=self.text_cond_dim,
                num_head=self.num_heads,
                dim_feedforward=self.text_cond_dim * self.hidden_dim_ratio,
                dropout=self.dropout,
                activation=self.activation,
            )
            self.frame_index_pe = SinusoidalPosEmb(self.text_cond_dim//2)
            self.frame_index_proj = nn.Sequential(
                nn.Linear(self.text_cond_dim//2, self.text_cond_dim),
                nn.SiLU(),
                nn.Linear(self.text_cond_dim, self.text_cond_dim),
                nn.LayerNorm(self.text_cond_dim)
            )
            desc_encoder_norm = nn.LayerNorm(self.text_cond_dim)
            self.desc_encoder = TransformerDecoder(desc_encoder_layer, 3, desc_encoder_norm)
            att_encoder_norm = nn.LayerNorm(self.text_cond_dim)
            self.att_encoder = TransformerDecoder(att_encoder_layer, 3, att_encoder_norm)

        self.att_linear_proj = nn.Linear(2*self.text_cond_dim, self.text_cond_dim)

        # Motion encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.latent_dim,
            num_head=self.num_heads,
            dim_feedforward=self.latent_dim * self.hidden_dim_ratio,
            dropout=self.dropout,
            activation=self.activation,
            conv=self.conformer
        )
        motion_encoder_norm = nn.LayerNorm(self.latent_dim)
        self.motion_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_enc_layers,
            norm=motion_encoder_norm,
        )

        # Energy-based cross-attention
        eb_norm = nn.LayerNorm(self.latent_dim)

        self.eb_decoders = EnergyBasedStageDecoder(
            num_layers=self.num_layers,
            d_model=self.latent_dim,
            num_head=self.num_heads,
            norm=eb_norm,
            gamma_music=self.config.model.Diffusion.gamma_music,
            gamma_music_attn=getattr(self.config.model.Diffusion, 'gamma_music_attn', None),
            gamma_music_drift=getattr(self.config.model.Diffusion, 'gamma_music_drift', None),
            num_frame=self.frame_motion,
            num_units=self.frame_cond,
            dim_feedforward=self.latent_dim * self.hidden_dim_ratio,
            dropout=self.dropout,
            activation=self.activation,
            stage_cond=self.stage_cond,
            conv=self.conformer,
            non_energy=self.non_energy,
            downsample_factor=self.downsample_factor,
            apply_gaussian_blur=apply_gaussian_blur,
            music_only=self.music_only,
            proj=self.config.model.Diffusion.ablation.proj,
            AdaLN=getattr(self.config.model.Diffusion.ablation, 'AdaLN', True)
        )

        # Skeleton embedding
        if self.config.model.Diffusion.ablation.kernel_size is not None and self.absolute:
            self.kernel_size = self.config.model.Diffusion.ablation.kernel_size
            self.stride_size = self.config.model.Diffusion.ablation.stride_size
            self.skeleton_embedding = nn.Conv2d(self.input_dim, self.latent_dim, kernel_size=self.kernel_size, stride=self.stride_size)
            self.projection = nn.Linear((self.num_joints // self.kernel_size[1]) * self.latent_dim, self.latent_dim)
            self.use_2D_conv = True
        else:
            self.skeleton_embedding = nn.Linear(self.input_dim, self.latent_dim)
            self.use_2D_conv = False
        self.final_film = DenseFiLM(self.latent_dim)
        self.final_norm = nn.LayerNorm(self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, self.output_dim)

        self._reset_parameters()


    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def mask_cond(self, output, force=False, skip=False, mode='train'):
        b, t, d = output.shape
        random_mask = self.training
        if mode == 'val':
            random_mask=True

        if force:
            return self.null_embedding.repeat(b, 1, 1).to(output.device)
        elif skip:
            return output
        elif random_mask and self.guidance_uncond_prob > 0:
            mask = torch.bernoulli(torch.ones(b, device=output.device) * self.guidance_uncond_prob).repeat(d, t, 1).permute(2, 1, 0)
            return torch.where(mask.to(bool), self.null_embedding.to(output.device), output)
        elif not random_mask and self.guidance_uncond_prob > 0:
            if output.shape[0] % 2 == 0:
                batch_size = output.shape[0] // 2
                output[:batch_size] = self.null_embedding.repeat(batch_size, 1, 1).to(output.device)
            else:
                print(f"Warning: output.shape[0] % 2 != 0, {output.shape}")
            return output
        else:
            return output

    def mask_cond_music(self, output, force=False, skip=False, mode='train'):
        b, t, d = output.shape
        random_mask = self.training
        if mode == 'val':
            random_mask=True

        if force:
            return self.music_null_embedding.repeat(b, 1, 1).to(output.device)
        elif skip:
            return output
        elif random_mask and self.guidance_uncond_prob > 0:
            mask = torch.bernoulli(torch.ones(b, device=output.device) * self.guidance_uncond_prob).repeat(d, t, 1).permute(2, 1, 0)
            return torch.where(mask.to(bool), self.music_null_embedding.to(output.device), output)
        elif not random_mask and self.guidance_uncond_prob > 0:
            if output.shape[0] % 2 == 0:
                batch_size = output.shape[0] // 2
                output[:batch_size] = self.music_null_embedding.repeat(batch_size, 1, 1).to(output.device)
            else:
                print(f"Warning: output.shape[0] % 2 != 0, {output.shape}")
            return output
        else:
            return output


    def mask_cond_joint(self, text_cond, music_cond, mode='train'):
        """
        Structured joint dropout for multi-condition CFG.
        Instead of independent Bernoulli draws, uses a single draw per sample
        to explicitly control all four conditioning states:
          - Both present  (~70%)
          - Text only      (~10%)  -> music null
          - Music only     (~10%)  -> text null
          - Fully uncond   (~10%)  -> both null
        """
        b, t, d = text_cond.shape
        device = text_cond.device
        random_mask = self.training
        if mode == 'val':
            random_mask = True

        p_uncond = getattr(self, '_cfg_p_uncond', 0.10)
        p_text_drop = getattr(self, '_cfg_p_text_drop', 0.10)
        p_music_drop = getattr(self, '_cfg_p_music_drop', 0.10)

        if random_mask and (p_uncond + p_text_drop + p_music_drop) > 0:
            # print("Using structured random dropout for multi-condition CFG")
            # Training / validation: structured random dropout
            u = torch.rand(b, device=device)
            # u < p_uncond:                                        both null
            # p_uncond <= u < p_uncond + p_text_drop:              text null (music only)
            # p_uncond + p_text_drop <= u < p_uncond + p_text_drop + p_music_drop: music null (text only)
            # u >= p_uncond + p_text_drop + p_music_drop:          both present
            drop_text = u < (p_uncond + p_text_drop)
            drop_music = (u < p_uncond) | ((u >= p_uncond + p_text_drop) & (u < p_uncond + p_text_drop + p_music_drop))

            text_mask = drop_text[:, None, None]  # [B, 1, 1]
            music_mask = drop_music[:, None, None]

            text_cond = torch.where(text_mask, self.null_embedding.to(device), text_cond)
            music_cond = torch.where(music_mask, self.music_null_embedding.to(device), music_cond)

        elif not random_mask and b % 3 == 0:
            # print("Using 3-pass multi-condition CFG")
            # Inference: 3-pass multi-condition CFG
            # Batch is structured as [uncond, text_only, full] (each of size B/3)
            # Replace projected-zero portions with learned null embeddings
            third = b // 3
            # Pass 1 (uncond): both text and music null
            text_cond[:third] = self.null_embedding.repeat(third, t, 1).to(device)
            # Pass 2 (text_only): music null, text stays real
            music_cond[:third] = self.music_null_embedding.repeat(third, t, 1).to(device)
            music_cond[third:2*third] = self.music_null_embedding.repeat(third, t, 1).to(device)
            # Pass 3 (full): both real — no changes

        return text_cond, music_cond
        
    def motion_preprocessing(self, motion):
        if self.use_2D_conv:
            motion = rearrange(motion, 'b f (j d) -> b d f j', d=self.body_features)
            # motion = motion.permute(0, 3, 1, 2)
            x = self.skeleton_embedding(motion)
            x = rearrange(x, 'b d f j -> b f (j d)')
            x = self.projection(x)
        else:
            x = self.skeleton_embedding(motion)
        
        x = x.permute(1, 0, 2)  # [num_frames, batch_size, latent_dim]
        x = self.pe_encoder(x)
        x = self.motion_encoder(x, pos=self.rotary_pe_encoder)

        # if self.arch == 'stage':
        #     x = self.motion_projection(x.permute(1, 2, 0)) # [batch_size, latent_dim, num_units]
        #     x = x.permute(0, 2, 1) # [batch_size, num_units, latent_dim]

        return x.permute(1, 0, 2) # [batch_size, num_frames, latent_dim]

    def get_motion_embedding(self, motion, mask=None):
        """
        Extract motion embedding for contrastive loss.
        Args:
            motion: [B, T, D] input motion (normalized)
            mask: [B, T] optional mask for valid frames
        Returns:
            motion_embed: [B, latent_dim] pooled motion embedding
        """
        encoded = self.motion_preprocessing(motion)  # [B, T, latent_dim]

        # Mean pool over time dimension with optional mask
        if mask is not None:
            mask_float = mask.float()
            motion_embed = (encoded * mask_float.unsqueeze(-1)).sum(1) / mask_float.sum(1).clamp(min=1).unsqueeze(-1)
        else:
            motion_embed = encoded.mean(dim=1)  # [B, latent_dim]

        return motion_embed

    def pose_regression(self, x):
        # x = featurewise_affine(x, self.final_film(modulator))
        x = self.final_norm(x)
        # if self.conformer:
        # output = self.final_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        # else:
        output = self.final_layer(x)
        return output


    def slice_attributes_by_continuous_values_vectorized(self, attributes, tolerance=1e-6):
        """
        Vectorized version to slice attributes tensor along T axis based on continuous D values.
        Each batch is processed separately since they contain different values.
        
        Args:
            attributes: Tensor of shape (B, T, D) where values are continuous along T for same text
            tolerance: Tolerance for comparing floating point values
            
        Returns:
            List of lists, where each inner list contains tuples (start_idx, end_idx) for each batch
        """
        B, T, D = attributes.shape
        
        if T <= 1:
            return [[(0, T)] for _ in range(B)]
        
        batch_slices = []
        
        # Process each batch separately
        for batch_idx in range(B):
            batch_data = attributes[batch_idx]  # Shape: (T, D)
            
            # Calculate differences between consecutive timesteps
            diffs = torch.abs(batch_data[1:] - batch_data[:-1])  # Shape: (T-1, D)
            
            # Check if any dimension changed beyond tolerance
            changes = torch.any(diffs > tolerance, dim=1)  # Shape: (T-1,)
            
            # Get indices where changes occur (add 1 because diffs is offset by 1)
            change_indices = torch.where(changes)[0] + 1
            change_indices = change_indices.tolist()
            
            # Create slice boundaries for this batch
            slices = []
            start_idx = 0
            
            for change_idx in change_indices:
                slices.append((start_idx, change_idx))
                start_idx = change_idx
                
            # Add the final slice
            slices.append((start_idx, T))
            
            batch_slices.append(slices)
        
        return batch_slices

    def generate_att_masks(self, attributes):
        """
        Generate attention masks for all batches.
        
        Args:
            attributes: Tensor of shape (B, L, D)
        
        Returns:
            Tensor of shape (B, L, L) containing attention masks for all batches
        """
        B, L = attributes.shape[0], attributes.shape[1]
        
        # Get slice boundaries for all batches
        all_batch_slices = self.slice_attributes_by_continuous_values_vectorized(attributes)
        
        # Initialize masks for all batches
        mask = torch.ones((B, L, L), dtype=torch.bool, device=attributes.device)
        
        # Generate mask for each batch
        for batch_idx, slices in enumerate(all_batch_slices):
            for start_idx, end_idx in slices:
                mask[batch_idx, start_idx:end_idx, start_idx:end_idx] = False
                
        return mask

    def generate_att_masks_optimized(self, attributes, tolerance=1e-6):
        """
        Vectorized generation of block attention masks based on continuous attribute values.

        Args:
            attributes: Tensor of shape (B, L, D) - Batch, Length, Dimension.
            tolerance: Tolerance for floating point comparison.
            
        Returns:
            Tensor of shape (B, L, L) containing attention masks (False=attended, True=masked).
        """
        B, L, D = attributes.shape

        if L <= 1:
            # If length is 1 or less, no masking needed (mask=False means attend)
            return torch.zeros((B, L, L), dtype=torch.bool, device=attributes.device)

        # --- Step 1: Vectorized Change Detection (across all B, L, D) ---

        # 1. Calculate differences between consecutive timesteps (L-1, D)
        # diffs_sq: (B, L-1, D) - Use squared difference for DDP stability if needed
        diffs = torch.abs(attributes[:, 1:] - attributes[:, :-1])

        # 2. Check if any dimension changed beyond tolerance (changes: B, L-1)
        # The 'changes' tensor is True where the attribute changes.
        changes = torch.any(diffs > tolerance, dim=2) 

        # 3. Create a change indicator tensor (B, L) with changes marked at the end of a block
        # indicator_t: True at t=L-1 means a change happened between L-2 and L-1.
        # indicator: (B, L). Set boundary (t=0) to True to mark start of first block.
        indicator = torch.cat([
            torch.ones((B, 1), dtype=torch.bool, device=attributes.device), # Start of every sequence
            changes
        ], dim=1)

        # --- Step 2: Calculate Block ID for Every Timestep ---

        # The cumulative sum of 'indicator' assigns a unique block ID to each attribute block.
        # block_id: (B, L). Example: [1, 1, 2, 2, 2, 3, 3]
        block_id = torch.cumsum(indicator.int(), dim=1)

        # --- Step 3: Vectorized Mask Generation ---

        # 1. Expand block_id to B, L, L
        # block_id_row: (B, L, 1)
        # block_id_col: (B, 1, L)
        block_id_row = block_id.unsqueeze(2)
        block_id_col = block_id.unsqueeze(1)

        # 2. The mask is True (masked/do not attend) where block_id_row != block_id_col.
        # This creates the necessary block diagonal structure:
        # Mask = (B, L, L)
        # Diagonal blocks are False (attend)
        # Off-diagonal blocks are True (mask)
        mask = (block_id_row != block_id_col)

        return mask

    def att_embedding(self, att, label_index=None):
        if self.frame_indexing:
            att = self.att_proj(att).permute(1, 0, 2) # [num_frames, batch_size, latent_dim]
            att = self.att_pe(att)

            T, B, _ = att.shape
            if label_index is None:
                # Default to a normalized index within the window: [0, 1]
                label_index = torch.linspace(0, 1, T, device=att.device, dtype=att.dtype).unsqueeze(0).repeat(B, 1) * 1000
            else:
                label_index = label_index.to(device=att.device, dtype=att.dtype) * 1000
                if label_index.dim() == 1:
                    label_index = label_index.unsqueeze(0).repeat(B, 1)

            # (B, T, 1) -> (T, B, D) for decoder memory
            frame_index_emb_btd = self.frame_index_proj(self.frame_index_pe(label_index.unsqueeze(-1))).squeeze(1)
            frame_index_emb = frame_index_emb_btd.permute(1, 0, 2)

            # 1) Inject into tokens (stronger than memory-only).
            att = att + frame_index_emb

            # 2) Also bias attention scores by distance in index space.
            # bias: (B, T, T), additive mask (more negative => less attention)
            dist = torch.abs(label_index.unsqueeze(2) - label_index.unsqueeze(1))
            bias = (-self.label_index_bias_scale * dist).to(dtype=att.dtype)
            # self-attn expects (B * num_heads, T, T)
            tgt_mask = bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape(-1, T, T)

            att = self.att_encoder(att, memory=frame_index_emb, tgt_mask=tgt_mask).permute(1, 0, 2) # [batch_size, num_frames, latent_dim]
        else:
            att = self.att_proj(att)
        return att

    def desc_embedding(self, description, label_index=None):
        desc_mask = torch.norm(description, dim=-1) < 1e-5 #(description.mean(dim=-1)).mean(dim=-1) < 1e-5  # [batch_size, num_frames]
        
        # Replace zero descriptions with null_description using broadcasting
        description = torch.where(desc_mask.unsqueeze(-1), self.null_description.to(description.device), description)
        if self.frame_indexing:
            desc = self.desc_proj(description).permute(1, 0, 2) # [num_frames, batch_size, latent_dim]
            desc = self.desc_pe(desc)

            T, B, _ = desc.shape
            if label_index is None:
                # Default to a normalized index within the window: [0, 1]
                label_index = torch.linspace(0, 1, T, device=desc.device, dtype=desc.dtype).unsqueeze(0).repeat(B, 1) * 1000
            else:
                label_index = label_index.to(device=desc.device, dtype=desc.dtype) * 1000
                if label_index.dim() == 1:
                    label_index = label_index.unsqueeze(0).repeat(B, 1)

            # (B, T, 1) -> (T, B, D) for decoder memory
            frame_index_emb_btd = self.frame_index_proj(self.frame_index_pe(label_index.unsqueeze(-1))).squeeze(1)
            frame_index_emb = frame_index_emb_btd.permute(1, 0, 2)

            # 1) Inject into tokens (stronger than memory-only).
            desc = desc + frame_index_emb

            # 2) Bias attention scores by distance in index space.
            dist = torch.abs(label_index.unsqueeze(2) - label_index.unsqueeze(1))
            bias = (-self.label_index_bias_scale * dist).to(dtype=desc.dtype)
            tgt_mask = bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape(-1, T, T)

            desc = self.desc_encoder(desc, memory=frame_index_emb, tgt_mask=tgt_mask).permute(1, 0, 2) # [batch_size, num_frames, latent_dim]
        else:
            desc = self.desc_proj(description)

        return desc

    def forward(self,
                sample,
                timestep,
                attributes,
                description,
                music,
                audio_mask,
                att_mask, 
                label_index=None,
                gamma_attn=1, 
                gamma_norm=1,
                cond_mask_skip=False,
                neutralization_ratio=0,
                mode='train',
                mask=None,
                **kwargs):
        """
        sample: [batch_size, num_frames, latent_dim]
        timestep: [batch_size]
        attributes: [batch_size, num_frames]
        music: [batch_size, num_frames, audio_features]
        mask: [batch_size, num_frames], False = padded frame, True = valid (optional)
        """
        self.neutralization_ratio = neutralization_ratio
        unit_mask = None
        # if self.arch=='stageV2' or self.arch=='stack' or self.arch=='separate':
            # unit_mask = self.generate_att_masks_optimized(description)#.permute(1, 0, 2)
        sample = self.motion_preprocessing(sample) # [batch_size, num_frames, latent_dim] or [batch_size, num_units, latent_dim]

        # Prepare padded mask: True = valid, False = padded -> tgt_key_padding_mask True = ignore (padded)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, device=sample.device, dtype=torch.bool)
            else:
                mask = mask.to(device=sample.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(sample.shape[0], -1)
            tgt_key_padding_mask = ~mask  # PyTorch: True = ignore position
        else:
            tgt_key_padding_mask = None

        # 1. Time embedding
        timesteps = timestep.expand(sample.shape[0]).clone()
        time_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)  # [1, batch_size, latent_dim]

        if not self.music_only:
            # 2-1. Attribute embedding
            att  = self.att_embedding(attributes, label_index)
            desc = self.desc_embedding(description, label_index)

            att_masked = self.att_linear_proj(torch.cat([att, desc], dim=-1))
        else:
            att_masked = None
            

        # music_null_mask = torch.mean(torch.norm(music, dim=-1), dim=-1) < 1e-5
        music = self.music_proj(music)   
        music_masked = torch.where(audio_mask.unsqueeze(-1).unsqueeze(-1), music, self.music_null_embedding.repeat(1, music.shape[1], 1).to(music.device))
        if not self.music_only:
            att_masked = torch.where(att_mask.unsqueeze(-1).unsqueeze(-1), att_masked, self.null_embedding.repeat(1, att_masked.shape[1], 1).to(description.device))

        if self.music_only:
            music_masked = self.mask_cond_music(music_masked, mode=mode)
            music_masked = self.music_pe(music_masked)
            cond = music_masked
        else:
            att_masked, music_masked = self.mask_cond_joint(att_masked, music_masked, mode=mode)
            cond = self.cond_pe(att_masked)
            music_masked = self.music_pe(music_masked)
            
        time_emb = torch.tile(time_emb, (music_masked.shape[1], 1, 1))
        if not self.film:
            time_emb = time_emb.permute(1, 0, 2)

        # 3. Decoder
        sample = self.query_pe(sample)
        sample = self.eb_decoders(tgt=sample, 
                                    conditions=cond, 
                                    music=music_masked,
                                    gamma_attn=gamma_attn, 
                                    gamma_norm=gamma_norm, 
                                    time_emb=time_emb, 
                                    pos=self.rotary_query,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    unit_mask=unit_mask)

        sample = self.pose_regression(sample)
        result = {'pred': sample}

        result['att'] = att_masked
        result['music'] = music

        return result

        

        