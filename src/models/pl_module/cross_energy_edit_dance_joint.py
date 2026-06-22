import pytorch_lightning as pl
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection
from einops import rearrange
import inspect
import time
import os
from pathlib import Path
import numpy as np
from typing import Optional, Union
import math
import copy
from pytorch_lightning.utilities import rank_zero_only

from src.utils.utility import get_model, get_obj_from_str, compute_foot_contact
from src.losses.metrics import MPJPE, BeatAlignment, MPJVE
from src.losses.losses import LossesCollection
from src.skeleton.preprocessing import motion_postprocessing

from src.skeleton.utility import get_motorica_skeleton_names
from src.skeleton.forward_kinematics import ForwardKinematics
from src.skeleton.smpl_fk import SMPLModel
from src.dataloader.dataset_loader import get_datasets
from src.visualization.utility import choose_majority, compute_tsne
from src.models.solver.integrators import ode

class CrossEnergyEditDanceJoint(pl.LightningModule):
    def __init__(self, config, data_info, mode='train', id=None):
        super().__init__()
        self.save_hyperparameters(config, ignore='data_info')

        self.use_ode_solver=False
        self.config = config
        self.data_type = config.data.type
        self.num_frames = config.data.motion_fps * config.data.chunk_time
        self.latent_dim = config.model.Diffusion.latent_dim
        self.num_joints = config.data.num_joints
        self.body_features = config.data.body_features
        self.input_dim = self.body_features * self.num_joints
        self.data_rep = config.data.representation
        self.gamma_attn = config.model.Diffusion.gamma_attn
        self.gamma_norm = config.model.Diffusion.gamma_norm
        self.use_ema = config.model.use_ema
        self.ema_decay = config.model.ema_decay
        if self.use_ema:
            self.model_ema = None

        self.absolute = config.data.absolute
        self.music_only = getattr(config.model.Diffusion.ablation, 'music_only', False)
        if self.music_only:
            self.cfg_factor = 2
        else:
            self.cfg_factor = 3

        if 'relative' in self.config.data.motion_preprocessing:
            self.input_dim = self.input_dim + 9 # 3 for translation, 6 for rotation
        if self.config.data.motion_preprocessing in ['absolute', 'face_forward']:
            self.input_dim = self.input_dim + 3 # 3 for root position
        if self.absolute:
            self.input_dim = self.num_joints * 3
        else:
            if self.data_type == 'tree':
                if self.data_rep == 'motorica':
                    self.fk = ForwardKinematics(selected_joints=get_motorica_skeleton_names())
                elif self.data_rep == 'smpl':
                    self.fk = SMPLModel(num_joints=self.num_joints, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset
        self.id = id
        self.normalizer=None
        if self.config.data.normalize_data:
            self.normalizer = data_info['normalizer']
        print("Dataset module {} initialized".format(config.data.name))
        self.skeleton_dict = data_info['skeleton_dict']

        # Set Model
        self.model_name = config.model.name # vae or diffusion or energy_diffusion
        if 'flow' in self.model_name:
            self.model_type = 'flow'
        else:
            self.model_type = 'diffusion'

        self.model = get_model(config, self.model_name)

        if mode == 'inference':
            self.load_pretrained_model(config.model.pretrained_energy, trainable=False)

        self.guidance_scale = config.model.Diffusion.guidance_scale
        self.guidance_scale_text = getattr(config.model.Diffusion, 'guidance_scale_text', self.guidance_scale)
        self.guidance_scale_music = getattr(config.model.Diffusion, 'guidance_scale_music', self.guidance_scale)
        self.do_classifier_free_guidance = (self.guidance_scale_text > 1.0) or (self.guidance_scale_music > 1.0)
        self.predict_epsilon = config.model.Diffusion.predict_epsilon

        # Set scheduler
        if self.model_type == 'diffusion':
            scheduler_type = "diffusers." + config.model.Diffusion.scheduler.type
            noise_scheduler_type = "diffusers." + config.model.Diffusion.noise_scheduler.type
            if not self.predict_epsilon:
                config.model.Diffusion.scheduler.params['prediction_type'] = 'sample'
                config.model.Diffusion.noise_scheduler.params['prediction_type'] = 'sample'
        elif self.model_type == 'flow':
            scheduler_type = "src.models.architecture.scheduler." + config.model.Diffusion.scheduler.type
            noise_scheduler_type = "src.models.architecture.scheduler." + config.model.Diffusion.noise_scheduler.type
        self.scheduler = get_obj_from_str(scheduler_type)(**config.model.Diffusion.scheduler.params)
        self.noise_scheduler = get_obj_from_str(noise_scheduler_type)(**config.model.Diffusion.noise_scheduler.params)

        # Set Losses
        self._losses = {
            mode: LossesCollection(config) 
            for mode in ["losses_train", "losses_val", "losses_test"]
        }

        self.losses = {
            key: self._losses["losses_" + key] for key in ["train", "test", "val"]
        }

        # Set Metrics
        self.metrics_dict = config.train.metrics
        self.configure_metrics()

    def compute_density_for_timestep_sampling(
        self,
        weighting_scheme: str,
        batch_size: int,
        logit_mean: float = None,
        logit_std: float = None,
        mode_scale: float = None,
        device: Union[torch.device, str] = "cpu",
        generator: Optional[torch.Generator] = None,
    ):
        """
        Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://huggingface.co/papers/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device=device, generator=generator)
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device=device, generator=generator)
        return u


    def temperature(
        self,
        t: torch.Tensor,
        tau_star: float,
        epsilon_max: float
    ) -> torch.Tensor:
        """
        From Energy-Matching paper: 
        Piecewise definition of eps(t):http://arxiv.org/abs/2504.10612
        - eps(t) = 0,                    for t < tau_star
        - eps(t) = linear ramp up to epsilon_max,  for tau_star <= t < 1
        - eps(t) = epsilon_max,          for t >= 1
        """
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(-1)
        eps = torch.zeros_like(t)

        # region where we ramp from 0 up to epsilon_max
        mask_mid = (t >= tau_star) & (t < 1.0)
        scale = 1.0 - tau_star  # length of the ramp
        eps[mask_mid] = epsilon_max * (t[mask_mid] - tau_star) / scale

        # region where we remain at epsilon_max
        eps[t >= 1.0] = epsilon_max

        return eps

    def select_model_and_forward(self, x, t, att, music, description, label_index, mode, mask=None, audio_mask=None, att_mask=None):
        if mode == 'inference':
            inference_model = self.model_ema if (self.use_ema and self.model_ema is not None and not self.training) else self.model
        else:
            inference_model = self.model
        input = {"timestep": t,
                 "attributes": att,
                 "description": description,
                 "music": music,
                 "mode": mode}
        input['sample'] = x
        input['label_index'] = label_index
        input['gamma_attn'] = self.gamma_attn
        input['gamma_norm'] = self.gamma_norm
        input['mask'] = mask
        input['audio_mask'] = audio_mask
        input['att_mask'] = att_mask
        model_output = inference_model(**input)
        noise_pred = model_output['pred']

        return noise_pred, model_output


    def flow_weight(self, t, cutoff=0.8):
        """
        Flow weighting function:
        - w_flow = 1 for t < cutoff
        - linearly from 1 down to 0 as t goes from cutoff..1
        - 0 for t >= 1
        """
        w = torch.ones_like(t)
        decay_region = (t >= cutoff) & (t < 1.0)
        w[decay_region] = 1.0 - (t[decay_region] - cutoff) / (1.0 - cutoff)
        w[t >= 1.0] = 0.0
        return w

    def _diffusion_forward(self, sample, music, attributes, description, label_index, mode, mask=None, audio_mask=None, att_mask=None):
        # 1. Sample noise that we will add to the latent
        noise = torch.randn_like(sample) # [batch_size, num_frames, latent_dim]
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            noise = noise * mask_expanded
        # 2. Sample random timesteps
        timestep_sampling = getattr(self.config.model.Diffusion, 'timestep_sampling', 'logit_normal')
        u = self.compute_density_for_timestep_sampling(
            weighting_scheme=timestep_sampling,
            batch_size=sample.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            device=sample.device,
        )
        indices = (u * self.noise_scheduler.config.get("num_train_timesteps")).long().to(sample.device)
        timesteps = self.noise_scheduler.timesteps.to(sample.device)[indices]

        # 3. Add noise to the sample according to the noise magnitude at each timestep
        noisy_sample = self.noise_scheduler.add_noise(sample.clone(), noise, timesteps)

        # 4. Predict the noise
        model_output, model_output_dict = self.select_model_and_forward(x=noisy_sample, 
                                                     t=timesteps, 
                                                     att=attributes, 
                                                     music=music, 
                                                     description=description, 
                                                     label_index=label_index, 
                                                     mode=mode,
                                                     mask=mask,
                                                     audio_mask=audio_mask,
                                                     att_mask=att_mask)

        output = {}
        output['model_output'] = model_output

        # Timestep-aware auxiliary loss weighting: down-weight aux losses at high noise
        use_aux_weight = getattr(self.config.loss, 'use_aux_weight', False)
        if use_aux_weight:
            max_t = self.noise_scheduler.config.num_train_timesteps
            t_ratio = timesteps.float() / max_t
            schedule = getattr(self.config.loss, 'aux_weight_schedule', 'cosine')
            if schedule == 'cosine':
                output['aux_weight'] = torch.cos(t_ratio * math.pi / 2).clamp(min=0)
            elif schedule == 'linear':
                output['aux_weight'] = (1.0 - t_ratio).clamp(min=0)
            else:
                output['aux_weight'] = torch.ones_like(t_ratio)
        if self.predict_epsilon:
            output['target'] = noise
            output['pred_motion'] = torch.stack([self.noise_scheduler.step(model_output[i], timesteps[i], noisy_sample[i])['pred_original_sample'] for i in range(noise.shape[0])])
        else:
            output['target'] = sample   # Sample is normalized
            output['pred_motion'] = model_output # For sample estimation case, the sample is the same as the predicted motion

        if 'att' in model_output_dict:
            output['att'] = model_output_dict['att']
        if 'music' in model_output_dict:
            output['music'] = model_output_dict['music']

        if mask is not None:
            output['pred_motion'] = output['pred_motion'] * mask_expanded

        return output

    def _prepare_prefix_blending(self, prefix_motion, prefix_frames, blend_frames, batch_size, num_frame, device):
        """
        Build masks and tensors for prefix-guided long-range generation.

        Returns a dict with:
            blend_mask:             [1, T, 1]  spatial blend weight (1=keep prefix, 0=generate)
            text_cfg_mask:          [1, T, 1]  temporal text-CFG weight (ramps 0→scale over overlap)
            prefix_padded:          [B, T, D]  clean prefix zero-padded to full window
            prefix_noise_persistent:[B, T, D]  fixed noise reused at every denoising step
        Returns None when prefix_motion is None.
        """
        if prefix_motion is None:
            return None

        T = num_frame

        # --- Blend mask M (applied to rotation dims only) ---
        # 0..hard_end        → 1.0   (100 % known past)
        # hard_end..prefix_frames → linear 1→0   (blend zone)
        # prefix_frames..T   → 0.0   (100 % new generation)
        blend_mask = torch.zeros(1, T, 1, device=device)
        hard_end = max(0, prefix_frames - blend_frames)
        blend_mask[:, :hard_end, :] = 1.0
        if blend_frames > 0 and prefix_frames > hard_end:
            actual_blend = prefix_frames - hard_end
            ramp = torch.linspace(1.0, 0.0, actual_blend, device=device)
            blend_mask[:, hard_end:prefix_frames, :] = ramp.unsqueeze(0).unsqueeze(-1)

        # --- Text-CFG temporal weight ---
        # 0..prefix_frames → ramp 0→guidance_scale_text
        # prefix_frames..T → guidance_scale_text
        text_cfg_mask = torch.full((1, T, 1), self.guidance_scale_text, device=device)
        ramp = torch.linspace(0.0, self.guidance_scale_text, prefix_frames, device=device)
        text_cfg_mask[:, :prefix_frames, :] = ramp.unsqueeze(0).unsqueeze(-1)

        # --- Padded prefix + persistent noise ---
        prefix_padded = torch.zeros(batch_size, T, self.input_dim, device=device)
        pf = min(prefix_frames, prefix_motion.shape[1])
        prefix_padded[:, :pf, :] = prefix_motion[:, -pf:, :]
        prefix_noise_persistent = torch.randn_like(prefix_padded)

        return {
            'blend_mask': blend_mask,
            'text_cfg_mask': text_cfg_mask,
            'prefix_padded': prefix_padded,
            'prefix_noise_persistent': prefix_noise_persistent,
        }

    def _apply_dynamic_cfg(self, noise_pred, prefix_ctx):
        """
        Apply classifier-free guidance with optional temporal text weighting.

        When *prefix_ctx* is provided the text guidance weight ramps from 0
        over the overlap frames so that the network trusts the motion prefix
        rather than restarting the conditioned action (fixes "semantic restart").
        """
        if self.music_only:
            noise_pred_uncond, noise_pred_music = noise_pred.chunk(2)
            w_text = prefix_ctx['text_cfg_mask'] if prefix_ctx is not None else self.guidance_scale_text
            return noise_pred_uncond + self.guidance_scale_music * (noise_pred_music - noise_pred_uncond)

        noise_pred_uncond, noise_pred_text, noise_pred_full = noise_pred.chunk(3)
        w_text = prefix_ctx['text_cfg_mask'] if prefix_ctx is not None else self.guidance_scale_text
        return (noise_pred_uncond
                + w_text * (noise_pred_text - noise_pred_uncond)
                + self.guidance_scale_music * (noise_pred_full - noise_pred_text))

    def _apply_latent_blending(self, samples, timestep, prefix_ctx, device):
        """
        Smoothly inject the known prefix rotations at the current noise level.

        Only blends rotation channels (dims 3:).  Translation (dims 0:3) is
        left entirely to the network — the post-hoc stitch pipeline handles
        translation alignment via FK-based alignment + interpolation.

        Blending translation in noisy latent space causes sliding artifacts
        because the network's prior fights against non-zero starting positions.
        """
        blend_mask = prefix_ctx['blend_mask']
        prefix_padded = prefix_ctx['prefix_padded']
        prefix_noise = prefix_ctx['prefix_noise_persistent']

        # DDPM: use alphas_cumprod for noise schedule
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep.long()].to(device)
        known_noisy = (alpha_prod_t.sqrt().view(1, 1, 1) * prefix_padded
                        + (1 - alpha_prod_t).sqrt().view(1, 1, 1) * prefix_noise)

        # Blend rotation dims only (3:), leave translation (0:3) untouched
        blended = samples.clone()
        blended[:, :, 3:] = blend_mask * known_noisy[:, :, 3:] + (1 - blend_mask) * samples[:, :, 3:]
        return blended

    def _diffusion_reverse(self, music, attributes, description, label_index, noise=None, mask=None, audio_mask=None, att_mask=None,
                           prefix_motion=None, prefix_frames=75, blend_frames=20):
        time_start = time.time()
        if attributes is None:
            batch_size = music.shape[0]
            device = music.device
            num_frame = music.shape[1]
        else:
            batch_size = attributes.shape[0]
            device = attributes.device
            num_frame = attributes.shape[1]

        if self.do_classifier_free_guidance:
            # For 3-pass multi-condition CFG, attributes are tripled: [uncond, text_only, full]
            batch_size = batch_size // self.cfg_factor

        # 1. Sample random noise
        if noise is None:
            samples = torch.randn(
                (batch_size, num_frame, self.input_dim),
                device=device,
                dtype=torch.float
            )
        else:
            samples = noise

        # 2. Prepare prefix blending context (None when no prefix)
        prefix_ctx = self._prepare_prefix_blending(
            prefix_motion, prefix_frames, blend_frames, batch_size, num_frame, device)

        # 3. Set timesteps
        self.scheduler.set_timesteps(self.config.model.Diffusion.scheduler.num_inference_timesteps, device=device)
        timesteps = self.scheduler.timesteps.to(device)

        # 4. Denoise loop
        for step_idx, timestep in enumerate(timesteps):
            if self.do_classifier_free_guidance:
                sample_model_input = torch.cat([samples] * self.cfg_factor)
            else:
                sample_model_input = samples

            noise_pred, _ = self.select_model_and_forward(x=sample_model_input,
                                                       t=timestep,
                                                       att=attributes,
                                                       music=music,
                                                       description=description,
                                                       label_index=label_index,
                                                       mode='inference',
                                                       mask=mask,
                                                       audio_mask=audio_mask,
                                                       att_mask=att_mask)

            if self.do_classifier_free_guidance:
                noise_pred = self._apply_dynamic_cfg(noise_pred, prefix_ctx)

            extra_step_kwargs = {}
            samples = self.scheduler.step(noise_pred, timestep, samples, **extra_step_kwargs)['prev_sample']

            if prefix_ctx is not None:
                samples = self._apply_latent_blending(samples, timestep, prefix_ctx, device)

        time_end = time.time()
        print(f"Time taken for diffusion reverse: {time_end - time_start} seconds")

        return samples

    def eval_forward(self, batch):
        pred = self.forward(batch)
        output = {'motion_pred': pred['motion_fk'], 
                  'motion_ref': batch['motion'], 
                  'motion_joint': batch['motion_joint'],
                  'key': batch['key'],
                  'current_motion_fps': batch['current_motion_fps'],
                  'target_motion_fps': batch['target_motion_fps']}
        return output

    def train_forward(self, batch, mode):
        motion = batch['motion']
        audio = batch['audio']
        att = batch['attributes']
        desc = batch['description']
        if 'label_index' in batch:
            label_index = batch['label_index']
        else:
            label_index = None

        if self.normalizer is not None:
            motion = self.normalizer.normalize(motion)
        
        if 'mask' in batch:
            mask = batch['mask']
        else:
            mask = None
        
        results = self._diffusion_forward(motion, audio, att, desc, label_index, mode, mask=mask, audio_mask=batch['audio_mask'], att_mask=batch['att_mask'])

        # Compute embeddings for disentangle and contrastive losses
        if self.config.loss.lambda_disentangle > 0 or self.config.loss.lambda_contrastive > 0:
            # Music embedding for disentangle loss (from model output)
            if 'music' in results:
                results['music_embed'] = results['music']  # [B, T, D] from FiLM projection

            # Text embedding for disentangle and contrastive losses (from model output)
            if 'att' in results:
                results['text_embed'] = results['att']  # [B, 2, D] attributes + description
                # Pool text embedding for contrastive loss
                results['text_embed_pooled'] = results['att'].mean(dim=1)  # [B, D]

            # Motion embedding for contrastive loss - use GT motion with stop_gradient (Option A)
            if self.config.loss.lambda_contrastive > 0:
                results['motion_embed'] = self.model.get_motion_embedding(motion, mask=mask).detach()  # [B, D]

        output = self.combine_output(batch, results)

        # output = {**combined_out, **results}
        
        return output
    
    def foot_sliding_optimization(self, body_pose, initial_trans, iters=50):
        optimized_trans = initial_trans.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([optimized_trans], lr=0.01)

        for i in range(iters):
            optimizer.zero_grad()

            joints = self.fk.forward(torch.concat([optimized_trans, body_pose], axis=-1), enable_grad=True)
            feet = joints[:, :, [10, 11], :]
            contact = compute_foot_contact(joints) # [bs, T, 2]
            feet_vel = feet[:, 1:] - feet[:, :-1] # [bs, T, 2, 3]
            # feet_vel = torch.stack((feet_vel[..., 0], feet_vel[..., 2]), dim=-1)
            trans_accel = optimized_trans[:, 2:] - 2 * optimized_trans[:, 1:-1] + optimized_trans[:, :-2]

            loss = (feet_vel.norm(dim=-1) * contact).mean()

            loss += 0.01 * torch.nn.functional.mse_loss(optimized_trans, initial_trans)
            loss += 0.01 * trans_accel.norm(dim=-1).mean()
            
            loss.backward()
            optimizer.step()

        return optimized_trans.detach()


    def step(self, batch, batch_idx, mode):
        if 'mask' in batch:
            mask = batch['mask']
        else:
            mask = None

        # Extract label_index for coverage weighting (Option 1 - Fixed Window)
        if 'label_index' in batch:
            label_index = batch['label_index']
        else:
            label_index = None

        loss = None

        # Training: compute loss only
        if mode == "train":
            results = self.train_forward(batch, mode)
            loss = self.losses[mode].update(results, self.config.data.name, self.skeleton_dict, mask, label_index=label_index)

        # Validation: accumulate loss AND metrics (both computed at epoch end)
        elif mode == "val":
            # Accumulate validation loss (computed/logged at epoch end)
            results = self.train_forward(batch, mode)
            loss = self.losses[mode].update(results, self.config.data.name, self.skeleton_dict, mask, label_index=label_index)

            # Run inference and accumulate metrics (computed at epoch end)
            eval_results = self.eval_forward(batch)
            for metric in self.metrics_dict:
                if metric == 'mpjpe':
                    getattr(self, metric).update(eval_results['motion_pred'], eval_results['motion_joint'], mask=mask)
                elif metric == 'mpjve':
                    getattr(self, metric).update(eval_results['motion_pred'], eval_results['motion_joint'], mask=mask)
                elif metric == 'beat_alignment':
                    getattr(self, metric).update(eval_results['motion_pred'],
                                                 eval_results['key'],
                                                 eval_results['current_motion_fps'],
                                                 batch['target_motion_fps'],
                                                 batch['dataset'],
                                                 batch['audio_mask'],
                                                 mask=mask)
                else:
                    raise ValueError(f"Metric {metric} not supported")

        # Test: only accumulate metrics (computed at epoch end)
        elif mode == "test":
            eval_results = self.eval_forward(batch)
            for metric in self.metrics_dict:
                if metric == 'mpjpe':
                    getattr(self, metric).update(eval_results['motion_pred'], eval_results['motion_joint'], mask=mask)
                elif metric == 'mpjve':
                    getattr(self, metric).update(eval_results['motion_pred'], eval_results['motion_joint'], mask=mask)
                elif metric == 'beat_alignment':
                    getattr(self, metric).update(eval_results['motion_pred'],
                                                 eval_results['key'],
                                                 eval_results['current_motion_fps'],
                                                 batch['target_motion_fps'],
                                                 batch['dataset'],
                                                 batch['audio_mask'],
                                                 mask=mask)
                else:
                    raise ValueError(f"Metric {metric} not supported")

        if mode == "test":
            return eval_results

        return loss

    def forward(self, batch, vertices=False, foot_optim=False,
                prefix_motion=None, prefix_frames=75, blend_frames=20):
        if self.config.data.motion_preprocessing == 'relative_pelvis_origin':
            forward_type = 'relative_pose'
        else:
            forward_type = None
        
        output = {}
        # motion = batch['motion']
        audio = batch['audio']
        att = batch['attributes']
        desc = batch['description']
        if 'label_index' in batch:
            label_index = batch['label_index']
        else:
            label_index = None

        if self.do_classifier_free_guidance:
            if self.music_only:
                attributes = None
                description = None
                if label_index is not None:
                    label_index = torch.cat([label_index, label_index], dim=0)
                music = torch.cat([torch.zeros_like(audio), audio], dim=0)
            else:
                # 3-pass multi-condition CFG: [uncond, text_only, full]
                # Pass 1 (uncond):    text=null, music=null
                # Pass 2 (text_only): text=real, music=null
                # Pass 3 (full):      text=real, music=real
                attributes = torch.cat([torch.zeros_like(att), att, att], dim=0)
                description = torch.cat([torch.zeros_like(desc), desc, desc], dim=0)
                music = torch.cat([torch.zeros_like(audio), torch.zeros_like(audio), audio], dim=0)
            if label_index is not None:
                label_index = torch.cat([label_index] * self.cfg_factor, dim=0)
            audio_mask = torch.cat([batch['audio_mask']] * self.cfg_factor, dim=0)
            att_mask = torch.cat([batch['att_mask']] * self.cfg_factor, dim=0)
        else:
            attributes = att
            music = audio
            label_index = label_index
            description = desc
            audio_mask = batch['audio_mask']
            att_mask = batch['att_mask']
        if 'noise' in batch:
            noise = batch['noise']
        else:
            noise = None
        if 'mask' in batch:
            mask = torch.cat([batch['mask']] * (self.cfg_factor if self.do_classifier_free_guidance else 1), dim=0)
        else:
            mask = None
        
        samples = self._diffusion_reverse(music, attributes, description, label_index, noise, mask=mask, audio_mask=audio_mask, att_mask=att_mask,
                                          prefix_motion=prefix_motion, prefix_frames=prefix_frames, blend_frames=blend_frames)
        normalized_samples = samples.clone()

        if self.normalizer is not None:
            samples = self.normalizer.unnormalize(samples)

        motion_processed = motion_postprocessing(samples, type=forward_type, data_type=self.config.data.type)

        if foot_optim:
            print("Run foot-sliding optimization process")
            optimized_trans = self.foot_sliding_optimization(motion_processed[:, :, 3:], motion_processed[:, :, :3])
            motion_processed = torch.cat([optimized_trans, motion_processed[:, :, 3:]], axis=-1)

        if self.data_type == 'tree' and not self.absolute:
            if self.data_rep == 'motorica':
                motion_fk = torch.stack([self.fk.forward(motion_processed[i], fill_dummy=True, skeleton=self.skeleton_dict[batch['key'][i]]) for i in range(motion_processed.shape[0])])
            elif self.data_rep == 'smpl':
                if vertices:
                    motion_kp, motion_verts = [], []
                    for i in range(motion_processed.shape[0]):
                        kp, verts = self.fk.forward(motion_processed[i], return_verts=True)
                        motion_kp.append(kp)
                        motion_verts.append(verts)
                    motion_fk = torch.stack(motion_kp)
                    motion_verts = torch.stack(motion_verts)
                    output['motion_verts'] = motion_verts
                else:
                    motion_fk = self.fk.forward(motion_processed)
            
            output['motion'] = motion_processed
            output['motion_fk'] = motion_fk
            output['normalized_motion'] = normalized_samples
            return output
        else:
            output['motion_fk'] = samples.reshape(samples.shape[0], samples.shape[1], self.num_joints, self.body_features)
            output['normalized_motion'] = normalized_samples
            return output
        
    def combine_output(self, batch, pred):
        output = {}
        # Necessary Data
        output['key'] = batch['key']
        output['current_motion_fps'] = batch['current_motion_fps']
        output['audio_mask'] = batch['audio_mask']

        # Ground Truth
        output['gt_motion_joint'] = batch['motion_joint'] # unnormalized joint positions
        if not self.absolute:
            output['gt_motion'] = batch['motion'] # unnormalized SMPL motion params

        # Main Loss
        if self.model_type=='diffusion':
            output['target'] = pred['target'] # normalized SMPL motion params
            output['model_output'] = pred['model_output']
        elif self.model_type=='flow':
            output['gt_vector_field'] = pred['gt_vector_field']
            output['pred_vector_field'] = pred['pred_vector_field']

        # Auxiliary Loss
        if self.model_type=='diffusion':
            output['pred_motion'] = pred['pred_motion']

        # For disentangle and contrastive losses
        if 'music_embed' in pred:
            output['music_embed'] = pred['music_embed']
        if 'text_embed' in pred:
            output['text_embed'] = pred['text_embed']
        if 'motion_embed' in pred:
            output['motion_embed'] = pred['motion_embed']
        if 'text_embed_pooled' in pred:
            output['text_embed_pooled'] = pred['text_embed_pooled']

        # Auxiliary loss weight (timestep-aware)
        if 'aux_weight' in pred:
            output['aux_weight'] = pred['aux_weight']

        # Normalization
        if self.normalizer is not None:
            output['pred_motion'] = self.normalizer.unnormalize(output['pred_motion']) # unnormalized SMPL motion params
        if self.absolute:
            B, T, _ = output['pred_motion'].shape
            output['pred_motion'] = output['pred_motion'].reshape(B, T, self.num_joints, self.body_features)
            
        return output

    def update_ema(self):
        """Update EMA model parameters.

        No @rank_zero_only: after DDP gradient sync, all ranks have identical
        model params, so each rank can update its own EMA copy independently.
        This keeps EMA weights consistent across GPUs without needing a broadcast.
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.model_ema.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'train')
        if self.use_ema and self.model_ema is not None:
            self.update_ema()

        return loss

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def all_epoch_end(self, mode):
        if self.trainer.sanity_checking:
            # Reset metrics and losses after sanity check without logging
            for metric in self.metrics_dict:
                getattr(self, metric).reset()
            self.losses['val'].reset()
            return

        log_output = {}

        if mode in ["train", "val"]:
            loss_dict = self.losses[mode].compute()
            self.losses[mode].reset()
            log_output.update({
                self.losses[mode].loss2logname(loss_name, mode): value
                for loss_name, value in loss_dict.items()
            })
            log_output.update({
                "epoch": float(self.trainer.current_epoch),
            })

        if mode in ["val", "test"]:
            # Compute accumulated metrics
            for metric in self.metrics_dict:
                metric_value = getattr(self, metric).compute()
                log_output.update({
                    f"Metrics/{metric}/{mode}": metric_value
                })
                if metric == 'beat_alignment':
                    beat_all = getattr(self, metric).compute_all()
                    for k, v in beat_all.items():
                        log_output[f"Metrics/{k}/{mode}"] = v
                getattr(self, metric).reset()

        if log_output:
            self.log_dict(log_output, sync_dist=True, rank_zero_only=True)

    def on_train_epoch_end(self):
        self.all_epoch_end('train')

    def on_validation_epoch_end(self):
        self.all_epoch_end('val')

    def on_test_epoch_end(self):
        self.all_epoch_end('test')

    def configure_optimizers(self):
        # Set Optimizer — separate param groups for weight decay
        if self.config.optim.params.get('weight_decay', 0.0) > 0:
            decay_params = []
            no_decay_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if 'AdaLN' in name or 'bias' in name or 'norm' in name or 'film' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            param_groups = [
                {'params': decay_params, 'weight_decay': self.config.optim.params.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ]
            optim_params = {k: v for k, v in self.config.optim.params.items() if k != 'weight_decay'}
            optimizer = get_obj_from_str(self.config.optim._target_)(
                params=param_groups,
                **optim_params
            )
        else:
            optimizer = get_obj_from_str(self.config.optim._target_)(
                params=self.parameters(),
                **self.config.optim.params
            )

        # Main LR scheduler
        main_scheduler = get_obj_from_str(self.config.lr_scheduler._target_)(
            optimizer,
            **self.config.lr_scheduler.params
        )

        # Warmup scheduler (if configured)
        warmup_steps = getattr(self.config, 'warmup', {}).get('steps', 0)
        if warmup_steps > 0:
            warmup_start_factor = getattr(self.config, 'warmup', {}).get('start_factor', 0.01)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            lr_scheduler = main_scheduler

        if self.use_ema and self.model_ema is None:
            self.model_ema = copy.deepcopy(self.model)
            for param in self.model_ema.parameters():
                param.requires_grad = False
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def configure_metrics(self):
        for metric in self.metrics_dict:
            if metric == 'mpjpe':
                self.mpjpe = MPJPE(self.config)
            elif metric == 'mpjve':
                self.mpjve = MPJVE(self.config)
            elif metric == 'beat_alignment':
                self.beat_alignment = BeatAlignment(self.config)
            else:
                raise ValueError(f"Metric {metric} not supported")

    def load_state_dict(self, state_dict, strict=True):
        """Override PyTorch Lightning's load_state_dict to handle EMA parameters"""
        from collections import OrderedDict
        
        # Separate model and EMA parameters
        model_dict = OrderedDict()
        ema_dict = OrderedDict()
        other_dict = OrderedDict()
        
        for k, v in state_dict.items():
            if k.startswith("model."):
                name = k.replace("model.", "")
                model_dict[name] = v
            elif k.startswith("model_ema.") and self.use_ema:
                name = k.replace("model_ema.", "")
                ema_dict[name] = v
            else:
                # Keep other Lightning-specific parameters (optimizers, lr_schedulers, etc.)
                other_dict[k] = v
        
        # Load the main model parameters
        if model_dict:
            self.model.load_state_dict(model_dict, strict=strict)
            print(f"Loaded {len(model_dict)} main model parameters")
        
        # Load EMA model parameters if available
        if self.use_ema and ema_dict:
            if self.model_ema is None:
                # Initialize EMA model if not already done
                import copy
                self.model_ema = copy.deepcopy(self.model)
                for param in self.model_ema.parameters():
                    param.requires_grad = False
            
            self.model_ema.load_state_dict(ema_dict, strict=strict)
            print(f"Loaded {len(ema_dict)} EMA model parameters")
        
        # Load other Lightning parameters using the parent class method
        if other_dict:
            super().load_state_dict(other_dict, strict=False)

    def on_load_checkpoint(self, checkpoint):
        """Called when loading a checkpoint. Handle optimizer state loading gracefully."""
        print("on_load_checkpoint called - checking optimizer compatibility...")
        
        # Print checkpoint keys for debugging
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Always clear optimizer states to avoid parameter group mismatch
        # This is a safe approach - model weights will be loaded but optimizer starts fresh
        if 'optimizer_states' in checkpoint:
            print(f"Found optimizer_states - clearing to avoid parameter group mismatch")
            checkpoint['optimizer_states'] = []
        
        if 'lr_schedulers' in checkpoint:
            print(f"Found lr_schedulers - clearing to match optimizer clearing")
            checkpoint['lr_schedulers'] = []
            
        print("Optimizer and scheduler states cleared - will start with fresh states")

    def load_pretrained_model(self, path, trainable=False):
        print("Loading pretrained model from {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        
        from collections import OrderedDict
        model_dict = OrderedDict()
        ema_dict = OrderedDict()
        
        for k, v in state_dict.items():
            if k.startswith("model."):
                name = k.replace("model.", "")
                model_dict[name] = v
            elif k.startswith("model_ema.") and self.use_ema:
                name = k.replace("model_ema.", "")
                ema_dict[name] = v
        
        self.model.load_state_dict(model_dict, strict=True)
        
        # Load EMA model if available and EMA is enabled
        if self.use_ema and ema_dict:
            if self.model_ema is None:
                self.model_ema = copy.deepcopy(self.model)
                for param in self.model_ema.parameters():
                    param.requires_grad = False
            self.model_ema.load_state_dict(ema_dict, strict=True)
            print("EMA model loaded successfully")
        
        if trainable:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            if self.use_ema and self.model_ema is not None:
                self.model_ema.eval()

    # def train_dataloader(self):
    #     return self.dataset.train_dataloader(shuffle=True)
    
    # def val_dataloader(self):
    #     return self.dataset.val_dataloader()
    
    # def test_dataloader(self):
    #     return self.dataset.test_dataloader()
