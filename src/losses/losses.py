import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric
from torch.nn import functional as F

from .kl import KLLoss

from src.skeleton.utility import get_motorica_skeleton_names
from src.skeleton.forward_kinematics import ForwardKinematics
from src.skeleton.smpl_fk import SMPLModel
from src.losses.utils import *
from src.skeleton.preprocessing import motion_postprocessing

class LossesCollection():
    def __init__(self, config):
            self.count = 0
            self.config = config
            self.num_joints = config.data.num_joints
            self.model_name = config.model.name
            self.dataset = config.data.name
            self.data_type = config.data.type
            self.data_rep = config.data.representation
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            if self.config.data.motion_preprocessing == 'relative_pelvis_origin':
                self.forward_type = 'relative_pose'
            else:
                self.forward_type = None

            if self.data_type == 'tree':
                if self.data_rep == 'motorica':
                    self.fk = ForwardKinematics(selected_joints=get_motorica_skeleton_names())
                elif self.data_rep == 'smpl':
                    self.fk = SMPLModel(num_joints=self.num_joints, device=self.device)

            losses = []

            if self.model_name == 'vae':
                # KL Loss
                losses.append("kl_motion")

                # Reconstruction Loss
                losses.append("recon_motion") # normalized motion SMPL params loss

                if self.config.loss.lambda_vel > 0:
                    losses.append("L2_vel")
                if self.config.loss.lambda_acc > 0:
                    losses.append("L2_acc")

                if self.config.loss.lambda_foot_skate > 0:
                    losses.append("foot_skate")

                if self.data_type == 'tree':
                    losses.append("recon_joints") # Joints reconstruction loss after Forward Kinematics
                    losses.append("recon_global") # Global motion reconstruction loss
            else:           
                losses.append("x_loss")
                
                # Reconstruction Loss
                if self.config.loss.lambda_vel > 0:
                    # losses.append("L2_vellatent")
                    losses.append("L2_vel")
                    # if not self.config.data.absolute:
                    #     losses.append("L2_vellatent")
                if self.config.loss.lambda_acc > 0:
                    losses.append("L2_acc")

                if self.config.loss.lambda_recon > 0:
                    if config.model.Diffusion.predict_epsilon:
                        losses.append("recon_motion")
                    losses.append("recon_joints")
                    # losses.append("recon_global")

                if self.config.loss.lambda_cd > 0:
                    losses.append("cd_loss")
                    losses.append("regularization_energy")

                if self.config.loss.lambda_foot_skate > 0:
                    losses.append("foot_skate")
                if config.loss.lambda_disentangle > 0:
                    losses.append("disentangle_music")
                if config.loss.lambda_contrastive > 0:
                    losses.append("contrastive_MT")
            
            losses.append("total")
            
            self.losses = losses

            self._losses_fn = {}
            self._params = {}

            for loss in losses:
                keyword = loss.split("_")[0]
                if keyword == "kl":
                    self._losses_fn[loss] = KLLoss()
                    self._params[loss] = config.loss.lambda_kl
                elif keyword == "recon":
                    self._losses_fn[loss] = L2_loss # nn.MSELoss(reduction='mean')
                    self._params[loss] = config.loss.lambda_recon
                elif keyword == "mpjpe":
                    self._losses_fn[loss] = L2_loss
                    self._params[loss] = config.loss.lambda_recon
                    if loss == "mpjpe_vel":
                        self._params[loss] = config.loss.lambda_vel
                    elif loss == "mpjpe_acc":
                        self._params[loss] = config.loss.lambda_acc
                elif keyword == "sim":
                    self._losses_fn[loss] = weighted_cos_sim
                    if "sim_vel" in loss:
                        self._params[loss] = config.loss.lambda_vel
                    elif "sim_acc" in loss:
                        self._params[loss] = config.loss.lambda_acc
                elif keyword == "L1":
                    self._losses_fn[loss] = L1_loss
                    if "L1_vel" in loss:
                        self._params[loss] = config.loss.lambda_vel
                    elif loss == "L1_acc":
                        self._params[loss] = config.loss.lambda_acc
                elif keyword == "angle":
                    self._losses_fn[loss] = nn.MSELoss(reduction='mean')
                    self._params[loss] = config.loss.lambda_angle
                elif keyword == "cont":
                    self._losses_fn[loss] = InfoNCELoss(config.loss.temperature)
                    self._params[loss] = config.loss.lambda_infonce
                elif keyword == "x":
                    self._losses_fn[loss] = MSELoss # nn.MSELoss(reduction='mean')
                    self._params[loss] = 1
                elif keyword == "L2":
                    self._losses_fn[loss] = L2_loss
                    if "L2_vel" in loss:
                        self._params[loss] = 0.5 * config.loss.lambda_vel
                    elif loss == "L2_acc":
                        self._params[loss] = 0.5 * config.loss.lambda_acc
                    else:
                        self._params[loss] = 1
                elif keyword == "mse":
                    self._losses_fn[loss] = MSELoss #nn.MSELoss(reduction='mean') #L2_loss
                    self._params[loss] = 1
                elif keyword == "disentangle":
                    self._losses_fn[loss] = DisentangleLoss(config.loss.temperature)
                    self._params[loss] = config.loss.lambda_disentangle
                elif keyword == "contrastive":
                    self._losses_fn[loss] = MotionTextContrastiveLoss(config.loss.temperature)
                    self._params[loss] = config.loss.lambda_contrastive
                elif keyword == "foot":
                    self._losses_fn[loss] = foot_skate_loss
                    self._params[loss] = config.loss.lambda_foot_skate
                elif keyword == "cd":
                    self._losses_fn[loss] = contrastive_divergence
                    self._params[loss] = config.loss.lambda_cd
                elif keyword == "regularization":
                    self._losses_fn[loss] = regularization
                    self._params[loss] = config.loss.lambda_energy_reg
                else:
                    ValueError(f"Loss {loss} not supported")
                setattr(self, loss, 0)

    def update(self, results, data_name, skeleton=None, mask=None, label_index=None):
        total: float = 0.0
        motion_joint = None
        if 'weights' in results:
            weights = results['weights']
        else:
            weights = None

        # Timestep-aware auxiliary weight: down-weight aux losses at high noise levels
        # Only applied to auxiliary losses (recon, vel, acc, foot_skate), NOT the main loss
        aux_weight = results.get('aux_weight', None)

        # Keep original binary mask for velocity/acceleration computation (requires boolean AND)
        binary_mask = mask

        # Compute coverage weights for partial label handling (Option 1 - Fixed Window)
        # This down-weights frames from incomplete label segments
        use_coverage_weighting = getattr(self.config.loss, 'use_coverage_weighting', False)
        if use_coverage_weighting and label_index is not None:
            start_threshold = getattr(self.config.loss, 'coverage_start_threshold', 0.1)
            min_weight = getattr(self.config.loss, 'coverage_min_weight', 0.3)
            coverage_weights = compute_label_coverage_weights(
                label_index,
                start_threshold=start_threshold,
                min_weight=min_weight
            )
            # Create weighted mask for loss computation (keeps binary_mask separate)
            weighted_mask = apply_coverage_weights_to_mask(mask, coverage_weights)
        else:
            weighted_mask = mask

        if self.model_name == 'vae':
            # main loss
            recon_loss = self._update_loss('recon_motion', results['model_output'], results['target'], weights=weights, mask=weighted_mask)
            kl_loss = self._update_loss('kl_motion', results['dist_motion'], results['dist_ref'])
            total += recon_loss + kl_loss

        else:
            # === Main loss (NO aux_weight — always full strength) ===
            total += self._update_loss('x_loss', results['model_output'], results['target'], weights=weights, mask=weighted_mask)

        # === Auxiliary losses (WITH aux_weight — scaled by noise level) ===
        # Reconstruction loss
        if self.config.loss.lambda_recon > 0:
            if self.config.data.absolute:
                motion_joint = results['pred_motion']
            elif self.config.model.Diffusion.predict_epsilon:
                total += self._update_loss('recon_motion', results['pred_motion'], results['gt_motion'], aux_weight=aux_weight, mask=weighted_mask)
            if motion_joint is None:
                # pred = motion_postprocessing(results['pred_motion'], type=self.forward_type, data_type=self.config.data.type)
                pred = results['pred_motion']
                if self.data_rep == 'motorica':
                    motion_joint = torch.stack([self.fk.forward(pred[i], fill_dummy=True, skeleton=skeleton[results['key'][i]]) for i in range(pred.shape[0])])
                    motion_joint_gt = torch.stack([self.fk.forward(results['gt_motion'][i], fill_dummy=True, skeleton=skeleton[results['key'][i]]) for i in range(results['gt_motion'].shape[0])])
                elif self.data_rep == 'smpl':
                    motion_joint = self.fk.forward(pred, enable_grad=True)
                    motion_joint_gt = self.fk.forward(results['gt_motion'])

            total += self._update_loss('recon_joints', motion_joint, motion_joint_gt, aux_weight=aux_weight, weights=weights, mask=weighted_mask)

            if 'recon_global' in self.losses:
                pred_global = results['pred_motion'][..., :3]
                gt_global = results['gt_motion'][..., :3]
                total += self._update_loss('recon_global', pred_global, gt_global, aux_weight=aux_weight, mask=weighted_mask)

        if self.config.loss.lambda_vel > 0:
            vel_pred = compute_vel(motion_joint)
            # Use binary_mask for velocity mask computation (requires boolean AND)
            if binary_mask is not None:
                binary_mask_bool = binary_mask.bool() if binary_mask.dtype != torch.bool else binary_mask
                vel_binary_mask = binary_mask_bool[:, 1:] & binary_mask_bool[:, :-1]
                vel_binary_mask = torch.cat((vel_binary_mask, vel_binary_mask[:, -1:]), dim=1)
                # Apply coverage weights to velocity mask if enabled
                if use_coverage_weighting and label_index is not None:
                    vel_coverage = coverage_weights[:, 1:].minimum(coverage_weights[:, :-1])
                    vel_coverage = torch.cat((vel_coverage, vel_coverage[:, -1:]), dim=1)
                    vel_mask = vel_binary_mask.float() * vel_coverage
                else:
                    vel_mask = vel_binary_mask.float()
            else:
                vel_mask = None
            vel_loss = self._update_loss('L2_vel', vel_pred, compute_vel(motion_joint_gt), aux_weight=aux_weight, mask=vel_mask)
            total += vel_loss

        if self.config.loss.lambda_acc > 0:
            acc_pred = compute_acc(motion_joint)
            # Use binary_mask for acceleration mask computation (requires boolean AND)
            if binary_mask is not None:
                binary_mask_bool = binary_mask.bool() if binary_mask.dtype != torch.bool else binary_mask
                acc_binary_mask = binary_mask_bool[:, 2:] & binary_mask_bool[:, 1:-1] & binary_mask_bool[:, :-2]
                acc_binary_mask = torch.cat((acc_binary_mask, acc_binary_mask[:, -2:]), dim=1)
                # Apply coverage weights to acceleration mask if enabled
                if use_coverage_weighting and label_index is not None:
                    acc_coverage = coverage_weights[:, 2:].minimum(coverage_weights[:, 1:-1]).minimum(coverage_weights[:, :-2])
                    acc_coverage = torch.cat((acc_coverage, acc_coverage[:, -2:]), dim=1)
                    acc_mask = acc_binary_mask.float() * acc_coverage
                else:
                    acc_mask = acc_binary_mask.float()
            else:
                acc_mask = None
            acc_loss = self._update_loss('L2_acc', acc_pred, compute_acc(motion_joint_gt), aux_weight=aux_weight, mask=acc_mask)
            total += acc_loss

        if self.config.loss.lambda_disentangle > 0:
            disentangle_loss = self._update_loss('disentangle_music', results['music_embed'], results['text_embed'], mask=weighted_mask)
            total += disentangle_loss

        if self.config.loss.lambda_contrastive > 0:
            contrastive_loss = self._update_loss('contrastive_MT', results['motion_embed'], results['text_embed_pooled'])
            total += contrastive_loss

        if self.config.loss.lambda_foot_skate > 0:
            foot_skate_loss = self._update_loss('foot_skate', motion_joint, motion_joint_gt, aux_weight=aux_weight, mask=weighted_mask)
            total += foot_skate_loss

        with torch.no_grad():
            setattr(self, "total", getattr(self, "total") + total.item())
        self.count += 1

        return total
    
    def compute(self):
        count = getattr(self, "count")
        if count == 0:
            count = 1
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def reset(self):
        for loss in self.losses:
            setattr(self, loss, 0)
        setattr(self, "count", 0)
 
    def _update_loss(self, name: str, pred, ref=None, aux_weight=None, **kwargs):
        if name == "foot_skate":
            val = self._losses_fn[name](pred, ref, data_rep=self.data_rep)
        else:
            val = self._losses_fn[name](pred, ref, **kwargs)

        # Store metric without breaking computational graph (unweighted for logging)
        with torch.no_grad():
            setattr(self, name, getattr(self, name) + val.item())

        # Apply loss weight
        weighted_val = self._params[name] * val

        # Apply timestep-aware auxiliary weight (batch-mean)
        if aux_weight is not None:
            weighted_val = weighted_val * aux_weight.mean()

        return weighted_val
    
    def loss2logname(self, loss: str, mode: str):
        if loss == "total":
            log_name = f"{loss}/{mode}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{mode}"

        return log_name
