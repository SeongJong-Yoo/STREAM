from torchmetrics import Metric
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from .losses import compute_vel, compute_acc
from einops import rearrange
import os

class MPJPE(Metric):
    def __init__(self, config):
        super().__init__(dist_sync_on_step=config.dist_sync_on_step)
        self.name = 'mpjpe'
        self.add_state('sum_error', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_joints', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        assert preds.shape == target.shape, "Shape mismatch at MPJPE Metric"
        error =torch.norm(preds - target, dim=-1)

        error = error.mean(dim=-1)
        if mask is not None:
            self.sum_error += (error * mask).sum()
            self.total_joints += mask.sum()
        else:
            self.sum_error += error.sum()
            self.total_joints += error.numel()

    def compute(self):
        return self.sum_error / self.total_joints
    
class MPJVE(Metric):
    def __init__(self, config):
        super().__init__(dist_sync_on_step=config.dist_sync_on_step)
        self.name = 'mpjve'
        self.add_state('sum_error', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_joints', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        assert preds.shape == target.shape, "Shape mismatch at MPJVE Metric"
        vel_pred = compute_vel(preds)
        vel_target = compute_vel(target)

        error = torch.mean(torch.norm(vel_pred - vel_target, dim=-1), dim=-1)

        if mask is not None:
            self.sum_error += (error * mask).sum()
            self.total_joints += mask.sum()
        else:
            self.sum_error += error.sum()
            self.total_joints += error.numel()

    def compute(self):
        return self.sum_error / self.total_joints


class BeatAlignment(Metric):
    def __init__(self, config):
        super().__init__(dist_sync_on_step=config.dist_sync_on_step)
        self.config = config
        self.name = 'beat_alignment'

        self.cache = {} # Save beat data

        self.add_state('beat_alignment', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('beat_precision', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('beat_f1', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_len', default=torch.tensor(0), dist_reduce_fx='sum')


    def update(self, preds, keys, fps, target_fps, dataset_name, audio_mask=None, mask: torch.Tensor = None):
        #TODO: Implement mask
        for i, key in enumerate(keys):
            if audio_mask is not None and not audio_mask[i]:
                continue
            motion_beats = self.compute_motion_beat(preds[i], fps[i])
            if key not in self.cache:
                # path = self.config.data.dir + dataset_name + '/sliced_beats/' + key + '.npy'
                if self.config.data.beat_based:
                    path = os.path.join(self.config.data.dir, dataset_name[i]+'_beats', 'sliced_beats', key + '.npy')
                else:
                    path = os.path.join(self.config.data.dir, dataset_name[i], 'sliced_beats', key + '.npy')
                self.load_beat_data(path)
            recall, precision, f1 = self.compute_beat_alignment(motion_beats, self.cache[key], target_fps[i].cpu().numpy())
            self.beat_alignment += recall
            self.beat_precision += precision
            self.beat_f1 += f1
            self.total_len += 1


    def compute(self):
        return self.beat_alignment / self.total_len

    def compute_all(self):
        n = self.total_len
        return {
            'beat_recall': self.beat_alignment / n,
            'beat_precision': self.beat_precision / n,
            'beat_f1': self.beat_f1 / n,
        }

    def compute_beat_alignment(self, motion_beats, music_beats, fps):
        """Returns (recall, precision, f1)."""
        if len(motion_beats) == 0 or len(music_beats) == 0:
            return 0, 0, 0

        # Recall (standard BAS): for each music beat, find closest kinematic beat
        recall_sum = 0
        for music_beat in music_beats:
            min_value = np.min((motion_beats * fps - music_beat * fps)**2)
            recall_sum += np.exp(-min_value / 2 / 9)
        recall = recall_sum / len(music_beats)

        # Precision: for each kinematic beat, find closest music beat
        precision_sum = 0
        for motion_beat in motion_beats:
            min_value = np.min((music_beats * fps - motion_beat * fps)**2)
            precision_sum += np.exp(-min_value / 2 / 9)
        precision = precision_sum / len(motion_beats)

        # F1
        if precision + recall == 0:
            return 0, 0, 0
        f1 = 2 * (precision * recall) / (precision + recall)

        return recall, precision, f1

    def compute_motion_beat(self, kp, fps):
        """
        Compute motion beats from keypoints
        Input:
            kp: (T, J, 3)
        Output:
            motion_beats: (T)
        """
        if len(kp.shape) == 2:
            kp = rearrange(kp[:,  9:], 'f (j d) -> f j d', d=3)
        vel = torch.mean(torch.norm(kp[1:] - kp[:-1], dim=-1), dim=-1)

        vel = vel.cpu().numpy()

        vel = gaussian_filter(vel, sigma=5)
        motion_beat = argrelextrema(vel, np.less)[0]
        motion_beat = motion_beat / fps.cpu().numpy()

        return motion_beat
        
        
    def load_beat_data(self, path):
        data = np.load(path, allow_pickle=True)[()]
        id = path.split('/')[-1].split('.')[0]
        self.cache[id] = data

class FootSkateRatio(Metric):
# https://github.com/CarstenEpic/humos/blob/a176d31f63de9e872bf7e141c5d5a7fbbc2f9924/humos/src/model/metrics.py#L46
# https://github.com/korrawe/guided-motion-diffusion/blob/2f6264a9b793333556ef911981983082a1113049/data_loaders/humanml/utils/metrics.py#L204
    # The foot skate ratio represents the percentage of frames in which either foot skids more than a specified distance (2.5 cm)
    #  while maintaining contact with the ground (foot height < 5 cm).
    def __init__(self, slide_thresh=0.025, contact_thresh=0.05, device='cpu'):
        super().__init__()
        self.slide_thresh = slide_thresh
        self.contact_thresh = contact_thresh
        self.device_ = device

        self.add_state("skate_frame_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_frame_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, foot_joints: torch.Tensor):
        # check input shape
        if foot_joints.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {foot_joints.ndim}D tensor")
        if foot_joints.shape[2] != 2:
            raise ValueError(f"Expected 2 foot joints, got {foot_joints.shape[2]} joints")
        # foot_joints: [B, T, 2, 3]  (x, y, z)
        B, T, F, _ = foot_joints.shape

        # [B, T-1, 2]: horizontal displacement between adjacent frames
        disp_xy = torch.norm(foot_joints[:, 1:, :, :2] - foot_joints[:, :-1, :, :2], dim=-1)

        # [B, T-1, 2]: both frames must have contact
        contact_prev = foot_joints[:, :-1, :, 2] < self.contact_thresh
        contact_next = foot_joints[:, 1:, :, 2] < self.contact_thresh
        in_contact = torch.logical_and(contact_prev, contact_next)

        # [B, T-1, 2]: detect frames where foot is sliding while grounded
        sliding = torch.logical_and(in_contact, disp_xy > self.slide_thresh)

        # [B, T-1]: any foot sliding in that frame
        skate_frames = sliding.any(dim=-1)

        self.skate_frame_count += skate_frames.sum()
        self.total_frame_count += skate_frames.numel()

    def compute(self):
        if self.total_frame_count == 0:
            return torch.tensor(0.0, device=self.device_)
        return self.skate_frame_count.float() / self.total_frame_count



if __name__ == "__main__":
    import pickle
    from pathlib import Path
    id = 'gWA_sFM_cAll_d25_mWA5_ch07'
    metric = FootSkateRatio(slide_thresh=0.025, contact_thresh=0.05)

    origin_root = Path('/mnt/hdd/Dataset/AIST/keypoints3d')
    pred_root = Path('./data/AIST_beats/prediction_result')

    with open(origin_root / id + '.pkl', 'rb') as f:
        origin_data = pickle.load(f)

    pred_data = np.load(pred_root / id + '.npy', allow_pickle=True)[()]

    # Apply update
    metric.update(foot_joints)

    # Compute result
    skate_ratio = metric.compute()
    print(f"Skate Ratio: {skate_ratio.item():.4f}")
