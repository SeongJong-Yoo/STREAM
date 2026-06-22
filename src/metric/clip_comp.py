"""
Compute CLIP score (motion-text cosine similarity) for saved motion .npy files.

Data format: reads .npy files saved by evaluate_and_save_results.py, each containing:
    - 'motion_gt'          : (T, D) rotation representation
    - 'motion_joint_gt'    : (T, J, 3) ground-truth 3D joints
    - 'motion_recon'       : (T, D) rotation representation (generated)
    - 'motion_joint_recon' : (T, J, 3) generated 3D joints
    - 'sample_id' or 'id'
    - 'data_name' or 'dataset'

Frame-level text labels are loaded from ./data/<dataset>/sliced_labels/<id>_label.npy
Each sample may contain multiple text segments; motion is sliced per segment.

Usage:
    python -m src.metric.clip_comp --exp_name <name> --dataset <dataset>
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.metric.evaluator_wrapper import Evaluators
from src.metric.back_process import back_process

# HumanML3D 22 joints = first 22 of SMPL 24 (drops L_Hand, R_Hand)
HML3D_JOINT_IDX = list(range(22))

MIN_SEGMENT_FRAMES = 10


def _to_hml3d(motion, motion_joint):
    """Convert SMPL 24-joint data to HumanML3D 22-joint format if needed."""
    if motion_joint.shape[1] == 22:
        return motion, motion_joint

    joints_22 = motion_joint[:, HML3D_JOINT_IDX, :]
    global_trans = motion[:, :3]
    rots = motion[:, 3:].reshape(motion.shape[0], -1, 6)
    rots_22 = rots[:, HML3D_JOINT_IDX, :]
    motion_22 = np.concatenate([global_trans, rots_22.reshape(motion.shape[0], -1)], axis=-1)
    return motion_22, joints_22


def _get_text_segments(label):
    """Extract text segments from frame-level label (T, 3) array.

    Segments are detected by transitions in column 1 (technique).

    Returns:
        list of (text_str, start_frame, end_frame) tuples.
    """
    T = label.shape[0]
    segments = []
    start = 0
    current_key = label[0, 1]

    for i in range(1, T):
        if label[i, 1] != current_key:
            text = _build_text(label[start])
            segments.append((text, start, i))
            start = i
            current_key = label[i, 1]

    # Last segment
    text = _build_text(label[start])
    segments.append((text, start, T))
    return segments


def _build_text(label_row):
    """Build text string from a single label row [genre, technique, description]."""
    parts = [str(label_row[col]).strip() for col in range(len(label_row))
             if str(label_row[col]).strip()]
    return ', '.join(parts) if parts else 'motion'


def _encode_clip(motion, motion_joint, text, eval_mean, eval_std, contrast_model, device):
    """Convert motion to 67-dim, normalize, encode, and return clip score."""
    motion, motion_joint = _to_hml3d(motion, motion_joint)

    feat = back_process(
        torch.from_numpy(motion).float(),
        torch.from_numpy(motion_joint).float(),
    )
    feat = (feat - eval_mean) / eval_std
    feat = torch.from_numpy(feat).float().unsqueeze(0).to(device)

    with torch.no_grad():
        em = contrast_model.encode_motion(feat, m_lens=150)
        et = contrast_model.encode_text([text])
        em = em / em.norm(dim=1, keepdim=True)
        et = et / et.norm(dim=1, keepdim=True)
    return (em @ et.T).item()


def _load_label(sample_id, data_name):
    """Load frame-level label array (T, 3) from sliced_labels."""
    npy_path = os.path.join('./data', data_name, 'sliced_labels',
                            f'{sample_id}_label.npy')
    if os.path.exists(npy_path):
        label = np.load(npy_path, allow_pickle=True)[()]
        if isinstance(label, dict):
            return label['data']
        return label
    return None


def calculate_clip_score(data_path, device, root_path=None):
    eval_mean = np.load('./src/metric/eval_mean.npy', allow_pickle=True)
    eval_std = np.load('./src/metric/eval_std.npy', allow_pickle=True)

    evaluator = Evaluators(device)
    contrast_model = evaluator.contrast_model

    files = sorted(f for f in os.listdir(data_path) if f.endswith('.npy'))
    clip_scores_gt = []
    clip_scores_recon = []

    for file in tqdm(files, desc='CLIP score'):
        data = np.load(os.path.join(data_path, file), allow_pickle=True)[()]

        if 'id' in data:
            sample_id = data['id'].split('/')[-1].split('.')[0]
        elif 'sample_id' in data:
            sample_id = data['sample_id']
        else:
            continue

        data_name = data.get('dataset', data.get('data_name', None))
        if data_name is None:
            continue

        # Load frame-level label and extract text segments
        label = _load_label(sample_id, data_name)
        if label is None:
            continue

        segments = _get_text_segments(label)

        for text, start, end in segments:
            if (end - start) < MIN_SEGMENT_FRAMES:
                continue

            # GT
            clip_scores_gt.append(
                _encode_clip(data['motion_gt'][start:end],
                             data['motion_joint_gt'][start:end],
                             text, eval_mean, eval_std, contrast_model, device)
            )

            # Recon
            clip_scores_recon.append(
                _encode_clip(data['motion_recon'][start:end],
                             data['motion_joint_recon'][start:end],
                             text, eval_mean, eval_std, contrast_model, device)
            )

    results = {
        'clip_score_gt': float(np.mean(clip_scores_gt)) if clip_scores_gt else 0.0,
        'clip_score_recon': float(np.mean(clip_scores_recon)) if clip_scores_recon else 0.0,
    }

    if root_path is not None:
        with open(os.path.join(root_path, 'metrics.txt'), 'a') as f:
            for key, value in results.items():
                f.write(f'{key}: {value}\n')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='motorica')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    root = os.path.join('./results', args.exp_name, args.dataset)
    data_path = os.path.join(root, 'motions')

    print(calculate_clip_score(data_path, device, root_path=root))
