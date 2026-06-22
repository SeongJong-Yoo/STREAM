"""
Editable Dance Score (EDS) Experiment — Variable-Length Segments + Tempo-Based Exchange

Evaluates the trade-off between semantic preservation (S_text) and
rhythmic adaptation (S_beat) during exchange editing using variable-length
segments from motorica_seg.

    EDS = (2 * S_text * S_beat) / (S_text + S_beat)

where both scores are normalized against Ground Truth human performance:
    S_text = min(1.0, Sim_exchange / Sim_GT)   [TMR clip-similarity]
    S_beat = min(1.0, BA_exchange / BA_GT)

Modes:
    Exchange (default):
        For each sample, re-generate with swapped music per condition.
        S_text and S_beat both depend on the exchange-regenerated motion.

    Text-only (--text_only):
        Generate once with zeroed audio (no music signal), then compute
        beat alignment against each exchange partner's music.  The same
        motion is reused across all conditions.  Serves as a baseline
        showing chance-level S_beat from text-conditioned motion alone.

Usage:
    # Exchange (default)
    python eds_experiment.py \
        --folder <model_output_folder> \
        --model <model_type> \
        --model_version <checkpoint_version> \
        --tmr_run_dir <tmr_checkpoint_dir> \
        [--tmr_dir ./TMR] \
        [--dataset motorica_seg]

    # Text-only baseline
    python eds_experiment.py \
        --folder <model_output_folder> \
        --model <model_type> \
        --model_version <checkpoint_version> \
        --tmr_run_dir <tmr_checkpoint_dir> \
        --text_only
"""

import os
import sys
import json
import random
import re
import textwrap
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import scipy.signal
if not hasattr(scipy.signal, 'hann'):
    try:
        from scipy.signal.windows import hann
        scipy.signal.hann = hann
    except ImportError:
        scipy.signal.hann = lambda n: np.hanning(n)
import librosa

from src.utils.inference.inference_util import load_config
from src.models.pl_module.cross_energy_edit_dance_joint import CrossEnergyEditDanceJoint
from src.metric.beat_align_score import calc_db, BA, BA_full, get_music_beat_fromwav
from src.metric.tmr_wrapper import TMRWrapper
from evaluate_and_save_seg import SegDataset, chunked_forward

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

MIN_FRAMES = 90          # 3 sec at 30 fps — skip segments shorter than this

# Exchange conditions: each sample gets up to 3 exchange partners
# Keys: condition name → (min_bpm_diff_exclusive, max_bpm_diff_exclusive_or_None)
EXCHANGE_CONDITIONS = {
    'low':  (10, 40),    # 10 < BPM diff < 40
    'mid':  (50, 80),    # 50 < BPM diff < 80
    'high': (90, None),  # 90 < BPM diff
}
REQUIRED_CONDITIONS = {'low', 'mid'}   # must have at least these to be listed

# Use system ffmpeg
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

SMPL_KINEMATIC_CHAIN = [
    [0, 1, 4, 7, 10],        # left leg
    [0, 2, 5, 8, 11],        # right leg
    [0, 3, 6, 9, 12, 15],    # spine to head
    [9, 13, 16, 18, 20, 22], # left arm
    [9, 14, 17, 19, 21, 23], # right arm
]

CHAIN_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']


def _parse_tmr_args():
    """Extract TMR-specific args before the main config parser runs."""
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--tmr_dir', type=str, default='./TMR',
                        help='Path to the TMR repository root')
    parser.add_argument('--tmr_run_dir', type=str, required=True,
                        help='TMR model run directory (contains config.json)')
    # Visualization flags
    parser.add_argument('--visualize', action='store_true',
                        help='Render side-by-side GT/exchange videos during EDS run')
    parser.add_argument('--vis_num_samples', type=int, default=5,
                        help='Max number of samples to visualize (default: 5)')
    parser.add_argument('--vis_output_dir', type=str, default='',
                        help='Output dir for videos (default: results/<exp>/eds/vis)')
    parser.add_argument('--vis_fps', type=int, default=30,
                        help='FPS for rendered videos (default: 30)')
    parser.add_argument('--vis_with_audio', action='store_true',
                        help='Merge exchange audio into video (requires ffmpeg)')
    # Text-only mode
    parser.add_argument('--text_only', action='store_true',
                        help='Text-only baseline: generate once with zeroed audio, '
                             'then compute EDS against exchange musics')
    tmr_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return tmr_args


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def compute_tempo(wav_path):
    """Estimate BPM from a wav file using librosa."""
    y, sr = librosa.load(wav_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.squeeze(tempo))


def build_tempo_exchange(keys, tempo_cache, length_cache,
                         conditions=None, required_conditions=None,
                         seed=42,
                         train_tempo_cache=None, train_length_cache=None,
                         exclusive=True):
    """Build multi-condition exchange pairs.

    For each key, find one exchange partner per BPM-diff condition.
    When exclusive=True (default), each partner is used at most once
    across all keys and conditions — no music overlap.

    Only keys that satisfy all *required_conditions* are included in
    the returned dict.

    Parameters
    ----------
    keys               : list of test-set sample keys
    tempo_cache        : dict  {key: BPM}  for test keys
    length_cache       : dict  {key: num_frames}  for test keys
    conditions         : dict  {name: (lo_exclusive, hi_exclusive_or_None)}
    required_conditions: set   condition names that must be found
    train_tempo_cache  : dict  {key: BPM}  for train candidates (optional)
    train_length_cache : dict  {key: num_frames}  for train candidates (optional)
    exclusive          : bool  if True, each partner used at most once globally

    Returns
    -------
    dict  {key: {condition_name: partner_key}}
    """
    if conditions is None:
        conditions = EXCHANGE_CONDITIONS
    if required_conditions is None:
        required_conditions = REQUIRED_CONDITIONS

    rng = random.Random(seed)

    # Unified candidate pool (test + train)
    pool = list(keys)
    pool_tempos = dict(tempo_cache)
    pool_lengths = dict(length_cache)

    if train_tempo_cache and train_length_cache:
        for k in train_tempo_cache:
            if k not in pool_tempos:
                pool.append(k)
                pool_tempos[k] = train_tempo_cache[k]
                pool_lengths[k] = train_length_cache[k]

    # Skip candidates with BPM == 0 (failed tempo detection)
    pool = [k for k in pool if pool_tempos.get(k, 0) > 0]

    rng.shuffle(pool)

    # Skip source keys with BPM == 0
    valid_keys = [k for k in keys if tempo_cache.get(k, 0) > 0]

    used_global = set()  # partners already assigned (exclusive mode)
    exchange = {}

    def _find_best(key, orig_tempo, orig_len, lo, hi, used_this_key, exclude_global):
        """Find best candidate: Pass 1 (length >= orig), Pass 2 (any length)."""
        best, best_len = None, -1
        for pass_num in (1, 2):
            for cand in pool:
                if cand == key or cand in used_this_key:
                    continue
                if exclude_global and cand in used_global:
                    continue
                diff = abs(orig_tempo - pool_tempos[cand])
                if diff <= lo:
                    continue
                if hi is not None and diff >= hi:
                    continue
                cand_len = pool_lengths[cand]
                if pass_num == 1 and cand_len < orig_len:
                    continue
                if cand_len > best_len:
                    best = cand
                    best_len = cand_len
            if best is not None:
                return best
        return None

    for key in valid_keys:
        orig_tempo = tempo_cache[key]
        orig_len = length_cache[key]
        pairs = {}
        used_this_key = set()  # each key's partners must be distinct

        for cond_name, (lo, hi) in conditions.items():
            # Try exclusive first, fall back to reuse if pool exhausted
            best = _find_best(key, orig_tempo, orig_len, lo, hi,
                              used_this_key, exclude_global=exclusive)
            if best is None and exclusive:
                best = _find_best(key, orig_tempo, orig_len, lo, hi,
                                  used_this_key, exclude_global=False)

            if best is not None:
                pairs[cond_name] = best
                used_this_key.add(best)

        # Only include if all required conditions are met
        if required_conditions.issubset(pairs.keys()):
            exchange[key] = pairs
            if exclusive:
                used_global.update(pairs.values())

    return exchange


def compute_beat_align(joints_np, wav_path):
    """Beat-Align scores for one motion / music pair.

    Parameters
    ----------
    joints_np : ndarray (T, 24, 3) or (T, 72)
    wav_path  : str — path to the .wav file

    Returns
    -------
    dict with keys 'recall', 'precision', 'f1', 'num_motion_beats',
    'num_music_beats'.  All 0.0 when the wav is missing, has no detected
    beats, or no dance beats.
    """
    zero = {'recall': 0.0, 'precision': 0.0, 'f1': 0.0,
            'num_motion_beats': 0, 'num_music_beats': 0}
    if not os.path.exists(wav_path):
        return zero
    joints = np.asarray(joints_np, dtype=np.float64)
    if joints.ndim == 3:
        joints = joints.reshape(joints.shape[0], -1)          # (T, J*3)
    T, D = joints.shape
    num_j = D // 3
    if num_j != 24:
        return zero
    roott = joints[:1, :3]
    joints = joints - np.tile(roott, (1, num_j))
    joints = joints.reshape(-1, num_j, 3)

    try:
        music_beats = get_music_beat_fromwav(wav_path, T)
    except Exception:
        return zero
    # Filter beats to motion's temporal range — get_music_beat_fromwav
    # may return beats beyond T when the wav is longer than the motion
    # (e.g. exchange partner's wav > original sample's motion length).
    music_beats = music_beats[music_beats < T]
    if len(music_beats) == 0:
        return zero
    dance_beats, _ = calc_db(joints)
    if len(dance_beats[0]) == 0:
        return zero

    f1, precision, recall = BA_full(music_beats, dance_beats)
    return {'recall': float(recall), 'precision': float(precision), 'f1': float(f1),
            'num_motion_beats': len(dance_beats[0]),
            'num_music_beats': len(music_beats)}


def build_exchange_input(data, exchange_audio, device):
    """Create model input with swapped audio from the exchange partner.

    The exchange partner's audio is guaranteed to be >= the original length
    (ensured by build_tempo_exchange), so we only truncate — never zero-pad.

    Parameters
    ----------
    data           : dict from dataset[idx]
    exchange_audio : Tensor (T_exch, D) — raw audio features of exchange partner
    device         : torch device
    """
    orig_T = data['audio'].shape[1]
    exch_audio = exchange_audio.unsqueeze(0).to(device)  # (1, T_exch, D)
    # Use the shorter of original and exchange lengths — avoid zero-padding
    T = min(orig_T, exch_audio.shape[1])
    exch_audio = exch_audio[:, :T, :]

    exch_input = {
        'audio': exch_audio,
        'attributes': data['attributes'][:, :T, :].clone(),
        'description': data['description'][:, :T, :].clone(),
        'audio_mask': data.get('audio_mask',
                               torch.tensor([True], dtype=torch.bool, device=device)),
        'att_mask': data.get('att_mask',
                             torch.tensor([True], dtype=torch.bool, device=device)),
        'text': data['text'],
    }
    return exch_input


def build_text_only_input(data, device):
    """Create model input with zeroed audio (text-only generation).

    Replaces the audio tensor with zeros so the model receives no music
    signal.  Text/attribute conditioning is preserved.

    Parameters
    ----------
    data   : dict from dataset[idx]
    device : torch device
    """
    text_input = {
        'audio': torch.zeros_like(data['audio']),
        'attributes': data['attributes'].clone(),
        'description': data['description'].clone(),
        'audio_mask': data.get('audio_mask',
                               torch.tensor([True], dtype=torch.bool, device=device)),
        'att_mask': data.get('att_mask',
                             torch.tensor([True], dtype=torch.bool, device=device)),
        'text': data['text'],
    }
    return text_input


def pad_input_to_length(input_dict, target_len):
    """Pad model input to target_len for consistent generation length.

    The model is trained on fixed-length (max_frames) chunks.  For shorter
    unchunked segments we zero-pad audio and repeat-extend attributes /
    description to match max_frames, then the caller truncates the output
    back to the original length.

    Returns (padded_dict, original_attr_len).
    """
    orig_len = input_dict['attributes'].shape[1]
    if orig_len >= target_len:
        return input_dict, orig_len

    padded = {}

    # Zero-pad audio
    audio = input_dict['audio']  # (1, T_audio, D)
    audio_pad = target_len - audio.shape[1]
    if audio_pad > 0:
        padded['audio'] = torch.nn.functional.pad(audio, (0, 0, 0, audio_pad))
    else:
        padded['audio'] = audio[:, :target_len, :]

    # Repeat the single text embedding (CLIP=512-d or T5=768-d, depending
    # on the checkpoint's label_converter) across target_len frames. Shape-
    # agnostic — the encoder choice is opaque to this function.
    padded['attributes'] = input_dict['attributes'][:, :1, :].repeat(1, target_len, 1)
    padded['description'] = input_dict['description'][:, :1, :].repeat(1, target_len, 1)

    # Pass through unchanged
    for k in ('audio_mask', 'att_mask', 'text'):
        if k in input_dict:
            padded[k] = input_dict[k]

    return padded, orig_len


# ──────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────

def plot_skeleton(ax, joints, alpha=1.0):
    """Plot a single 24-joint skeleton frame with colored chains.

    SMPL uses Y-up; matplotlib 3D uses Z-up.  We map:
        SMPL X -> mpl X,  SMPL Z -> mpl Y,  SMPL Y -> mpl Z (up).
    """
    for chain, color in zip(SMPL_KINEMATIC_CHAIN, CHAIN_COLORS):
        xs = joints[chain, 0]
        ys = joints[chain, 2]   # SMPL Z → mpl Y
        zs = joints[chain, 1]   # SMPL Y → mpl Z (up)
        ax.plot(xs, ys, zs, '-o', color=color, alpha=alpha,
                markersize=2, linewidth=1.5)


def setup_3d_limits(all_joints):
    """Compute consistent axis limits from all joint data.

    Accounts for Y↔Z swap (SMPL Y-up → matplotlib Z-up).
    """
    x_min, x_max = all_joints[:, :, 0].min(), all_joints[:, :, 0].max()
    y_min, y_max = all_joints[:, :, 2].min(), all_joints[:, :, 2].max()  # SMPL Z → mpl Y
    z_min, z_max = all_joints[:, :, 1].min(), all_joints[:, :, 1].max()  # SMPL Y → mpl Z

    margin = 0.2
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * (1 + margin)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    return {
        'xlim': (x_mid - max_range / 2, x_mid + max_range / 2),
        'ylim': (y_mid - max_range / 2, y_mid + max_range / 2),
        'zlim': (z_mid - max_range / 2, z_mid + max_range / 2),
    }


def draw_beat_timeline(ax_timeline, frame, total_frames,
                       music_beats, kinematic_beats, ba_dict):
    """Draw a horizontal timeline showing beats and current playhead."""
    ax_timeline.cla()
    ax_timeline.set_xlim(0, total_frames)
    ax_timeline.set_ylim(-0.5, 1.5)
    ax_timeline.set_yticks([0.3, 1.0])
    ax_timeline.set_yticklabels(['Music', 'Motion'], fontsize=7)
    ax_timeline.axhline(y=0.3, color='#cccccc', linewidth=0.5)
    ax_timeline.axhline(y=1.0, color='#cccccc', linewidth=0.5)

    if music_beats is not None and len(music_beats) > 0:
        ax_timeline.scatter(music_beats, [0.3] * len(music_beats),
                            marker='^', c='red', s=20, zorder=3, label='Music beat')

    if kinematic_beats is not None and len(kinematic_beats) > 0:
        ax_timeline.scatter(kinematic_beats, [1.0] * len(kinematic_beats),
                            marker='v', c='blue', s=20, zorder=3, label='Motion beat')

    ax_timeline.axvline(x=frame, color='black', linewidth=1.2, linestyle='-', zorder=4)

    if ba_dict:
        score_text = (f"BAS  F1={ba_dict['f1']:.3f}  "
                      f"P={ba_dict['precision']:.3f}  R={ba_dict['recall']:.3f}")
        ax_timeline.text(total_frames * 0.98, 1.4, score_text,
                         ha='right', va='top', fontsize=7, color='#555555')

    ax_timeline.set_xlabel('Frame', fontsize=7)
    ax_timeline.tick_params(axis='x', labelsize=6)
    ax_timeline.spines['top'].set_visible(False)
    ax_timeline.spines['right'].set_visible(False)


def extract_beats_for_vis(joints_np, wav_path):
    """Extract music beats and kinematic beats for visualization.

    Returns (music_beats, kinematic_beats, ba_dict) or (None, None, None).
    """
    if not os.path.exists(wav_path):
        return None, None, None

    joints = np.asarray(joints_np, dtype=np.float64)
    if joints.ndim == 3:
        joints = joints.reshape(joints.shape[0], -1)
    T = joints.shape[0]
    num_j = joints.shape[1] // 3

    roott = joints[:1, :3]
    joints = joints - np.tile(roott, (1, num_j))
    joints = joints.reshape(-1, num_j, 3)

    try:
        music_beats = get_music_beat_fromwav(wav_path, T)
    except Exception:
        return None, None, None
    music_beats = music_beats[music_beats < T]

    if len(music_beats) == 0:
        return None, None, None

    dance_beats, _ = calc_db(joints)
    kinematic = dance_beats[0] if len(dance_beats[0]) > 0 else np.array([])
    f1, prec, rec = BA_full(music_beats, dance_beats)

    return music_beats, kinematic, {'f1': f1, 'precision': prec, 'recall': rec}


def render_eds_sample(model_joints, gt_joints, output_path, fps=30,
                      title='', text_desc='',
                      music_beats_exch=None, kin_beats_model=None, ba_exch=None,
                      music_beats_orig=None, kin_beats_gt=None, ba_gt=None):
    """Render side-by-side GT vs model motion with beat timelines.

    Layout:
        +------------------+------------------+
        |   GT motion      |  Model motion    |
        |  (original audio)|  (exch audio)    |
        +------------------+------------------+
        |  GT beat timeline | Model beat timeline |
        +------------------+------------------+
    """
    T = min(model_joints.shape[0], gt_joints.shape[0])
    model_joints = model_joints[:T]
    gt_joints = gt_joints[:T]

    all_data = np.concatenate([model_joints, gt_joints], axis=0)
    lims = setup_3d_limits(all_data)

    fig = plt.figure(figsize=(16, 9))

    ax_gt = fig.add_axes([0.02, 0.28, 0.46, 0.65], projection='3d')
    ax_model = fig.add_axes([0.52, 0.28, 0.46, 0.65], projection='3d')

    ax_tl_gt = fig.add_axes([0.06, 0.08, 0.40, 0.15])
    ax_tl_model = fig.add_axes([0.56, 0.08, 0.40, 0.15])

    wrapped = '\n'.join(textwrap.wrap(text_desc, width=100)) if text_desc else ''

    def update(frame):
        for ax3d in [ax_gt, ax_model]:
            ax3d.cla()
            ax3d.set_xlim(*lims['xlim'])
            ax3d.set_ylim(*lims['ylim'])
            ax3d.set_zlim(*lims['zlim'])
            ax3d.set_xlabel('X', fontsize=6)
            ax3d.set_ylabel('Y', fontsize=6)
            ax3d.set_zlabel('Z', fontsize=6)
            ax3d.tick_params(labelsize=5)
            ax3d.view_init(elev=15, azim=-60 + frame * 0.3)

        plot_skeleton(ax_gt, gt_joints[frame])
        ax_gt.set_title('GT (original audio)', fontsize=9, pad=2)

        plot_skeleton(ax_model, model_joints[frame])
        ax_model.set_title('Model (exchanged audio)', fontsize=9, pad=2)

        draw_beat_timeline(ax_tl_gt, frame, T,
                           music_beats_orig, kin_beats_gt, ba_gt)
        draw_beat_timeline(ax_tl_model, frame, T,
                           music_beats_exch, kin_beats_model, ba_exch)

        fig.suptitle(f'{title}\n{wrapped}', fontsize=9, y=0.98)
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    if not output_path.endswith('.mp4'):
        output_path += '.mp4'
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(output_path, writer=writer)

    plt.close(fig)
    return output_path


def render_eds_multicond(gt_joints, cond_data, output_path, fps=30,
                         title='', text_desc=''):
    """Render GT + low/mid/high BPM exchange motions with beat timelines.

    Parameters
    ----------
    gt_joints    : ndarray (T, 24, 3)
    cond_data    : dict  {cond_name: {
                      'joints': ndarray (T, 24, 3),
                      'music_beats': array, 'kin_beats': array,
                      'ba': dict, 'bpm': float, 'exch_key': str}}
    output_path  : str
    fps          : int
    title        : str
    text_desc    : str

    Layout (4 columns):
        +--------+--------+--------+--------+
        |   GT   |  Low   |  Mid   |  High  |
        +--------+--------+--------+--------+
        | GT TL  | Low TL | Mid TL | High TL|
        +--------+--------+--------+--------+
    """
    cond_order = ['low', 'mid', 'high']
    active_conds = [c for c in cond_order if c in cond_data]
    n_cols = 1 + len(active_conds)  # GT + conditions

    T = gt_joints.shape[0]
    for c in active_conds:
        T = min(T, cond_data[c]['joints'].shape[0])
    gt_joints = gt_joints[:T]

    # Shared axis limits across all motions
    all_data = [gt_joints]
    for c in active_conds:
        all_data.append(cond_data[c]['joints'][:T])
    lims = setup_3d_limits(np.concatenate(all_data, axis=0))

    fig = plt.figure(figsize=(5 * n_cols, 8))
    col_w = 1.0 / n_cols
    pad = 0.02

    # Create axes: 3D skeleton on top, beat timeline on bottom
    ax3d_list = []
    ax_tl_list = []
    for i in range(n_cols):
        x0 = i * col_w + pad
        w = col_w - 2 * pad
        ax3d = fig.add_axes([x0, 0.28, w, 0.62], projection='3d')
        ax_tl = fig.add_axes([x0 + 0.02, 0.06, w - 0.04, 0.16])
        ax3d_list.append(ax3d)
        ax_tl_list.append(ax_tl)

    # Extract GT beats
    gt_mb = cond_data.get('_gt_music_beats')
    gt_kb = cond_data.get('_gt_kin_beats')
    gt_ba = cond_data.get('_gt_ba')
    gt_bpm = cond_data.get('_gt_bpm', 0)

    wrapped = '\n'.join(textwrap.wrap(text_desc, width=120)) if text_desc else ''

    def update(frame):
        # -- GT column --
        ax3d_list[0].cla()
        ax3d_list[0].set_xlim(*lims['xlim'])
        ax3d_list[0].set_ylim(*lims['ylim'])
        ax3d_list[0].set_zlim(*lims['zlim'])
        ax3d_list[0].tick_params(labelsize=4)
        ax3d_list[0].view_init(elev=15, azim=-60 + frame * 0.3)
        plot_skeleton(ax3d_list[0], gt_joints[frame])
        ax3d_list[0].set_title(f'GT ({gt_bpm:.0f} BPM)', fontsize=8, pad=2)

        draw_beat_timeline(ax_tl_list[0], frame, T, gt_mb, gt_kb, gt_ba)

        # -- Condition columns --
        for ci, cond_name in enumerate(active_conds):
            col = ci + 1
            cd = cond_data[cond_name]
            joints_c = cd['joints'][:T]

            ax3d_list[col].cla()
            ax3d_list[col].set_xlim(*lims['xlim'])
            ax3d_list[col].set_ylim(*lims['ylim'])
            ax3d_list[col].set_zlim(*lims['zlim'])
            ax3d_list[col].tick_params(labelsize=4)
            ax3d_list[col].view_init(elev=15, azim=-60 + frame * 0.3)
            plot_skeleton(ax3d_list[col], joints_c[frame])

            bpm_diff = abs(cd['bpm'] - gt_bpm)
            ax3d_list[col].set_title(
                f'{cond_name.upper()} ({cd["bpm"]:.0f} BPM, \u0394{bpm_diff:.0f})',
                fontsize=8, pad=2)

            draw_beat_timeline(ax_tl_list[col], frame, T,
                               cd.get('music_beats'), cd.get('kin_beats'),
                               cd.get('ba'))

        fig.suptitle(f'{title}\n{wrapped}', fontsize=9, y=0.98)
        return []

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    if not output_path.endswith('.mp4'):
        output_path += '.mp4'
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(output_path, writer=writer)

    plt.close(fig)
    return output_path


def merge_audio(video_path, wav_path, output_path):
    """Merge audio into video using ffmpeg. Returns True on success."""
    import subprocess
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', wav_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    # ── 0  TMR args (parsed first, stripped from sys.argv) ───
    tmr_args = _parse_tmr_args()

    # ── 1  Config / model / dataset ──────────────────────────
    config, args = load_config()
    config.evaluation = False

    data_name  = args.dataset if args.dataset else config.data.dataset[0]
    data_dir   = config.data.dir
    audio_root = os.path.join(data_dir, data_name, 'sliced_audio')

    text_only   = tmr_args.text_only

    exp_name    = config.name
    eds_subdir  = 'eds_text_only' if text_only else 'eds'
    results_dir = os.path.join('./results', exp_name, eds_subdir)
    os.makedirs(results_dir, exist_ok=True)

    if text_only:
        print('*** TEXT-ONLY mode: generate once with zeroed audio, '
              'then evaluate beat alignment against exchange musics ***')

    # Visualization setup
    do_vis = tmr_args.visualize
    vis_count = 0
    vis_max = tmr_args.vis_num_samples
    vis_fps = tmr_args.vis_fps
    vis_with_audio = tmr_args.vis_with_audio
    vis_dir = tmr_args.vis_output_dir or os.path.join(results_dir, 'vis')
    if do_vis:
        os.makedirs(vis_dir, exist_ok=True)
        print(f'Visualization enabled: up to {vis_max} samples → {vis_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset — variable-length segments via SegDataset
    print('Loading SegDataset …')
    dataset = SegDataset(config, args, device=device)
    dataset_info = dataset.get_dataset_info()

    all_keys = dataset.keys
    num_total = len(all_keys)
    print(f'  total test segments: {num_total}')

    # Model
    if 'joint' in config.model.name:
        model = CrossEnergyEditDanceJoint(config, dataset_info, mode='inference')
    model = model.to(device).eval()

    max_frames = int(config.data.motion_fps * config.data.chunk_time)

    # ── 2  Pre-compute tempos & audio lengths ───────────────
    pairs_path = os.path.join(data_dir, data_name, 'exchange_pairs_3cond.json')

    # Audio length cache for test keys (frames in audio features)
    length_cache = {}
    for key in all_keys:
        length_cache[key] = dataset.datas[key]['audio'].shape[0]

    # Filter to keys with >= MIN_FRAMES for pair building
    long_keys = [k for k in all_keys if length_cache[k] >= MIN_FRAMES]
    print(f'  keys >= {MIN_FRAMES} frames (3 sec): {len(long_keys)} / {num_total}')

    audio_feat_dir = Path(data_dir) / data_name / 'sliced_audio_features'

    if os.path.exists(pairs_path):
        print(f'  Loading cached exchange pairs from {pairs_path}')
        with open(pairs_path, 'r') as f:
            saved = json.load(f)
        exchange_pairs = saved['pairs']   # {key: {low: .., mid: .., high: ..}}
        tempo_cache = {k: v for k, v in saved.get('tempos', {}).items()}

        # Collect all unique partner keys that are from train set
        train_audio_pool = {}
        train_keys_used = set()
        for cond_dict in exchange_pairs.values():
            for partner in cond_dict.values():
                if partner not in dataset.datas:
                    train_keys_used.add(partner)
        if train_keys_used:
            print(f'  Loading {len(train_keys_used)} train audio features for exchange …')
            for tk in tqdm(train_keys_used, desc='Train audio'):
                feat_path = audio_feat_dir / f'{tk}_audio.npy'
                if feat_path.exists():
                    feat = np.load(feat_path, allow_pickle=True)[()]
                    train_audio_pool[tk] = torch.from_numpy(feat['audio'])
    else:
        print('Computing tempos from wav files …')
        tempo_cache = {}
        for key in tqdm(all_keys, desc='Tempo (test)'):
            wav_path = os.path.join(audio_root, key + '.wav')
            if os.path.exists(wav_path):
                try:
                    tempo_cache[key] = compute_tempo(wav_path)
                except Exception as e:
                    print(f'  WARNING: could not compute tempo for {key}: {e}')
                    tempo_cache[key] = 120.0
            else:
                print(f'  WARNING: wav not found for {key}')
                tempo_cache[key] = 120.0

        tempos_arr = np.array(list(tempo_cache.values()))
        print(f'  Test tempo stats — mean: {tempos_arr.mean():.1f}, '
              f'min: {tempos_arr.min():.1f}, max: {tempos_arr.max():.1f} BPM')

        # ── 2b  Load train audio pool (tempo + length only) ──
        print('Scanning train audio features for exchange pool …')
        from src.dataloader.motorica_dataset import MOTORICA_TEST_KEY

        all_audio_files = sorted(audio_feat_dir.glob('*.npy'))

        train_tempo_cache = {}
        train_length_cache = {}
        train_audio_pool = {}

        for feat_path in tqdm(all_audio_files, desc='Train pool'):
            fid = feat_path.stem.split('_audio')[0]
            base_id = fid.split('_chunk')[0]
            if base_id in MOTORICA_TEST_KEY:
                continue
            feat = np.load(feat_path, allow_pickle=True)[()]
            audio_len = feat['audio'].shape[0]
            train_length_cache[fid] = audio_len

            wav_path = os.path.join(audio_root, fid + '.wav')
            if os.path.exists(wav_path):
                try:
                    train_tempo_cache[fid] = compute_tempo(wav_path)
                except Exception:
                    train_tempo_cache[fid] = 120.0
            else:
                train_tempo_cache[fid] = 120.0

        print(f'  Train pool: {len(train_tempo_cache)} segments')

        # ── 3  Build multi-condition exchange pairs ───────────
        print('Building 3-condition exchange pairs (length-aware) …')
        print(f'  Conditions: {EXCHANGE_CONDITIONS}')
        print(f'  Required:   {REQUIRED_CONDITIONS}')

        exchange_pairs = build_tempo_exchange(
            long_keys, tempo_cache, length_cache,
            conditions=EXCHANGE_CONDITIONS,
            required_conditions=REQUIRED_CONDITIONS,
            seed=42,
            train_tempo_cache=train_tempo_cache,
            train_length_cache=train_length_cache,
        )

        # Report stats per condition
        print(f'\n  Total keys with required pairs: {len(exchange_pairs)} / {len(long_keys)}')
        for cname in EXCHANGE_CONDITIONS:
            diffs = []
            from_train = 0
            length_ok = 0
            count = 0
            for k, cond_dict in exchange_pairs.items():
                v = cond_dict.get(cname)
                if v is None:
                    continue
                count += 1
                t_partner = tempo_cache.get(v, train_tempo_cache.get(v, 0))
                diffs.append(abs(tempo_cache[k] - t_partner))
                if v not in tempo_cache:
                    from_train += 1
                exch_len = length_cache.get(v, train_length_cache.get(v, 0))
                if exch_len >= length_cache[k]:
                    length_ok += 1
            if diffs:
                print(f'  [{cname}] {count} pairs — '
                      f'Δ mean={np.mean(diffs):.1f} min={np.min(diffs):.1f} '
                      f'max={np.max(diffs):.1f} BPM, '
                      f'from_train={from_train}, len_ok={length_ok}')
            else:
                print(f'  [{cname}] 0 pairs')

        # Preload train audio features for selected partners
        train_keys_used = set()
        for cond_dict in exchange_pairs.values():
            for partner in cond_dict.values():
                if partner not in dataset.datas:
                    train_keys_used.add(partner)
        if train_keys_used:
            print(f'  Loading {len(train_keys_used)} train audio features …')
            for tk in tqdm(train_keys_used, desc='Train audio'):
                feat_path = audio_feat_dir / f'{tk}_audio.npy'
                if feat_path.exists():
                    feat = np.load(feat_path, allow_pickle=True)[()]
                    train_audio_pool[tk] = torch.from_numpy(feat['audio'])

        # Save for reproducibility
        saved = {
            'conditions': {k: list(v) for k, v in EXCHANGE_CONDITIONS.items()},
            'required': list(REQUIRED_CONDITIONS),
            'min_frames': MIN_FRAMES,
            'pairs': exchange_pairs,
            'tempos': tempo_cache,
        }
        with open(pairs_path, 'w') as f:
            json.dump(saved, f, indent=2)
        print(f'  Saved exchange pairs → {pairs_path}')

    # ── 4  Load TMR model ────────────────────────────────────
    print('Loading TMR model …')
    tmr = TMRWrapper(tmr_args.tmr_dir, tmr_args.tmr_run_dir, device)

    # ── 5  Run GT + exchange forward passes ──────────────────
    mode_label = 'text-only' if text_only else 'exchange'
    print(f'\nRunning {mode_label} experiment …')
    cond_names = list(EXCHANGE_CONDITIONS.keys())  # ['low', 'mid', 'high']

    # GT accumulators (one entry per sample, shared across conditions)
    gt_sim_scores = []
    gt_beat_f1    = []
    gt_beat_prec  = []
    gt_beat_rec   = []
    gt_beat_weighted = []
    gt_is_chunked = []   # parallel bool: was this sample chunked?

    # Per-condition exchange accumulators
    exch_sim_scores = {c: [] for c in cond_names}
    exch_beat_f1    = {c: [] for c in cond_names}
    exch_beat_prec  = {c: [] for c in cond_names}
    exch_beat_rec   = {c: [] for c in cond_names}
    exch_beat_weighted = {c: [] for c in cond_names}
    exch_is_chunked = {c: [] for c in cond_names}  # parallel bool per condition

    skipped = 0
    chunked_count = 0
    processed = 0

    for idx in tqdm(range(len(dataset)), desc='Samples'):
        data = dataset[idx]
        key = data['key']
        T = data['audio'].shape[1]

        # Skip short segments
        if T < MIN_FRAMES:
            skipped += 1
            continue

        # Check multi-condition exchange pairs exist for this key
        cond_dict = exchange_pairs.get(key)
        if cond_dict is None:
            skipped += 1
            continue

        # — GT TMR + BeatAlign (once per sample, from human ground truth) ─
        text = data['text']
        gt_motion_fk = data['gt_motion_joint'].cpu()  # (1, T, 24, 3)
        gt_sim_scores.extend(tmr.paired_scores(gt_motion_fk, [text]))

        gt_wav = os.path.join(audio_root, key + '.wav')
        if 'gt_motion_joint' in data:
            gt_joints = data['gt_motion_joint'].squeeze(0).cpu().numpy()
        else:
            gt_joints = None
        ba_gt_dict = compute_beat_align(gt_joints, gt_wav)
        gt_beat_f1.append(ba_gt_dict['f1'])
        gt_beat_prec.append(ba_gt_dict['precision'])
        gt_beat_rec.append(ba_gt_dict['recall'])
        gt_beat_weighted.append(ba_gt_dict['recall'])

        sample_chunked = T > max_frames
        gt_is_chunked.append(sample_chunked)

        if text_only:
            # ── Text-only: generate once with zeroed audio ────────
            text_input = build_text_only_input(data, device)
            with torch.no_grad():
                if T > max_frames:
                    text_out = chunked_forward(model, text_input, max_frames, device)
                    chunked_count += 1
                else:
                    text_out = model.forward(text_input)
            text_fk = text_out['motion_fk'].detach().cpu()  # (1, T, 24, 3)
            text_joints = text_fk.squeeze(0).numpy()

            # TMR (same for all conditions — motion doesn't change)
            tmr_sim_text = tmr.paired_scores(text_fk, [text])

            # BA per condition (same motion, different exchange music)
            vis_cond_data = {}
            for cond_name in cond_names:
                exch_key = cond_dict.get(cond_name)
                if exch_key is None:
                    continue
                exch_sim_scores[cond_name].extend(tmr_sim_text)
                exch_wav = os.path.join(audio_root, exch_key + '.wav')
                ba_exch = compute_beat_align(text_joints, exch_wav)
                exch_beat_f1[cond_name].append(ba_exch['f1'])
                exch_beat_prec[cond_name].append(ba_exch['precision'])
                exch_beat_rec[cond_name].append(ba_exch['recall'])
                _density_e = min(1.0, ba_exch['num_motion_beats'] / ba_exch['num_music_beats']) if ba_exch['num_music_beats'] > 0 else 0.0
                exch_beat_weighted[cond_name].append(ba_exch['recall'] * _density_e)
                exch_is_chunked[cond_name].append(sample_chunked)

                if do_vis and vis_count < vis_max:
                    mb_exch, kb_exch, ba_vis_exch = extract_beats_for_vis(
                        text_joints, exch_wav)
                    vis_cond_data[cond_name] = {
                        'joints': text_joints,
                        'music_beats': mb_exch,
                        'kin_beats': kb_exch,
                        'ba': ba_vis_exch,
                        'bpm': tempo_cache.get(exch_key, 0),
                        'exch_key': exch_key,
                        'wav': exch_wav,
                    }

            # ── Visualization: render all conditions together ──
            if do_vis and vis_count < vis_max and len(vis_cond_data) > 0:
                t_orig = tempo_cache.get(key, 0)
                mb_orig, kb_gt, ba_vis_gt = extract_beats_for_vis(gt_joints, gt_wav)
                vis_cond_data['_gt_music_beats'] = mb_orig
                vis_cond_data['_gt_kin_beats'] = kb_gt
                vis_cond_data['_gt_ba'] = ba_vis_gt
                vis_cond_data['_gt_bpm'] = t_orig

                bpm_strs = ', '.join(
                    f'{c}: {vis_cond_data[c]["bpm"]:.0f}'
                    for c in cond_names if c in vis_cond_data)
                vis_title = (f'{exp_name} [text-only] | {key}\n'
                             f'Original: {t_orig:.0f} BPM | {bpm_strs}')

                final_path = os.path.join(vis_dir, f'{key}_multicond.mp4')
                print(f'  [VIS {vis_count+1}/{vis_max}] {key}  '
                      f'({T} frames, {t_orig:.0f} BPM → {bpm_strs})')

                audio_cond = 'mid' if 'mid' in vis_cond_data else next(
                    (c for c in cond_names if c in vis_cond_data), None)

                if vis_with_audio and audio_cond and os.path.exists(vis_cond_data[audio_cond]['wav']):
                    tmp_path = os.path.join(vis_dir, f'{key}_multicond_tmp.mp4')
                    render_eds_multicond(
                        gt_joints, vis_cond_data, tmp_path,
                        fps=vis_fps, title=vis_title, text_desc=text)
                    if merge_audio(tmp_path, vis_cond_data[audio_cond]['wav'], final_path):
                        os.remove(tmp_path)
                        print(f'    → {final_path} (with {audio_cond} audio)')
                    else:
                        os.rename(tmp_path, final_path)
                        print(f'    → {final_path} (audio merge failed, silent)')
                else:
                    render_eds_multicond(
                        gt_joints, vis_cond_data, final_path,
                        fps=vis_fps, title=vis_title, text_desc=text)
                    print(f'    → {final_path}')

                vis_count += 1
        else:
            # ── Exchange: re-generate per condition with swapped music ─
            vis_cond_data = {}  # collect per-condition data for visualization
            for cond_name in cond_names:
                exch_key = cond_dict.get(cond_name)
                if exch_key is None:
                    continue

                # Resolve exchange audio tensor
                if exch_key in dataset.datas:
                    exch_audio = dataset.datas[exch_key]['audio']
                elif exch_key in train_audio_pool:
                    exch_audio = train_audio_pool[exch_key]
                else:
                    continue

                exch_input = build_exchange_input(data, exch_audio, device)
                exch_T = exch_input['audio'].shape[1]
                with torch.no_grad():
                    if exch_T > max_frames:
                        exch_out = chunked_forward(model, exch_input, max_frames, device)
                    else:
                        exch_out = model.forward(exch_input)
                exch_fk = exch_out['motion_fk'].detach().cpu()

                # TMR clip-similarity
                exch_sim_scores[cond_name].extend(tmr.paired_scores(exch_fk, [text]))

                # BeatAlign
                exch_wav = os.path.join(audio_root, exch_key + '.wav')
                ba_exch = compute_beat_align(exch_fk.squeeze(0).numpy(), exch_wav)
                exch_beat_f1[cond_name].append(ba_exch['f1'])
                exch_beat_prec[cond_name].append(ba_exch['precision'])
                exch_beat_rec[cond_name].append(ba_exch['recall'])
                _density_e = min(1.0, ba_exch['num_motion_beats'] / ba_exch['num_music_beats']) if ba_exch['num_music_beats'] > 0 else 0.0
                exch_beat_weighted[cond_name].append(ba_exch['recall'] * _density_e)
                exch_is_chunked[cond_name].append(sample_chunked)

                # Collect for multi-condition visualization
                if do_vis and vis_count < vis_max:
                    exch_joints_vis = exch_fk.squeeze(0).numpy()
                    mb_exch, kb_exch, ba_vis_exch = extract_beats_for_vis(
                        exch_joints_vis, exch_wav)
                    t_exch = tempo_cache.get(exch_key, 0)
                    vis_cond_data[cond_name] = {
                        'joints': exch_joints_vis,
                        'music_beats': mb_exch,
                        'kin_beats': kb_exch,
                        'ba': ba_vis_exch,
                        'bpm': t_exch,
                        'exch_key': exch_key,
                        'wav': exch_wav,
                    }

            # ── Visualization: render all conditions together ──
            if do_vis and vis_count < vis_max and len(vis_cond_data) > 0:
                t_orig = tempo_cache.get(key, 0)
                mb_orig, kb_gt, ba_vis_gt = extract_beats_for_vis(gt_joints, gt_wav)
                vis_cond_data['_gt_music_beats'] = mb_orig
                vis_cond_data['_gt_kin_beats'] = kb_gt
                vis_cond_data['_gt_ba'] = ba_vis_gt
                vis_cond_data['_gt_bpm'] = t_orig

                bpm_strs = ', '.join(
                    f'{c}: {vis_cond_data[c]["bpm"]:.0f}'
                    for c in cond_names if c in vis_cond_data)
                vis_title = (f'{exp_name} | {key}\n'
                             f'Original: {t_orig:.0f} BPM | {bpm_strs}')

                final_path = os.path.join(vis_dir, f'{key}_multicond.mp4')
                print(f'  [VIS {vis_count+1}/{vis_max}] {key}  '
                      f'({T} frames, {t_orig:.0f} BPM → {bpm_strs})')

                # Pick mid-condition wav for audio overlay (most representative)
                audio_cond = 'mid' if 'mid' in vis_cond_data else next(
                    (c for c in cond_names if c in vis_cond_data), None)

                if vis_with_audio and audio_cond and os.path.exists(vis_cond_data[audio_cond]['wav']):
                    tmp_path = os.path.join(vis_dir, f'{key}_multicond_tmp.mp4')
                    render_eds_multicond(
                        gt_joints, vis_cond_data, tmp_path,
                        fps=vis_fps, title=vis_title, text_desc=text)
                    if merge_audio(tmp_path, vis_cond_data[audio_cond]['wav'], final_path):
                        os.remove(tmp_path)
                        print(f'    → {final_path} (with {audio_cond} audio)')
                    else:
                        os.rename(tmp_path, final_path)
                        print(f'    → {final_path} (audio merge failed, silent)')
                else:
                    render_eds_multicond(
                        gt_joints, vis_cond_data, final_path,
                        fps=vis_fps, title=vis_title, text_desc=text)
                    print(f'    → {final_path}')

                vis_count += 1

        processed += 1

    print(f'\n  Processed: {processed}, Skipped: {skipped} '
          f'(< {MIN_FRAMES} frames or missing pairs)')
    print(f'  Chunked generation used: {chunked_count} times')

    # ── 6  Aggregate metrics ──────────────────────────────────
    print('\nAggregating metrics …')

    if processed == 0:
        raise RuntimeError('No samples were processed — check test set / data paths.')

    def _hmean(a, b):
        return 2.0 * a * b / (a + b) if (a + b) > 0 else 0.0

    def _aggregate_eds(gt_sim, gt_rec, exch_sim_dict, exch_rec_dict,
                       exch_f1_dict, exch_prec_dict, cond_names_list,
                       gt_rec_w=None, exch_rec_w_dict=None):
        """Compute per-condition and average EDS from score lists.

        Returns (cond_results_dict, avg_S_text, avg_S_beat, avg_EDS,
                 avg_S_beat_w, avg_EDS_w).
        """
        sim_gt = float(np.mean(gt_sim)) if len(gt_sim) > 0 else 0.0
        rec_gt = float(np.mean(gt_rec)) if len(gt_rec) > 0 else 0.0
        rec_w_gt = float(np.mean(gt_rec_w)) if (gt_rec_w is not None and len(gt_rec_w) > 0) else 0.0
        cr = {}
        for c in cond_names_list:
            n = len(exch_sim_dict[c])
            if n == 0:
                cr[c] = None
                continue
            sim_e  = float(np.mean(exch_sim_dict[c]))
            rec_e  = float(np.mean(exch_rec_dict[c]))  if exch_rec_dict[c]  else 0.0
            f1_e   = float(np.mean(exch_f1_dict[c]))   if exch_f1_dict[c]   else 0.0
            prec_e = float(np.mean(exch_prec_dict[c])) if exch_prec_dict[c] else 0.0
            w_e    = float(np.mean(exch_rec_w_dict[c])) if (exch_rec_w_dict and exch_rec_w_dict[c]) else 0.0
            s_text = min(1.0, max(0.0, sim_e / sim_gt)) if sim_gt > 0 else 0.0
            s_beat = min(1.0, max(0.0, rec_e / rec_gt)) if rec_gt > 0 else 0.0
            s_beat_w = min(1.0, max(0.0, w_e / rec_w_gt)) if rec_w_gt > 0 else 0.0
            cr[c] = {
                'num_pairs':            n,
                'TMR_sim_exchange':     sim_e,
                'BA_recall_exchange':   rec_e,
                'BA_F1_exchange':       f1_e,
                'BA_prec_exchange':     prec_e,
                'BA_weighted_exchange': w_e,
                'S_text':               s_text,
                'S_beat':               s_beat,
                'S_beat_weighted':      s_beat_w,
                'EDS':                  _hmean(s_text, s_beat),
                'EDS_weighted':         _hmean(s_text, s_beat_w),
            }
        active = [c for c in cond_names_list if cr[c] is not None]
        if active:
            a_text   = float(np.mean([cr[c]['S_text']          for c in active]))
            a_beat   = float(np.mean([cr[c]['S_beat']          for c in active]))
            a_beat_w = float(np.mean([cr[c]['S_beat_weighted'] for c in active]))
            a_eds    = float(np.mean([cr[c]['EDS']             for c in active]))
            a_eds_w  = float(np.mean([cr[c]['EDS_weighted']    for c in active]))
        else:
            a_text = a_beat = a_beat_w = a_eds = a_eds_w = 0.0
        return cr, a_text, a_beat, a_eds, a_beat_w, a_eds_w

    # --- Overall (all samples) ---
    gt_is_arr = np.array(gt_is_chunked, dtype=bool)

    cond_results, avg_S_text, avg_S_beat, avg_EDS, avg_S_beat_w, avg_EDS_w = _aggregate_eds(
        gt_sim_scores, gt_beat_rec,
        exch_sim_scores, exch_beat_rec, exch_beat_f1, exch_beat_prec,
        cond_names,
        gt_rec_w=gt_beat_weighted, exch_rec_w_dict=exch_beat_weighted)

    # GT stats for the report
    Sim_GT     = float(np.mean(gt_sim_scores))   if gt_sim_scores else 0.0
    BA_rec_GT  = float(np.mean(gt_beat_rec))     if gt_beat_rec   else 0.0
    BA_F1_GT   = float(np.mean(gt_beat_f1))      if gt_beat_f1    else 0.0
    BA_prec_GT = float(np.mean(gt_beat_prec))    if gt_beat_prec  else 0.0
    BA_w_GT    = float(np.mean(gt_beat_weighted)) if gt_beat_weighted else 0.0

    # --- Chunked / unchunked split ---
    def _split_by_mask(values, mask):
        """Split a list into two lists based on a boolean mask."""
        t_list, f_list = [], []
        for v, m in zip(values, mask):
            (t_list if m else f_list).append(v)
        return t_list, f_list

    n_chunked   = int(gt_is_arr.sum())
    n_unchunked = len(gt_is_arr) - n_chunked

    # Split exchange accumulators per condition
    def _split_exch(scores_dict, chunked_dict):
        ch, unch = {}, {}
        for c in cond_names:
            ch[c], unch[c] = _split_by_mask(scores_dict[c], chunked_dict[c])
        return ch, unch

    exch_sim_ch, exch_sim_unch     = _split_exch(exch_sim_scores, exch_is_chunked)
    exch_rec_ch, exch_rec_unch     = _split_exch(exch_beat_rec,   exch_is_chunked)
    exch_f1_ch, exch_f1_unch       = _split_exch(exch_beat_f1,    exch_is_chunked)
    exch_prec_ch, exch_prec_unch   = _split_exch(exch_beat_prec,  exch_is_chunked)
    exch_w_ch, exch_w_unch         = _split_exch(exch_beat_weighted, exch_is_chunked)

    # Use OVERALL GT anchor for both groups (same normalization baseline)
    # so chunked vs unchunked are compared on the same scale.
    if n_chunked > 0:
        cond_ch, avg_text_ch, avg_beat_ch, avg_eds_ch, avg_beat_w_ch, avg_eds_w_ch = _aggregate_eds(
            gt_sim_scores, gt_beat_rec,
            exch_sim_ch, exch_rec_ch, exch_f1_ch, exch_prec_ch,
            cond_names,
            gt_rec_w=gt_beat_weighted, exch_rec_w_dict=exch_w_ch)
    else:
        cond_ch = {c: None for c in cond_names}
        avg_text_ch = avg_beat_ch = avg_beat_w_ch = avg_eds_ch = avg_eds_w_ch = 0.0

    if n_unchunked > 0:
        cond_unch, avg_text_unch, avg_beat_unch, avg_eds_unch, avg_beat_w_unch, avg_eds_w_unch = _aggregate_eds(
            gt_sim_scores, gt_beat_rec,
            exch_sim_unch, exch_rec_unch, exch_f1_unch, exch_prec_unch,
            cond_names,
            gt_rec_w=gt_beat_weighted, exch_rec_w_dict=exch_w_unch)
    else:
        cond_unch = {c: None for c in cond_names}
        avg_text_unch = avg_beat_unch = avg_beat_w_unch = avg_eds_unch = avg_eds_w_unch = 0.0

    # ── 7  Report ─────────────────────────────────────────────
    active_conds = [c for c in cond_names if cond_results[c] is not None]

    results = {
        'mode':           'text_only' if text_only else 'exchange',
        'num_samples':    processed,
        'num_skipped':    skipped,
        'num_chunked':    n_chunked,
        'num_unchunked':  n_unchunked,
        'conditions':     {k: list(v) for k, v in EXCHANGE_CONDITIONS.items()},
        # GT (anchor)
        'TMR_sim_GT':     Sim_GT,
        'BA_recall_GT':   BA_rec_GT,
        'BA_F1_GT':       BA_F1_GT,
        'BA_prec_GT':     BA_prec_GT,
        'BA_weighted_GT': BA_w_GT,
    }
    for c in cond_names:
        if cond_results[c] is not None:
            results[c] = cond_results[c]
    results['avg_S_text']   = avg_S_text
    results['avg_S_beat']   = avg_S_beat
    results['avg_S_beat_w'] = avg_S_beat_w
    results['avg_EDS']      = avg_EDS
    results['avg_EDS_w']    = avg_EDS_w
    # Chunked / unchunked breakdown
    results['chunked_avg_EDS']   = avg_eds_ch
    results['unchunked_avg_EDS'] = avg_eds_unch
    results['chunked']   = {c: cond_ch[c]   for c in cond_names if cond_ch[c]   is not None}
    results['unchunked'] = {c: cond_unch[c]  for c in cond_names if cond_unch[c] is not None}

    mode_title = 'TEXT-ONLY (zeroed audio)' if text_only else 'Exchange'
    hline = '=' * 68
    print(f'\n{hline}')
    print(f'        EDS — {mode_title} — 3 Conditions')
    print(hline)
    print(f'  Mode                   {mode_title}')
    print(f'  Samples processed      {processed}')
    print(f'  Samples skipped        {skipped}')
    print(f'  Chunked / unchunked    {n_chunked} / {n_unchunked}')
    print(f'  --- Ground Truth (anchor) ---')
    print(f'  BA recall  GT          {BA_rec_GT:.4f}  ({len(gt_beat_rec)} samples)')
    print(f'  BA F1      GT          {BA_F1_GT:.4f}')
    print(f'  BA prec    GT          {BA_prec_GT:.4f}')
    print(f'  BA weighted GT         {BA_w_GT:.4f}')
    print(f'  TMR sim    GT          {Sim_GT:.4f}')
    for c in cond_names:
        lo, hi = EXCHANGE_CONDITIONS[c]
        range_str = f'{lo}-{hi} BPM' if hi else f'{lo}+ BPM'
        cr = cond_results[c]
        if cr is None:
            print(f'  --- [{c}] ({range_str}) — no pairs ---')
            continue
        print(f'  --- [{c}] ({range_str}) — {cr["num_pairs"]} pairs ---')
        print(f'  BA recall  {mode_label:9s}  {cr["BA_recall_exchange"]:.4f}')
        print(f'  BA F1      {mode_label:9s}  {cr["BA_F1_exchange"]:.4f}')
        print(f'  BA wt      {mode_label:9s}  {cr["BA_weighted_exchange"]:.4f}')
        print(f'  TMR sim    {mode_label:9s}  {cr["TMR_sim_exchange"]:.4f}')
        print(f'  S_beat                 {cr["S_beat"]:.4f}')
        print(f'  S_beat_w               {cr["S_beat_weighted"]:.4f}')
        print(f'  S_text                 {cr["S_text"]:.4f}')
        print(f'  EDS                    {cr["EDS"]:.4f}')
        print(f'  EDS_w                  {cr["EDS_weighted"]:.4f}')
    print(f'  --- Total Average (across {len(active_conds)} conditions) ---')
    print(f'  avg S_beat             {avg_S_beat:.4f}')
    print(f'  avg S_beat_w           {avg_S_beat_w:.4f}')
    print(f'  avg S_text             {avg_S_text:.4f}')
    print(f'  avg EDS                {avg_EDS:.4f}')
    print(f'  avg EDS_w              {avg_EDS_w:.4f}')

    # Chunked vs unchunked breakdown (using shared overall GT anchor)
    # Also report raw BA per group for diagnostics
    gt_rec_ch, gt_rec_unch = _split_by_mask(gt_beat_rec, gt_is_arr)
    raw_ba_gt_ch   = float(np.mean(gt_rec_ch))   if gt_rec_ch   else 0.0
    raw_ba_gt_unch = float(np.mean(gt_rec_unch))  if gt_rec_unch else 0.0

    print(f'  --- Chunked samples ({n_chunked}, raw BA_GT={raw_ba_gt_ch:.4f}) ---')
    if n_chunked > 0:
        for c in cond_names:
            cr = cond_ch[c]
            if cr is None:
                continue
            print(f'    [{c}] S_text={cr["S_text"]:.4f}  '
                  f'S_beat={cr["S_beat"]:.4f}  S_beat_w={cr["S_beat_weighted"]:.4f}  '
                  f'EDS={cr["EDS"]:.4f}  EDS_w={cr["EDS_weighted"]:.4f}  '
                  f'(raw BA={cr["BA_recall_exchange"]:.4f}, {cr["num_pairs"]} pairs)')
        print(f'    avg S_text={avg_text_ch:.4f}  '
              f'avg S_beat={avg_beat_ch:.4f}  avg S_beat_w={avg_beat_w_ch:.4f}  '
              f'avg EDS={avg_eds_ch:.4f}  avg EDS_w={avg_eds_w_ch:.4f}')
    else:
        print(f'    (none)')
    print(f'  --- Unchunked samples ({n_unchunked}, raw BA_GT={raw_ba_gt_unch:.4f}) ---')
    if n_unchunked > 0:
        for c in cond_names:
            cr = cond_unch[c]
            if cr is None:
                continue
            print(f'    [{c}] S_text={cr["S_text"]:.4f}  '
                  f'S_beat={cr["S_beat"]:.4f}  S_beat_w={cr["S_beat_weighted"]:.4f}  '
                  f'EDS={cr["EDS"]:.4f}  EDS_w={cr["EDS_weighted"]:.4f}  '
                  f'(raw BA={cr["BA_recall_exchange"]:.4f}, {cr["num_pairs"]} pairs)')
        print(f'    avg S_text={avg_text_unch:.4f}  '
              f'avg S_beat={avg_beat_unch:.4f}  avg S_beat_w={avg_beat_w_unch:.4f}  '
              f'avg EDS={avg_eds_unch:.4f}  avg EDS_w={avg_eds_w_unch:.4f}')
    else:
        print(f'    (none)')
    print(hline)

    out_path = os.path.join(results_dir, 'eds_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved  {out_path}')
    print(f'Pairs  {pairs_path}')

    if do_vis:
        print(f'Videos {vis_dir}  ({vis_count} rendered)')


if __name__ == '__main__':
    main()
