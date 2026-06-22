import numpy as np
import pickle
from copy import deepcopy

from tqdm  import tqdm

from scipy import linalg
# kinetic, manual
import torch
import os, sys
import argparse
# from render import ax_to_6v
sys.path.append(os.getcwd())
from src.skeleton.smpl_fk import SMPLModel
# from src.utils.audio_module.utility import compute_motion_beat, compute_beat_alignment
from src.metric.beat_align_score import calc_db, BA, BA_full, get_music_beat_fromwav
from src.visualization.skeleton import create_video_from_keypoints
from src.metric.features.kinetic import extract_kinetic_features
from src.metric.features.manual_new import extract_manual_features
from scipy.ndimage import uniform_filter1d

FOOT_IDX = {
    'motorica': [15, 16],
    'smpl': [7, 8, 10, 11]
}

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=False, default='motorica')

    args = parser.parse_args()
    return args

def calculate_skating_ratio(motions):
    thresh_height = 0.05  # 10
    fps = 30.0
    thresh_vel = 0.50  # 20 cm /s
    avg_window = 5  # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :]  # [bs, 2, 3, max_len]

    foot_min = np.stack([np.min(verts_feet[i, :, 1, :]) for i in range(verts_feet.shape[0])])
    while foot_min.ndim != verts_feet.ndim:
        foot_min = foot_min[..., np.newaxis]
    
    verts_feet = verts_feet - foot_min
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],
                                          axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in adjacent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height),
                                  (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :])  # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio, skate_vel

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


def normalize_one(feat):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10)

def quantized_metrics(predicted_pkl_root, gt_pkl_root, root_path=None):
    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []

    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'manual_features_new'))]
    
    gt_freatures_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'manual_features_new'))]
    
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m) # 
    if gt_freatures_k.shape[1] == 72:
        gt_freatures_k = gt_freatures_k[:,:66]
    if pred_features_k.shape[1] == 72:
        pred_features_k = pred_features_k[:,:66]

# T x 24 x 3 --> 72
# T x72 -->32 
    # print(gt_freatures_k.mean(axis=0))
    # print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    # print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    # print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    # print(pred_features_m.std(axis=0))

    # gt_freatures_k = normalize_one(gt_freatures_k)
    # gt_freatures_m = normalize_one(gt_freatures_m) 
    # pred_features_k = normalize_one(pred_features_k)
    # pred_features_m = normalize_one(pred_features_m)     
    
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    # # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)
    
    # print(gt_freatures_k.mean(axis=0))
    print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    print(pred_features_m.std(axis=0))

    
    # print(gt_freatures_k)
    # print(gt_freatures_m)

    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)
    # div_k_gt = '***'
    # div_m_gt = '***'
    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_m': fid_m, 'div_k': div_k, 'div_m' : div_m, 'div_k_gt': div_k_gt, 'div_m_gt': div_m_gt}

    if root_path is not None:
        with open(os.path.join(root_path, 'metrics.txt'), 'a') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
    return metrics


def calc_fid(kps_gen, kps_gt):

    print(kps_gen.shape)
    print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)
    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)
    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calculate_and_save(joint3d, id, root):
    joint3d = joint3d[:1024,:22,:]
    assert len(joint3d.shape) == 3
    if isinstance(joint3d, torch.Tensor):
        joint3d = joint3d.detach().cpu().numpy()
    joint3d = joint3d.reshape(joint3d.shape[0], 22*3)
    
    roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
    joint3d = joint3d - np.tile(roott, (1, 22))  # Calculate relative offset with respect to root

    # relative
    joint3d_relative = joint3d.copy()
    joint3d_relative = joint3d_relative.reshape(-1, 22, 3)
    joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]

    joint3d_relative = joint3d_relative.reshape(-1, 22, 3)
    kinetic_features = extract_kinetic_features(joint3d_relative)
    manual_features = extract_manual_features(joint3d_relative)
    
    np.save(os.path.join(root, 'kinetic_features', id+ '.npy'), kinetic_features)
    np.save(os.path.join(root, 'manual_features_new', id+ '.npy'), manual_features)

def change_axis(motion):
    y_value = deepcopy(motion[:, :, 1])
    motion[:, :, 1] = motion[:, :, 2]
    motion[:, :, 2] = y_value
    return motion

def calc_and_save_feats(root):
    if not os.path.exists(os.path.join(root, 'gt_features')):
        os.mkdir(os.path.join(root, 'gt_features'))
    if not os.path.exists(os.path.join(root, 'pred_features')):
        os.mkdir(os.path.join(root, 'pred_features'))
    if not os.path.exists(os.path.join(root, 'gt_features', 'kinetic_features')):
        os.mkdir(os.path.join(root, 'gt_features', 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'gt_features', 'manual_features_new')):
        os.mkdir(os.path.join(root, 'gt_features', 'manual_features_new'))
    if not os.path.exists(os.path.join(root, 'pred_features', 'kinetic_features')):
        os.mkdir(os.path.join(root, 'pred_features', 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'pred_features', 'manual_features_new')):
        os.mkdir(os.path.join(root, 'pred_features', 'manual_features_new'))
    
    folder = os.path.join(root, 'motions')
    for file in tqdm(os.listdir(folder)):
        id = file.split('.')[0]
        if file[-3:] == 'npy':
            data = np.load(os.path.join(folder, file), allow_pickle=True)[()]
            joint3d_recon = data['motion_joint_recon']
            joint3d_gt = data['motion_joint_gt']
            if 'sample_id' in data:
                id = data['sample_id']
            if 'id' in data:
                id = data['id'].split('/')[-1].split('.')[0]
        else:
            continue

        # joint3d_recon = change_axis(joint3d_recon)
        # joint3d_gt = change_axis(joint3d_gt)
        # real_gt = np.load(os.path.join('./data/motorica/sliced_motion_smpl', id + '_motion.npy'), allow_pickle=True)[()]['motion']['motion_positions']
        # create_video_from_keypoints(keypoints=real_gt, output_path='./test.mp4', link_type='smpl', flipped=True)#, gt=joint3d_gt, gt_link_type='smpl')
        calculate_and_save(joint3d_recon, id, root=os.path.join(root, 'pred_features'))
        calculate_and_save(joint3d_gt, id, root=os.path.join(root, 'gt_features'))

def calculate_beat_alignment(data_path, root_path=None, dataset=None):
    files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    beat_alignment_recon = []
    beat_alignment_gt = []
    beat_f1_recon = []
    beat_precision_recon = []
    beat_recall_recon = []
    beat_f1_gt = []
    beat_precision_gt = []
    beat_recall_gt = []
    for file in tqdm(files):
        data = np.load(os.path.join(data_path, file), allow_pickle=True)[()]
        if 'id' in data:
            id = data['id'].split('/')[-1].split('.')[0]
            if 'motion' in id:
                id = id.split('_motion')[0]
        elif 'sample_id' in data:
            id = data['sample_id']
        else:
            raise ValueError(f"Unknown id type: {data}")
        # if 'dataset' in data:
        #     data_name = data['dataset']
        # else:
        data_name = dataset

        wav_path = os.path.join('./data/', data_name, 'sliced_audio', id + '.wav')
        if not os.path.exists(wav_path):
            print(f"Warning: {wav_path} not found")
            continue

        motion_joint_recon = data['motion_joint_recon']
        motion_joint_gt = data['motion_joint_gt']

        motion_joint_recon = motion_joint_recon.reshape(motion_joint_recon.shape[0], 24*3)
        motion_joint_gt = motion_joint_gt.reshape(motion_joint_gt.shape[0], 24*3)
        roott = motion_joint_recon[:1, :3]
        motion_joint_recon = motion_joint_recon - np.tile(roott, (1, 24))
        motion_joint_gt = motion_joint_gt - np.tile(roott, (1, 24))
        motion_joint_recon = motion_joint_recon.reshape(-1, 24, 3)
        motion_joint_gt = motion_joint_gt.reshape(-1, 24, 3)

        music_beats = get_music_beat_fromwav(wav_path, motion_joint_recon.shape[0])
        dance_beats, length = calc_db(motion_joint_recon, id)
        if len(dance_beats[0])==0:
            continue
        beat_alignment_recon.append(BA(music_beats, dance_beats))
        f1, prec, rec = BA_full(music_beats, dance_beats)
        beat_f1_recon.append(f1)
        beat_precision_recon.append(prec)
        beat_recall_recon.append(rec)

        dance_beats, length = calc_db(motion_joint_gt, id)
        if len(dance_beats[0])==0:
            continue
        beat_alignment_gt.append(BA(music_beats, dance_beats))
        f1, prec, rec = BA_full(music_beats, dance_beats)
        beat_f1_gt.append(f1)
        beat_precision_gt.append(prec)
        beat_recall_gt.append(rec)

    results = {
        'beat_alignment_recon': np.mean(beat_alignment_recon),
        'beat_alignment_gt': np.mean(beat_alignment_gt),
        'beat_f1_recon': np.mean(beat_f1_recon) if beat_f1_recon else 0.0,
        'beat_f1_gt': np.mean(beat_f1_gt) if beat_f1_gt else 0.0,
        'beat_precision_recon': np.mean(beat_precision_recon) if beat_precision_recon else 0.0,
        'beat_precision_gt': np.mean(beat_precision_gt) if beat_precision_gt else 0.0,
        'beat_recall_recon': np.mean(beat_recall_recon) if beat_recall_recon else 0.0,
        'beat_recall_gt': np.mean(beat_recall_gt) if beat_recall_gt else 0.0,
    }
    if root_path is not None:
        with open(os.path.join(root_path, 'metrics.txt'), 'a') as f:
            for key, value in results.items():
                f.write(f'{key}: {value}\n')
    return results


def calculate_foot_skate_ratio(data_path, root_path=None):
    files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    motion_joint_recon = []
    motion_joint_gt = []
    for file in files:
        data = np.load(os.path.join(data_path, file), allow_pickle=True)[()]
        motion_joint_recon.append(data['motion_joint_recon'])
        motion_joint_gt.append(data['motion_joint_gt'])
    
    motion_joint_recon = np.stack(motion_joint_recon).transpose(0, 2, 3, 1)
    motion_joint_gt = np.stack(motion_joint_gt).transpose(0, 2, 3, 1)

    skating_ratio_recon, skate_vel_recon = calculate_skating_ratio(motion_joint_recon)
    skating_ratio_gt, skate_vel_gt = calculate_skating_ratio(motion_joint_gt)
    results = {'skating_ratio_recon': np.mean(skating_ratio_recon), 'skating_ratio_gt': np.mean(skating_ratio_gt)}
    if root_path is not None:
        with open(os.path.join(root_path, 'metrics.txt'), 'a') as f:
            for key, value in results.items():
                f.write(f'{key}: {value}\n')
    return results



TMR_MIN_SEGMENT_FRAMES = 10


def calculate_tmr_similarity(data_path, tmr_wrapper, dataset=None, root_path=None):
    """Compute segment-level TMR text-motion similarity for saved motion .npy files.

    Frame-level labels (T, 3) from ``sliced_labels`` can contain multiple text
    segments within a single 5-sec clip (e.g. technique changes mid-clip).
    This function splits each clip into contiguous segments by technique
    transitions and scores each segment independently.

    Parameters
    ----------
    data_path    : str – folder containing the .npy motion files
    tmr_wrapper  : TMRWrapper instance (from src.metric.tmr_wrapper)
    dataset      : str | None – dataset name override
    root_path    : str | None – if given, append results to metrics.txt

    Returns
    -------
    dict with 'tmr_sim_recon' and 'tmr_sim_gt' (mean per-segment scores)
    """
    files = sorted(f for f in os.listdir(data_path) if f.endswith('.npy'))
    sim_recon_list = []
    sim_gt_list = []

    for file in tqdm(files, desc='TMR similarity'):
        data = np.load(os.path.join(data_path, file), allow_pickle=True)[()]
        if 'id' in data:
            sample_id = data['id'].split('/')[-1].split('.')[0]
        elif 'sample_id' in data:
            sample_id = data['sample_id']
        else:
            continue
        data_name = data.get('dataset', dataset)
        if data_name is None:
            continue

        motion_recon = data['motion_joint_recon']  # (T, 24, 3)
        motion_gt = data['motion_joint_gt']         # (T, 24, 3)

        # Load frame-level label and extract text segments
        label = _load_label_for_tmr(sample_id, data_name)
        if label is None:
            continue

        segments = _get_text_segments_for_tmr(label)

        for text, start, end in segments:
            if (end - start) < TMR_MIN_SEGMENT_FRAMES:
                continue

            seg_gt = motion_gt[start:end]        # (seg_T, 24, 3)
            seg_recon = motion_recon[start:end]  # (seg_T, 24, 3)

            gt_t = torch.from_numpy(seg_gt).float().unsqueeze(0)       # (1, seg_T, 24, 3)
            recon_t = torch.from_numpy(seg_recon).float().unsqueeze(0)  # (1, seg_T, 24, 3)

            try:
                scores_gt = tmr_wrapper.paired_scores(gt_t, [text])
                scores_recon = tmr_wrapper.paired_scores(recon_t, [text])
                sim_gt_list.append(scores_gt[0])
                sim_recon_list.append(scores_recon[0])
            except Exception:
                continue

    results = {
        'tmr_sim_recon': float(np.mean(sim_recon_list)) if sim_recon_list else 0.0,
        'tmr_sim_gt':    float(np.mean(sim_gt_list))    if sim_gt_list    else 0.0,
    }
    if root_path is not None:
        with open(os.path.join(root_path, 'metrics.txt'), 'a') as f:
            for key, value in results.items():
                f.write(f'{key}: {value}\n')
    return results


def _load_label_for_tmr(sample_id, dataset_name):
    """Load frame-level label array (T, 3) from sliced_labels.

    Returns numpy array of shape (T, 3) with [genre, technique, description]
    per frame, or None if not found.
    """
    npy_path = os.path.join('./data', dataset_name, 'sliced_labels',
                            f'{sample_id}_label.npy')
    if os.path.exists(npy_path):
        raw = np.load(npy_path, allow_pickle=True)[()]
        if isinstance(raw, dict):
            label = raw['data']
        else:
            label = raw
        if isinstance(label, np.ndarray) and label.ndim == 2:
            return label
        return None

    # Fallback: .txt label → single-segment pseudo-label
    txt_path = os.path.join('./data', dataset_name, 'sliced_labels',
                            f'{sample_id}_label.txt')
    if os.path.exists(txt_path):
        from src.utils.utility import read_label_segment_txt
        d = read_label_segment_txt(txt_path)
        parts = [d.get('genre', ''), d.get('label', ''),
                 d.get('description', '')]
        text = ', '.join(p for p in parts if p) or f'dance motion {sample_id}'
        # Return 1-row array so _get_text_segments_for_tmr produces one segment
        return np.array([[text, text, '']])

    return None


def _get_text_segments_for_tmr(label):
    """Extract text segments from frame-level label (T, 3) array.

    Detects transitions in column 1 (technique) to find segment boundaries.

    Returns list of (text_str, start_frame, end_frame) tuples.
    """
    T = label.shape[0]
    segments = []
    start = 0
    current_key = label[0, 1]

    for i in range(1, T):
        if label[i, 1] != current_key:
            text = _build_text_for_tmr(label[start])
            segments.append((text, start, i))
            start = i
            current_key = label[i, 1]

    # Last segment
    text = _build_text_for_tmr(label[start])
    segments.append((text, start, T))
    return segments


def _build_text_for_tmr(label_row):
    """Build text string from a single label row [genre, technique, description]."""
    parts = [str(label_row[col]).strip() for col in range(len(label_row))
             if str(label_row[col]).strip()]
    return ', '.join(parts) if parts else 'motion'


if __name__ == '__main__':
    args = load_config()
    DATASET = args.dataset
    MODEL_NAME = args.exp_name
    # MODEL_NAME = 'EDGE'
    device = f"cuda:0"
    smplx_model = SMPLModel()

    # gt_root = os.path.join('./data', DATASET)
    gt_root = os.path.join('./results', MODEL_NAME, DATASET)
    pred_root = os.path.join('./results', MODEL_NAME, DATASET)
    print('Calculating and saving features')

    # # calc_and_save_feats(gt_root)
    calc_and_save_feats(pred_root)

    print('Calculating metrics')
    pred_data_root = os.path.join(pred_root, 'pred_features')
    gt_data_root = os.path.join(gt_root, 'gt_features')
    print(quantized_metrics(pred_data_root, gt_data_root, root_path=pred_root))
    print(calculate_beat_alignment(os.path.join(pred_root, 'motions'), root_path=pred_root, dataset=DATASET))
    print(calculate_foot_skate_ratio(os.path.join(pred_root, 'motions'), root_path=pred_root))

    # TMR_DIR = '/home/yoo/project/motion/ECCV_Comp/TMR'
    # TMR_RUN_DIR = '/home/yoo/project/motion/ECCV_Comp/TMR/outputs/finetune_motorica'
    # from src.metric.tmr_wrapper import TMRWrapper
    # tmr = TMRWrapper(TMR_DIR, TMR_RUN_DIR, device, ckpt_name='epoch-epoch=299')
    # print(calculate_tmr_similarity(os.path.join(pred_root, 'motions'), tmr, dataset=DATASET, root_path=pred_root))
    
