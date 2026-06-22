import numpy as np
import pickle

from tqdm  import tqdm
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
# kinetic, manual
import torch
import os, sys
import argparse
# from render import ax_to_6v
sys.path.append(os.getcwd())
from src.skeleton.smpl_fk import SMPLModel
# from src.utils.audio_module.utility import compute_motion_beat, compute_beat_alignment
from src.metric.beat_align_score import calc_db, BA, get_music_beat_fromwav
from scipy.ndimage import uniform_filter1d
from src.skeleton.preprocessing import motion_preprocessing

TEST_LIST = {
    'AIST': [
        "mBR0"
        "mLH4",
        "mKR2",
        "mBR0",
        "mLO2",
        "mJB5",
        "mWA0",
        "mJS3",
        "mMH3",
        "mHO5",
        "mPO1",
    ],
    'motorica': [
        "kthjazz_gCH_sFM_cAll_d02_mCH_ch01_whitemanpaulandhisorchestraloisiana_006",
        "kthjazz_gJZ_sFM_cAll_d02_mJZ_ch01_bennygoodmansugarfootstomp_003",
        "kthjazz_gTP_sFM_sngl_d02_015",
        "kthmisc_gCA_sFM_cAll_d01_mCA_ch24",
        "kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001",
        "kthstreet_gLH_sFM_cAll_d01_mLH_ch01_thisisit_001",
        "kthstreet_gLH_sFM_cAll_d02_mLH_ch01_lala_001",
        "kthstreet_gLO_sFM_cAll_d02_mLO_ch01_arethafranklinrocksteady_002",
        "kthstreet_gPO_sFM_cAll_d01_mPO_ch01_bombom_002",
        "kthstreet_gPO_sFM_cAll_d02_mPO_ch01_bombom_001"
    ]
}

def calculate_beat_alignment(data_path, dataset=None):
    files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    beat_alignment_gt = []
    fk = SMPLModel()
    for file in tqdm(files):
        if not select_key_id(file.split('_motion')[0], dataset) in TEST_LIST[dataset]:
            continue
        data = np.load(os.path.join(data_path, file), allow_pickle=True)[()]['motion']
        data_name = file.split('_motion')[0]

        wav_path = os.path.join('./data/', dataset, 'sliced_audio', data_name + '.wav')
        if not os.path.exists(wav_path):
            print(f"Warning: {wav_path} not found")
            continue

        motion_joint_gt = data['motion_positions']
        # motion = data['motion_data']
        # motion, _ = motion_preprocessing(motion, data_type='tree', method='face_forward', dataset=dataset, fk=fk)
        # motion_joint_gt = fk(motion)
        motion_joint_gt = motion_joint_gt.reshape(motion_joint_gt.shape[0], 24*3)
        roott = motion_joint_gt[:1, :3]
        motion_joint_gt = motion_joint_gt - np.tile(roott, (1, 24))
        motion_joint_gt = motion_joint_gt.reshape(-1, 24, 3)

        music_beats = get_music_beat_fromwav(wav_path, motion_joint_gt.shape[0])
        dance_beats, length = calc_db(motion_joint_gt)
        beat_alignment_gt.append(BA(music_beats, dance_beats))

    print('beat alignment: ', np.mean(beat_alignment_gt))
    # results = {'beat_alignment_recon': np.mean(beat_alignment_recon), 'beat_alignment_gt': np.mean(beat_alignment_gt)}
    # return results


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

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)



def select_key_id(key, name):
    if name.lower() == 'aist':
        return key.split('_')[4]
    elif name.lower() == 'motorica':
        return key.split('_chunk')[0]
    elif name.lower() == 'humanml3d':
        return key.split('_chunk')[0]
    else:
        raise ValueError(f"Invalid dataset name: {name}")

def calc_and_save_feats(root, dataset):
    if not os.path.exists(os.path.join(root, 'gt_features')):
        os.mkdir(os.path.join(root, 'gt_features'))
    if not os.path.exists(os.path.join(root, 'neutralized_features')):
        os.mkdir(os.path.join(root, 'neutralized_features'))
    if not os.path.exists(os.path.join(root, 'gt_features', 'kinetic_features')):
        os.mkdir(os.path.join(root, 'gt_features', 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'gt_features', 'manual_features_new')):
        os.mkdir(os.path.join(root, 'gt_features', 'manual_features_new'))
    if not os.path.exists(os.path.join(root, 'neutralized_features', 'kinetic_features')):
        os.mkdir(os.path.join(root, 'neutralized_features', 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'neutralized_features', 'manual_features_new')):
        os.mkdir(os.path.join(root, 'neutralized_features', 'manual_features_new'))
    
    folder = os.path.join(root, 'sliced_motion_neutralized_smpl')
    gt_folder = os.path.join(root, 'sliced_motion_smpl')
    for file in tqdm(os.listdir(folder)):
        id = file.split('_motion')[0]
        if not select_key_id(id, dataset) in TEST_LIST[dataset]:
            continue
        
        neutral_data = np.load(os.path.join(folder, file), allow_pickle=True)[()]['motion']
        gt_data = np.load(os.path.join(gt_folder, file), allow_pickle=True)[()]['motion']
        joint3d_neutral = neutral_data['motion_positions']
        joint3d_gt = gt_data['motion_positions']

        calculate_and_save(joint3d_neutral, id, root=os.path.join(root, 'neutralized_features'))
        calculate_and_save(joint3d_gt, id, root=os.path.join(root, 'gt_features'))


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

   
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    
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

    return metrics

if __name__ == '__main__':
    DATASET = 'AIST'
    data_path = './data/' + DATASET # + '/sliced_motion_smpl'

    smplx_model = SMPLModel()
    # calc_and_save_feats(gt_root)
    # calc_and_save_feats(data_path, dataset=DATASET)
    
    print('Calculating metrics')
    neutral_data_root = os.path.join(data_path, 'neutralized_features')
    gt_data_root = os.path.join(data_path, 'gt_features')
    # print(quantized_metrics(neutral_data_root, gt_data_root, root_path=data_path))
    # print("GT beat alignment: ", calculate_beat_alignment(os.path.join(data_path, 'sliced_motion_smpl'), dataset=DATASET))
    print("Neutralized beat alignment: ", calculate_beat_alignment(os.path.join(data_path, 'sliced_motion_neutralized_smpl'), dataset=DATASET))
