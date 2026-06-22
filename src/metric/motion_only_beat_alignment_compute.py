import numpy as np
import pickle
from copy import deepcopy
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
from src.visualization.skeleton import create_video_from_keypoints

def calculate_beat_alignment(motion, wav_path):
    """
    motion: (T, 24*3)
    """
    roott = motion[:1, :3]
    motion = motion - np.tile(roott, (1, 24))
    motion = motion.reshape(-1, 24, 3)

    music_beats = get_music_beat_fromwav(wav_path, motion.shape[0])
    dance_beats, length = calc_db(motion)
    BA(music_beats, dance_beats)

    print('beat alignment: ', BA(music_beats, dance_beats))

def change_y_axis(motion):
    y_value = deepcopy(motion[:, :, 1])
    motion[:, :, 1] = motion[:, :, 2]
    motion[:, :, 2] = y_value
    return motion

data_path = './results/editable_results/example_2-2/recon_kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_chunk121.npy'
wav_path = './data/motorica/sliced_audio/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_chunk121.wav'

data = np.load(data_path, allow_pickle=True)[()]
recon = change_y_axis(data['recon_motion_joint']).reshape(-1, 24*3)
gt = change_y_axis(data['edited_motion_joint']).reshape(-1, 24*3)


calculate_beat_alignment(recon, wav_path)
calculate_beat_alignment(gt, wav_path)

roott = recon[:1, :3]
recon = recon - np.tile(roott, (1, 24))
recon = recon.reshape(-1, 24, 3)

roott = gt[:1, :3]
gt = gt - np.tile(roott, (1, 24))
gt = gt.reshape(-1, 24, 3)

create_video_from_keypoints(keypoints=recon,
                            output_path = './test.mp4',
                            link_type='smpl',
                            gt_link_type='smpl',
                            gt=gt,
                            audio_path=wav_path)