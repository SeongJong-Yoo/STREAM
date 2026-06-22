import numpy as np
import os, sys
sys.path.append(os.getcwd())
import json, torch, sys
# kinetic, manual
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt 
from tqdm import tqdm
import scipy.signal
# Monkey-patch for librosa compatibility with newer scipy
if not hasattr(scipy.signal, 'hann'):
    try:
        from scipy.signal.windows import hann
        scipy.signal.hann = hann
    except ImportError:
        # Fallback to numpy if scipy windows not available
        import numpy as np
        scipy.signal.hann = lambda n: np.hanning(n)

import librosa

def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        #print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]

        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
    
        return beat_axis
    
def get_music_beat_fromwav(fpath, length):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    # EPS = 1e-6
    data, _ = librosa.load(fpath, sr=SR)[:length]
    # print("loaded music data shape", data.shape)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    start_bpm = librosa.beat.tempo(y=data)[0]
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
        trim=False,
    )
    return beat_idxs


def get_music_beat_from_musicfea35(fpath, length):


    data = np.load(fpath)[:length]
    beat_idxs = data[-1]

    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]
  
    return beat_idxs



def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))


def BA_full(music_beats, motion_beats, sigma_sq=9):
    """Compute Beat Alignment Recall, Precision, and F1.

    Recall (standard BAS): for each music beat, find closest kinematic beat.
    Precision: for each kinematic beat, find closest music beat.
    F1: harmonic mean of precision and recall.
    """
    if len(music_beats) == 0 or len(motion_beats[0]) == 0:
        return 0.0, 0.0, 0.0

    kinematic_beats = motion_beats[0]

    # Recall (standard BAS)
    recall_sum = 0
    for mb in music_beats:
        dist = np.min((kinematic_beats - mb) ** 2)
        recall_sum += np.exp(-dist / (2 * sigma_sq))
    recall = recall_sum / len(music_beats)

    # Precision (penalty for jitter)
    precision_sum = 0
    for kb in kinematic_beats:
        dist = np.min((music_beats - kb) ** 2)
        precision_sum += np.exp(-dist / (2 * sigma_sq))
    precision = precision_sum / len(kinematic_beats)

    # F1
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1, precision, recall

def calc_ba_score(motionroot, musicroot):
    # gt_list = []
    ba_scores = []

    for pkl in tqdm(os.listdir(motionroot)):
        data = np.load(os.path.join(motionroot, pkl), allow_pickle=True)[()]
        joint3d = data['motion_joint_gt']
        
        joint3d = joint3d.reshape(joint3d.shape[0], 24*3)
        roott = joint3d[:1, :3]
        joint3d = joint3d - np.tile(roott, (1, 24)) 
        joint3d = joint3d.reshape(-1, 24, 3)

        # joint3d = np.load(os.path.join(motionroot, pkl), allow_pickle=True).item()['pred_position'][:, :]
        dance_beats, length = calc_db(joint3d, pkl)        
        # music_beats = get_mb(pkl.split('.')[0] + '.json', length)
        music_beats = get_music_beat_fromwav(os.path.join(musicroot, pkl.split('.')[0] + '.wav'), joint3d.shape[0])

        ba_scores.append(BA(music_beats, dance_beats))
        
    return np.mean(ba_scores)

if __name__ == '__main__':
    music_root = "./data/AIST/sliced_audio"
    pred_root = './results/gt_test/AIST/motions'

    print(calc_ba_score(pred_root, music_root))
  