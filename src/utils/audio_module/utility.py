import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
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
import matplotlib.pyplot as plt

def compute_beats(audio=None, audio_path=None, visualize=False, hop_length=512):
    if audio:
        y, sr = audio
    elif audio_path:
        y, sr = librosa.load(audio_path)
    else:
        raise ValueError("Either audio or audio_path must be provided")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    # tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    if visualize:
        import matplotlib.pyplot as plt
        
        hop_length = 512
        fig, ax = plt.subplots(nrows=2, sharex=True)
        times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
        M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, center=False)
        times_spec = np.linspace(0, len(y)/sr, M.shape[1])
        librosa.display.specshow(librosa.power_to_db(M, ref=np.max), 
                                 y_axis='mel', x_axis='time',hop_length=hop_length, ax=ax[0], x_coords=times_spec)
        
        ax[0].label_outer()
        ax[0].set(title='Mel spectrogram')
        ax[1].plot(times, librosa.util.normalize(onset_env), label='Onset strength')
        ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
        ax[1].legend()
        plt.show()
        plt.savefig('./images/beats.png')

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)

    return tempo, beat_times

def compute_motion_beat(kp, vis=False, fps=60):
    vel = np.mean(np.linalg.norm(kp[1:] - kp[:-1], axis=-1, ord=2), axis=-1)

    vel = gaussian_filter(vel, sigma=2)
    motion_beat = argrelextrema(vel, np.less, order=10)[0]
    
    if vis:
        # Create visualization of velocity and local minima
        plt.figure(figsize=(12, 4))
        plt.plot(vel, label='Velocity')
        plt.vlines(motion_beat, ymin=np.min(vel), ymax=np.max(vel), 
                   colors='r', linestyles='--', label='Local Minima')
        plt.title('Motion Velocity and Detected Beats')
        plt.xlabel('Frame')
        plt.ylabel('Velocity')
        plt.legend()
        plt.show()
        plt.savefig(f'./images/motion_velocity_{id}.png')
    
    return motion_beat/fps, vel
    
def compute_beat_alignment(motion_beats, music_beats, fps):
    ba = 0
    # for motion_beat in motion_beats:
    #     min_value = np.min(np.abs(motion_beat - music_beats))
    #     ba += np.exp(-min_value / 2 / 9)
    # return ba / len(motion_beats)

    if len(motion_beats) == 0:
        return 0
    for music_beat in music_beats:
        min_value = np.min((motion_beats * fps - music_beat * fps)**2)   # 1000 changes seconds to milliseconds
        ba += np.exp(-min_value / 2 / 9)
    if len(music_beats) == 0:
        return 0
    return ba / len(music_beats)


def compute_beat_alignment_full(motion_beats, music_beats, fps):
    """Compute Beat Alignment Recall, Precision, and F1.

    Returns (recall, precision, f1).
    """
    if len(motion_beats) == 0 or len(music_beats) == 0:
        return 0, 0, 0

    # Recall (standard BAS)
    recall_sum = 0
    for music_beat in music_beats:
        min_value = np.min((motion_beats * fps - music_beat * fps) ** 2)
        recall_sum += np.exp(-min_value / 2 / 9)
    recall = recall_sum / len(music_beats)

    # Precision (penalty for jitter)
    precision_sum = 0
    for motion_beat in motion_beats:
        min_value = np.min((music_beats * fps - motion_beat * fps) ** 2)
        precision_sum += np.exp(-min_value / 2 / 9)
    precision = precision_sum / len(motion_beats)

    # F1
    if precision + recall == 0:
        return 0, 0, 0
    f1 = 2 * (precision * recall) / (precision + recall)

    return recall, precision, f1