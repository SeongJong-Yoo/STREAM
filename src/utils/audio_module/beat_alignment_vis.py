import os
import sys
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.audio_module.utility import compute_motion_beat, compute_beat_alignment

def beat_visualization(audio_path):
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    hop_length = 512
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                            y_axis='mel', x_axis='time', hop_length=hop_length,
                            ax=ax[0])
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_env),
            label='Onset strength')
    ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',
            linestyle='--', label='Beats')
    ax[1].legend()
    plt.savefig(f'./images/beats_visualization.png')

def visualization(motion_beats, vel, audio_path, fps):
    hop_length = 512
    y, sr = librosa.load(audio_path)
    total_time = len(y) / sr
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    score = compute_beat_alignment(motion_beats*fps, times[beats]*fps)

    id = audio_path.split('/')[-1].split('.')[0]
        
    fig, ax = plt.subplots(nrows=3, sharex=True)
    fig.suptitle(f'Beat Alignment Score: {score:.3f}', fontsize=12)
    times_vel = np.linspace(0, total_time, len(vel))
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, center=False)
    times_spec = np.linspace(0, len(y)/sr, M.shape[1])
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max), 
                                y_axis='mel', x_axis='time',hop_length=hop_length, ax=ax[0], x_coords=times_spec)
    
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Music Beats')
    ax[1].vlines(motion_beats, 0, 1, alpha=0.5, color='g', linestyle='--', label='Motion Beats')
    ax[1].legend()
    ax[2].plot(times_vel, vel)
    ax[2].set(title='Motion Velocity')
    ax[2].vlines(times[beats], np.min(vel), np.max(vel), alpha=0.5, color='r', linestyle='--', label='Music Beats')
    ax[2].vlines(motion_beats, np.min(vel), np.max(vel), alpha=0.5, color='g', linestyle='--', label='Motion Beats')
    ax[2].legend()
    plt.show()
    plt.savefig(f'./images/beats_alignment_{id}.png')


if __name__ == '__main__':
    audio_path = './data/motorica_beats/sliced_audio'
    # audio_path = '/mnt/hdd/Dataset/AIST/wavs'
    audio_lists = Path(audio_path).glob('*.wav')

    audio_lists = list(audio_lists)  # Convert generator to list
    random_audio = np.random.choice(audio_lists)
    audio_path = str(random_audio)
    beat_visualization(audio_path)
    id = audio_path.split('/')[-1].split('.')[0]
    kp = np.load(f'./data/motorica_beats/sliced_motion/{id}_motion.npy', allow_pickle=True)[()]
    fps = kp['current_fps']
    kp = kp['motion']['motion_positions']
    motion_beats, vel = compute_motion_beat(kp, fps=fps)
    visualization(motion_beats, vel, audio_path, fps=fps)



