import librosa
import soundfile as sf
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.audio_module.utility import compute_beats
import numpy as np
from tqdm import tqdm
import pickle

def downsample(data, target_fps, src_fps):
    ratio = int(src_fps / target_fps)
    data = data[::ratio]
    return data

def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def combine_beats_to_audio_features(audio, features, file_name):
    output = {}
    tempo, beats = compute_beats(audio)
    output['beats'] = beats
    output['tempo'] = tempo
    output['jukebox'] = features

    np.save(file_name, output)

if __name__ == "__main__":
    # audio_path = '/mnt/hdd/Dataset/AIST/wavs/gJS_sBM_cAll_d01_mJS2_ch07.wav'
    # tempo, beats = compute_beats(audio_path=audio_path, visualize=True)
    # wav_dir = './data/AIST/sliced_audio'
    wav_dir = './data/motorica_beats/sliced_audio'
    output_dir = './data/motorica_beats'

    wavs = Path(wav_dir).glob("*")
    wavs = sorted(wavs)

    for wav in tqdm(wavs):
        id = wav.stem
        wav_path = os.path.join(wav_dir, f"{id}.wav")

        audio = librosa.load(wav_path, sr=44100)
        tempo, beats = compute_beats(audio)
        save_path = Path(output_dir) / 'sliced_beats'
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(f"{save_path}/{id}.npy", {'tempo':tempo, 'beats':beats})
            

