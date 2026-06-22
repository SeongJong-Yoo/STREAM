import librosa
import numpy as np
from pathlib import Path
import os
import sys
from tqdm import tqdm
import pickle

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.dataloader.utility import slice_datas, slice_data

def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def process_audio_file(audio_name, FPS = 50, HOP_LENGTH = 512):
    SR = FPS * HOP_LENGTH
    waveform, _ = librosa.load(audio_name, mono=True, sr=SR)

    # tempo, beats_dense = librosa.beat.beat_track(y=waveform, sr=detected_sr)
    tempogram = librosa.feature.tempogram(y=waveform, sr=SR).T

    envelope = librosa.onset.onset_strength(y=waveform, sr=SR)
    # peak_idxs = librosa.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    # peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    # peak_onehot[peak_idxs] = 1.0

    chroma_cens = librosa.feature.chroma_cens(y=waveform, sr=SR, n_chroma=12).T

    mfcc = librosa.feature.mfcc(y=waveform, sr=SR, n_mfcc=20).T
    # rms = librosa.feature.rms(y=waveform)

    f1 = envelope[:, None]
    f2 = mfcc
    f3 = chroma_cens
    f4 = tempogram
    # NORMALIZE Features that are not between -1.0 and 1.0 already
    # f1_norm = librosa.util.normalize(f1)
    # f2 = librosa.util.normalize(f2)
    # f5 = librosa.util.normalize(f5)

    len = min(f1.shape[0], f2.shape[0], f3.shape[0], f4.shape[0])


    audio_feature = np.concatenate([f1[:len,:], f2[:len,:], f3[:len,:], f4[:len,:]], axis=-1)

    return audio_feature

def match_audio_feature(audio_path, FPS, HP, ref_len):
    init_FPS = FPS
    if FPS > 50:
        gap = 0.01
    else:
        gap = 0.005
    visit = [False, False]
    audio_feature = process_audio_file(audio_name=audio_path, FPS=FPS, HOP_LENGTH=HP)
    if ref_len < len(audio_feature):
        FPS = FPS - gap
        while True:
            audio_feature = process_audio_file(audio_name=audio_path, FPS=FPS, HOP_LENGTH=HP)

            if len(audio_feature) == ref_len:
                return audio_feature, FPS
            elif ref_len > len(audio_feature):
                visit[0] = True
                if visit[0] and visit[1]:
                    gap = gap / 2
                FPS = FPS + gap
            else:
                visit[1] = True
                if visit[0] and visit[1]:
                    gap = gap / 2
                FPS = FPS - gap
            if init_FPS - FPS > 3:
                print("File: ", audio_path, ", Adjusted FPS: ", FPS)
                exit()

    elif ref_len > len(audio_feature):
        FPS = FPS + gap
        while True:
            audio_feature = process_audio_file(audio_name=audio_path, FPS=FPS, HOP_LENGTH=HP)

            if len(audio_feature) == ref_len:
                return audio_feature, FPS
            elif ref_len > len(audio_feature):
                visit[0] = True
                if visit[0] and visit[1]:
                    gap = gap / 2
                FPS = FPS + gap
            else:
                visit[1] = True
                if visit[0] and visit[1]:
                    gap = gap / 2
                FPS = FPS - gap
            if init_FPS - FPS > 3:
                print("File: ", audio_path, ", Adjusted FPS: ", FPS)
                exit()
    else:
        if init_FPS - FPS > 3:
            print("File: ", audio_path, ", Adjusted FPS: ", FPS)
        return audio_feature, FPS

# def batch_audio_feature_compute(audios, target_fps=60, hop_length=512, name='audio'):
#     audio_features = []
#     for i in range(len(audios)):
#         audio_feature, FPS = match_audio_feature(audios[i], target_fps, hop_length, name=f"{name}_{i}")
#         audio_features.append(audio_feature)
#     return audio_features

# if __name__ == "__main__":
#     FPS = 60
#     HP = 512
#     LENGTH = 5 # in seconds
#     audio_path = './data/AIST/sliced_audio'
#     output_dir = './data/AIST/sliced_audio_features35'
#     os.makedirs(output_dir, exist_ok=True)
#     wavs = Path(audio_path).glob("*.wav")
#     wavs = sorted([wav.stem for wav in wavs])

#     for wav in tqdm(wavs):
#         wav_path = os.path.join(audio_path, f"{wav}.wav")
#         audio_feature, FPS = match_audio_feature(audio_path=wav_path, FPS=FPS, HP=HP, ref_len=int(FPS * LENGTH))
#         np.save(os.path.join(output_dir, f"{wav}_audio.npy"), audio_feature)

if __name__ == "__main__":
    FPS = 100
    WINDOW_SIZE = 5
    HOP_LENGTH = 1
    output_dir = './data/AIST'
    wav_dir = '/mnt/hdd/Dataset/AIST/wavs'
    kp_dir = '/mnt/hdd/Dataset/AIST/keypoints3d'
    os.makedirs(os.path.join(output_dir, 'sliced_audio_features417'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sliced_motion'), exist_ok=True)

    wavs = Path(wav_dir).glob("*.wav")
    wavs = sorted([wav.stem for wav in wavs])

    kps = Path(kp_dir).glob("*")
    kps = sorted(list(kps))

    for name in tqdm(wavs):
        kp_data = load_pickle_data(os.path.join(kp_dir, f"{name}.pkl"))['keypoints3d_optim']
        ref_len = int(len(kp_data) * (FPS / 60))
        wav_path = os.path.join(wav_dir, f"{name}.wav")
        ref_data, sr = librosa.load(wav_path, sr=44100)
        audio_feature, _ = match_audio_feature(audio_path=wav_path, FPS=FPS, HP=512, ref_len=ref_len)
        # audio_feature = process_audio_file(wav_path, FPS=FPS, HOP_LENGTH=512)

        datas = [{'data':ref_data, 'FPS':sr}, {'data':audio_feature, 'FPS':FPS}]
        _, audio_features = slice_datas(datas, WINDOW_SIZE, HOP_LENGTH, mode='overlap')

        if audio_feature.shape[-1] != 417:
            print(f"audio_feature.shape[-1] != 417: {audio_feature.shape[-1]}")
            continue

        # datas2 = [{'data':ref_data, 'FPS':sr}, {'data':kp_data, 'FPS':60}]
        # _, kp_features = slice_datas(datas2, WINDOW_SIZE, HOP_LENGTH, mode='overlap', save2=True, save_path=os.path.join(output_dir, 'sliced_motion'), id=name)

        for i, feature in enumerate(audio_features):
            save_path = os.path.join(output_dir, 'sliced_audio_features417', f"{name}_chunk{i}_audio.npy")
            np.save(save_path, feature)
