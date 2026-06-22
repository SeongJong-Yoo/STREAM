import os
import numpy as np
import soundfile as sf
from pathlib import Path
import pickle
from tqdm import tqdm

def chop_data(data, s_idx, e_idx):
    return data[s_idx:e_idx]

def downsample(data, target_fps, src_fps):
    ratio = int(src_fps / target_fps)
    data = data[::ratio]
    return data

def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(data, save_path, id='', save=True):
    if not save:
        return
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if isinstance(data, list):
        for i in range(len(data)):
            file_name = f"{save_path}/{id}_chunk{i}_motion.npy"
            np.save(file_name, data[i])
    else:
        file_name = f"{save_path}/{id}.npy"
        np.save(file_name, data)
    


def slice_datas(datas, window_size=5, stride=0.5, mode='overlap', save1=False, save2=False, save_path=None, id=None):
    data1 = datas[0]['data']
    data2 = datas[1]['data']
    FPS1 = datas[0]['FPS']
    FPS2 = datas[1]['FPS']

    if np.abs(data1.shape[0] / FPS1 - data2.shape[0] / FPS2) > 0.1:
        print(f"Warning: {id} has different length")
        return

    if len(data1.shape) == 1:
        data1 = data1.reshape(-1, 1)
    if len(data2.shape) == 1:
        data2 = data2.reshape(-1, 1)

    total_time = min(data1.shape[0] / FPS1, data2.shape[0] / FPS2)
    total_len = data1.shape[0]
    ws_len_ref = int(window_size * FPS1)
    s_len_ref = int(stride * FPS1)
    ws_len_second = int(window_size * FPS2)
    s_len_second = int(stride * FPS2)
    num_chunks = (total_len - ws_len_ref) // s_len_ref + 1

    sliced_data1 = []
    sliced_data2 = []
    for i in range(num_chunks):
        # e_time = s_time + window_size
        sliced_data1.append(chop_data(data1, i * s_len_ref, i * s_len_ref + ws_len_ref))
        sliced_data2.append(chop_data(data2, i * s_len_second, i * s_len_second + ws_len_second))

    if num_chunks * s_len_ref + ws_len_ref == total_len or mode == 'discard':
        # save_data(sliced_data1, save_path, id, save1)
        save_data(sliced_data2, save_path, id, save2)
        return sliced_data1, sliced_data2
    
    last_chunk1 = np.zeros((ws_len_ref, data1.shape[1]))
    last_chunk2 = np.zeros((ws_len_second, data2.shape[1]))
    s_time = num_chunks * stride
    if mode == 'pad':
        last_chunk1[:int((total_time - s_time) * FPS1)] = data1[int(s_time * FPS1):]
        last_chunk2[:int((total_time - s_time) * FPS2)] = data2[int(s_time * FPS2):]
    elif mode == 'none':
        last_chunk1 = data1[int(s_time * FPS1):]
        last_chunk2 = data2[int(s_time * FPS2):]
    elif mode == 'overlap':
        last_chunk1 = data1[int(total_time * FPS1) - ws_len_ref:int(total_time * FPS1)]
        last_chunk2 = data2[int(total_time * FPS2) - ws_len_second:int(total_time * FPS2)]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    sliced_data1.append(last_chunk1)
    sliced_data2.append(last_chunk2)

    # save_data(sliced_data1, save_path, id, save1)
    save_data(sliced_data2, save_path, id, save2)
    return sliced_data1, sliced_data2

output_dir = './data/AIST'
wav_dir = '/mnt/hdd/Dataset/AIST/wavs'
kp_dir = '/mnt/hdd/Dataset/AIST/keypoints3d'

wavs = Path(wav_dir).glob("*")
wavs = sorted([wav.stem for wav in wavs])
kps = Path(kp_dir).glob("*")
kps = sorted(list(kps))


for wav in tqdm(wavs):
    id = wav
    kp_path = os.path.join(kp_dir, f"{id}.pkl")
    kp_data = load_pickle_data(kp_path)['keypoints3d_optim']
    kp_data = downsample(kp_data, 30, 60)
    audio, sr = sf.read(os.path.join(wav_dir, f"{id}.wav"))
    datas = [{'data':audio, "FPS": sr}, {'data':kp_data, "FPS": 30}]
    audio_sliced, kp_sliced = slice_datas(datas, 5, 1, mode='overlap', save2=False, id=id)

    for i, audio in enumerate(audio_sliced):
        audio_path = Path(output_dir) / 'sliced_audio'
        audio_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{audio_path}/{id}_chunk{i}.wav"
        sf.write(file_name, audio, sr)

