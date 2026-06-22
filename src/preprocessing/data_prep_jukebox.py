import numpy as np
import pickle
from pathlib import Path

import jukemirlib
from tqdm import tqdm


def downsample(data, target_fps, src_fps):
    ratio = int(src_fps / target_fps)
    data = data[::ratio]
    return data

def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_jukebox_features(src, dest, ref):
    if not isinstance(src, Path):
        src = Path(src)
    if not isinstance(dest, Path):
        dest = Path(dest)

    if not dest.exists():
        raise FileNotFoundError(f"Destination path {dest} does not exist. Please create it first.")
    audio_files = list(src.glob("*.wav"))
    audio_files = sorted(audio_files)
    # jukemirlib.setup_models(cache_dir='~/.cache/jukemirlib', device='cuda')

    for audio_file in tqdm(audio_files):
        audio_name = audio_file.stem
        ref_name = audio_name + '_motion.npy'
        ref_file = ref / ref_name
        if not ref_file.exists():
            raise FileNotFoundError(f"Reference file {ref_file} does not exist. Please create it first.")
        
        save_path = dest / f"{audio_name}_audio.npy"
        if save_path.exists():
            continue
        ref_data = np.load(ref_file, allow_pickle=True)[()]
        motion_data = ref_data['motion']['motion_data']
        ref_len = motion_data.shape[0]
        current_fps = ref_data['current_fps']
        target_fps = ref_data['target_fps']
        
        audio = jukemirlib.load_audio(audio_file)
        reps = jukemirlib.extract(audio, layers=[66], downsample_target_rate=current_fps)[66][:ref_len]
        if reps.shape[0] != ref_len:
            print(f"Warning: {audio_name} has a different length than the reference file {ref_name}. Please check the data and FPS")
            break
        output = {'audio': reps, 'current_fps': current_fps, 'target_fps': target_fps}
        np.save(save_path, output)



if __name__ == "__main__":
    sliced_audio_path = Path('./data/motorica/sliced_audio')
    ref_path = Path('./data/motorica/sliced_motion_smpl')
    output_audio_path = Path('./data/motorica/sliced_audio_features')

    # sliced_audio_path = Path('./data/AIST/sliced_audio')
    # ref_path = Path('./data/AIST/sliced_motion_smpl')
    # output_audio_path = Path('./data/AIST/sliced_audio_features')
    
    # sliced_audio_path = Path('./data/test_data/ex2/sliced_audio')
    # ref_path = Path('./data/test_data/ex2/sliced_motion_smpl')
    # output_audio_path = Path('./data/test_data/ex2/sliced_audio_features')
    if not sliced_audio_path.exists():
        raise FileNotFoundError(f"Sliced audio path {sliced_audio_path} does not exist. Please create it first.")
    
    if not output_audio_path.exists():
        output_audio_path.mkdir(parents=True, exist_ok=True)
    extract_jukebox_features(sliced_audio_path, output_audio_path, ref_path)

