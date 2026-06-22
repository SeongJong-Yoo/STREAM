import jukemirlib
from pathlib import Path
import numpy as np
from tqdm import tqdm



def extract_jukebox_features(src, dest):
    if not isinstance(src, Path):
        src = Path(src)
    if not isinstance(dest, Path):
        dest = Path(dest)

    if not dest.exists():
        raise FileNotFoundError(f"Destination path {dest} does not exist. Please create it first.")
    audio_files = list(src.glob("*.wav"))
    audio_files = sorted(audio_files)
    for audio_file in tqdm(audio_files, desc="Extracting Jukebox features"):
        audio_name = audio_file.stem
        save_path = dest / f"{audio_name}.npy"
        if save_path.exists():
            continue
        audio = jukemirlib.load_audio(audio_file)
        reps = jukemirlib.extract(audio, layers=[66], downsample_target_rate=30)
        np.save(save_path, reps)


if __name__ == "__main__":
    # sliced_audio_path = Path("/fs/nexus-projects/PhysicsFall/editable_dance_project/data/motorica_beats/sliced_audio")
    # output_audio_path = Path("/fs/nexus-projects/PhysicsFall/editable_dance_project/data/motorica_beats/jukebox_features")
    sliced_audio_path = Path('./data/AIST/sliced_audio')
    output_audio_path = Path('./data/AIST/sliced_audio_features')
    if not sliced_audio_path.exists():
        raise FileNotFoundError(f"Sliced audio path {sliced_audio_path} does not exist. Please create it first.")
    
    if not output_audio_path.exists():
        output_audio_path.mkdir(parents=True, exist_ok=True)
    extract_jukebox_features(sliced_audio_path, output_audio_path)