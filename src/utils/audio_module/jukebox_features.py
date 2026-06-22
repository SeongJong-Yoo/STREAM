# Jukebox Feature Extraction from https://github.com/Stanford-TML/EDGE?tab=readme-ov-file 
import os
from functools import partial
from pathlib import Path
import argparse

import jukemirlib
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.dataloader.utility import slice_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="./data/test_data")
    parser.add_argument("--output_dir", type=str, default="./data/jukebox_features_sliced")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--layer", type=int, default=66)
    parser.add_argument("--group_id", type=int, default=1)  # -1 for all
    parser.add_argument("--group_file", type=str, default="./src/utils/audio_module/AIST_group.txt")
    return parser.parse_args()

def compute_jukebox_features(audios, fps, layer):
    features = []
    for audio in audios:
        reps = jukemirlib.extract(audio.squeeze(), layers=[layer], downsample_target_rate=fps)
        features.append(reps[layer])
    return features

def extract(fpath, output_dir, fps, layer, skip_completed=False):

    os.makedirs(output_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(output_dir, audio_name)

    if skip_completed:
        return
    
    sr = 44100
    audio = jukemirlib.load_audio(fpath)
    audio_sliced = slice_data(audio, sr, window_size=5, stride=0.5, mode='overlap')
    features = compute_jukebox_features(audio_sliced, fps, layer)

    return features, save_path



def extract_folder(args, groups=None):
    src = args.src_dir
    fpaths = Path(src).glob("*")
    fpaths = sorted(list(fpaths))
    extract_ = partial(extract, output_dir=args.output_dir, fps=args.fps, layer=args.layer, skip_completed=False)
    for fpath in tqdm(fpaths):
        if groups is not None:
            if fpath.stem not in groups:
                continue
        rep, path = extract_(fpath)
        if isinstance(rep, list):
            for i in range(len(rep)):
                np.save(path + f"_chunk{i}", rep[i])
        else:
            np.save(path, rep)


if __name__ == "__main__":
    args = parse_args()

    groups = None

    if args.group_id != -1:
        with open(args.group_file, 'r') as f:
            groups = [line.strip().split(',')[0] for line in f]
    
    extract_folder(args, groups)
