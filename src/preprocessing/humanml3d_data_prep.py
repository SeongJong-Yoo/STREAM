import numpy as np
import torch
import json
import pickle
from pathlib import Path
import os
import sys
import librosa
import soundfile as sf
from copy import deepcopy
import math
from scipy.spatial.transform import Rotation as R
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing.utils import process_smpl_data, slice_dict_data
from src.preprocessing.label_format import write_compact_label, _segments_from_per_frame_label
from tqdm import tqdm
from src.skeleton.smpl_fk import SMPLModel

def generate_index(label):
    total_len = label.shape[0]
    label_len = label.shape[1]
    output = np.zeros((total_len), dtype=int)
    label_idx = 1
    if label_len == 1:
        label_idx = 0
    initial_label = label[0, label_idx]
    sliced_index_list = []
    counter = 0

    def _ramp(n):
        # 0..1 ramp over n frames; degenerate cases collapse to zeros so we
        # don't divide by zero on a single-frame segment.
        if n <= 1:
            return np.zeros(n)
        return np.arange(n) / (n - 1)

    for i in range(total_len):
        current_label = label[i, label_idx]
        if current_label != initial_label:
            sliced_index_list.append(_ramp(counter))
            initial_label = current_label
            counter = 0
        counter += 1
    sliced_index_list.append(_ramp(counter))
    output = np.concatenate(sliced_index_list)
    return output

def slice_audio_frame(audio, start_frame, end_frame, fps):
    if not isinstance(audio, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if len(audio.shape) == 1:
        audio = audio.reshape(-1, 1)
    if start_frame < 0 or end_frame > audio.shape[0]:
        raise ValueError("Start or end frame is out of bounds.")
    if start_frame >= end_frame:
        raise ValueError("Start frame must be less than end frame.")
    sliced_data = audio[start_frame:end_frame]
    return sliced_data

def slice_motion_sequence(data, start_frame, end_frame):
    if isinstance(data, list) or isinstance(data, np.ndarray):
        if len(data) <= end_frame:
            raise ValueError(f"End frame {end_frame} exceeds the length of the motion sequence, which is {len(data)}.")
        sliced_data = data[start_frame:end_frame]
    elif isinstance(data, dict):
        sliced_data = slice_dict_data(data, start_frame, end_frame)
        if sliced_data is None:
            raise ValueError(f"End frame {end_frame} exceeds the length of the motion sequence in the dictionary, which is {len(data['motion_data'])}.")
    else:
        raise TypeError("Unsupported data type. Use 'list' or 'dict'.")
    
    return sliced_data

def build_humanml3d_label_payload(per_frame_label, fps):
    """Compact payload for HumanML3D where each segment carries its full
    list of candidate descriptions (option b in the refactor plan)."""
    segments = _segments_from_per_frame_label(per_frame_label)
    for seg in segments:
        joined = seg.pop("description", "")
        descs = [p.strip() for p in str(joined).split(";") if p.strip()]
        seg["descriptions"] = descs if descs else [""]
    return {
        "fps": int(fps),
        "length": int(per_frame_label.shape[0]),
        "segments": segments,
    }


def slice_label(labels, label_index, start_frame, end_frame, description_idx):
    sliced_label = labels[start_frame:end_frame]
    sliced_label_index = label_index[start_frame:end_frame]
    description = sliced_label[:, 2]
    total_num_of_descriptions = len(description[0].split(';'))
    cur_idx = description_idx % total_num_of_descriptions
    # split each description by ";" and select the one at description_idx
    try:
        description = [desc.split(';')[cur_idx].strip() for desc in description]
    except IndexError:
        description = [desc.split(';')[0].strip() for desc in description]
    description = np.array(description)
    sliced_label[:, 2] = description
    sliced_label_dict = {
        'data': sliced_label,
        'label_index' : sliced_label_index,
        'current_fps': STANDARD_LABEL_FPS,
        'target_fps': STANDARD_MOTION_FPS
    }
    return sliced_label_dict


if __name__ == "__main__":
    AUDIO_FEATURES='jukebox' # jukebox or tempogram
    LABEL_ONLY=True
    AUDIO=False
    MOTION=False
    LABEL=True
    DATA_TYPE = 'HumanML3D'  # AIST or motorica or HumanML3D
    DATA_REPRESENTATION = 'smpl' # smpl or motorica
    LABEL_FORMAT = 'json'  # 'json' (compact, descriptions list) or 'npy' (legacy)
    STANDARD_AUDIO_FPS = 30
    STANDARD_MOTION_FPS = 30
    STANDARD_LABEL_FPS = 30

    # ORIGINAL_MOTION_FPS = 30 # 30 for AIST and 120 for Motorica
    WINDOW_SIZE = 5 # Window size in seconds
    STRIDE = 1 # Stride in seconds

    ignore_list = []
    movement_music_length_list = {}

    ORIGINAL_MOTION_FPS = 30
    root_dir = '/mnt/hdd/Dataset/HumanML3D/smpl'
    output_dir = Path('./data/HumanML3D')
    motion_summary_path = "./data/HumanML3D/processed_texts/combined_motion_summaries.json"
    motion_description_path = Path("./data/HumanML3D/texts")
    kp_dir = root_dir
    fk = SMPLModel()

    if os.path.exists(os.path.join(root_dir, 'ignore_list.txt')):
        with open(os.path.join(root_dir, 'ignore_list.txt'), 'r') as f:
            ignore_list = f.read().splitlines()

    if DATA_REPRESENTATION == 'smpl':
        if not os.path.exists(os.path.join(output_dir, 'sliced_motion_smpl')):
            os.makedirs(os.path.join(output_dir, 'sliced_motion_smpl'))
    if not os.path.exists(os.path.join(output_dir, 'sliced_labels')):
        os.makedirs(os.path.join(output_dir, 'sliced_labels'))

    kps = Path(kp_dir).glob("*.npy")

    print(f"Looking for .npy files in {kp_dir} recursively")
    # recursively find all .npy files
    kps = Path(kp_dir).rglob("*.npy")
    kps = list(kps)
    if len(kps  )==0:
        raise ValueError(f"No .npy files found in {kp_dir}")
    
    print(f'len(kps): {len(list(kps))}')

    # process the label
    label_path = "./data/HumanML3D/texts"
    index_file_path = "./data/HumanML3D/index.csv"
    humanml3d_index_data = pd.read_csv(index_file_path)
    # for the source_path column, remove the leading "./pose_data"
    humanml3d_index_data['source_path'] = humanml3d_index_data['source_path'].str.replace("./pose_data/", "")

    with open(motion_summary_path, 'r') as f:
        humanml3d_motion_summary_dict = json.load(f)

    kps = sorted(list(kps))
    print(f"Found {len(kps)} keypoint files.")
    refs = kps

    # counter = 0
    for ref in tqdm(refs):
        name = Path(ref).stem
        if name in ignore_list:
            continue
        # strip the name from the "ref"
        path_to_file = str(Path(ref).parent)
        kp_data = process_smpl_data(path_to_file, name, fk)
           
        ORIGINAL_MOTION_FPS = kp_data['fps']

        # Load Audio Data
        # create an audio placeholder that looks like a silent audio
        audio = np.zeros((int(len(kp_data['motion_data']) / ORIGINAL_MOTION_FPS), 4800), dtype=np.float32)


        file_name = Path(ref)
        # only get the relative path from kp_dir
        file_name = str(file_name.relative_to(kp_dir))
        index_entry = humanml3d_index_data[humanml3d_index_data['source_path'].str.strip().str.rsplit('.').str[0] == file_name.split('.')[0]]
        if index_entry.empty:
            print(f"Warning: {file_name} not found in index file")
            continue
        # create a empty label place holder that is three dimensions: ([[style, techinque, description], ...])
        label = np.full((len(kp_data['motion_data']), 3), '', dtype='object')  # Initialize a 2D array with 3 columns for style, technique, and description
        label_id = np.full((len(kp_data['motion_data']),), '', dtype='object')  # Initialize a 1D array for label ids
        
        # if index entry has more than one track, we need to handle it
        for i in range(len(index_entry)):
            start_frame = index_entry['start_frame'].values[i]
            end_frame = index_entry['end_frame'].values[i]
            if end_frame == 0:
                end_frame = len(kp_data['motion_data'])-1
            curr_label_id = index_entry['new_name'].values[i].split('.')[0]
            # the starting time and ending time recorded in the index file are based on 20 fps, we need to convert to 30 fps
            start_frame = int(start_frame * STANDARD_MOTION_FPS / 20)
            end_frame = int(end_frame * STANDARD_MOTION_FPS / 20)
            
            # according to the humanml3d processing script, we should trim the first few seconds. https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb.
            # however, we will just adjust the start_frame and end_frame accordingly
            if 'humanact12' not in file_name:
                if 'Eyes_Japan_Dataset' in file_name:
                    start_frame += 3 * STANDARD_MOTION_FPS
                    end_frame += 3 * STANDARD_MOTION_FPS
                if 'MPI_HDM05' in file_name:
                    start_frame += 3 * STANDARD_MOTION_FPS
                    end_frame += 3 * STANDARD_MOTION_FPS
                if 'TotalCapture' in file_name:
                    start_frame += 1 * STANDARD_MOTION_FPS
                    end_frame += 1 * STANDARD_MOTION_FPS
                if 'MPI_Limits' in file_name:
                    start_frame += 1 * STANDARD_MOTION_FPS
                    end_frame += 1 * STANDARD_MOTION_FPS
                if 'Transitions_mocap' in file_name:
                    start_frame += int(0.5 * STANDARD_MOTION_FPS)
                    end_frame += int(0.5 * STANDARD_MOTION_FPS)
            
            # fill the label with label_id
            summary = humanml3d_motion_summary_dict.get(f"{curr_label_id}.txt", '')
            description = motion_description_path / f"{curr_label_id}.txt"
            if description.exists():
                with open(description, 'r') as f:
                    description = f.read().strip()
            else:
                raise ValueError(f"Description file {description} does not exist")
            # convert description into a list of sentence. Split by newline
            description_list = description.split('\n')
            # for each item in the description_list, split by "." and only keep the first section (the actual description)
            description_list = [item.split('.')[0] for item in description_list if len(item.strip()) > 0]
            description_list = [item.split('#')[0] for item in description_list if len(item.strip()) > 0]
            # fill the label array
            label[start_frame:end_frame, 1] = summary.lower()
            # randomly select one description from the list
            description = ";".join(description_list)
            label[start_frame:end_frame, 2] = description.lower()
            if "dance" in description:
                style = "dance"
            else:
                style = "general motion"
            label[start_frame:end_frame, 0] = style.lower()
            label_id[start_frame:end_frame] = curr_label_id

        label_index = generate_index(label)
        # generate a mask from the label. If all three columns are empty, then the mask is 0, otherwise 1
        label_mask = np.where(np.all(label == '', axis=1), 0, 1)
        # sanity check if label_mask is the same length as motion data
        assert label_mask.shape[0] == kp_data['motion_data'].shape[0], f"Label mask length {label_mask.shape[0]} is not equal to motion data length {kp_data['motion_data'].shape[0]}"
        # Generate a collection of slices with start and end indices
        total_frames = len(kp_data['motion_data'])
        slice_length = int(WINDOW_SIZE * STANDARD_MOTION_FPS)
        stride_length = int(STRIDE * STANDARD_MOTION_FPS)

        slices_index = [
            (start, min(start + slice_length, total_frames))
            for start in range(0, total_frames - slice_length + 1, stride_length)
        ]
        # get the value of label_mask for each slice, if any of the value is zero, then we ignore this slice
        valid_slices = []
        for (start, end) in slices_index:
            if np.all(label_mask[start:end]):
                valid_slices.append((start, end))

        for idx, (start_frame, end_frame) in enumerate(valid_slices):
            sliced_motion = slice_motion_sequence(kp_data, start_frame=start_frame, end_frame=end_frame)
            if LABEL_FORMAT == 'json':
                sliced_per_frame = label[start_frame:end_frame]
                json_payload = build_humanml3d_label_payload(sliced_per_frame, fps=STANDARD_MOTION_FPS)
                sliced_label = None  # written below as JSON; legacy npy path skipped
            else:
                sliced_label = slice_label(label, label_index, start_frame=start_frame, end_frame=end_frame, description_idx=idx)

            # get the id file name
            motion_id = label_id[start_frame:end_frame]
            # check if there is only one unique id in the motion_id
            unique_ids = np.unique(motion_id)
            # find the majority id and use that as the id
            if len(unique_ids) == 0:
                print(f"Warning: No valid label id found for slice {idx} of file {name}, skipping...")
                continue
            if len(unique_ids) > 1:
                id_counts = {uid: np.sum(motion_id == uid) for uid in unique_ids}
                majority_id = max(id_counts, key=id_counts.get)
                print(f"Warning: Multiple label ids found for slice {idx} of file {name}: {unique_ids}. using majority id {majority_id}.")
                unique_ids = majority_id
            else:
                unique_ids = unique_ids[0]

            segment_id = start_frame // (WINDOW_SIZE * STANDARD_MOTION_FPS)
            output_file_name = f"{unique_ids}_chunk{segment_id}"
            # save motion
            motion_output_path = output_dir / "sliced_motion_smpl" /f"{output_file_name}_motion.npy"
            motion_output = {'motion': sliced_motion, 'current_fps': STANDARD_MOTION_FPS, 'target_fps': STANDARD_MOTION_FPS}
            np.save(motion_output_path, motion_output)
            # save label
            if LABEL_FORMAT == 'json':
                label_output_path = output_dir / "sliced_labels" / f"{output_file_name}_label.json"
                write_compact_label(
                    str(label_output_path),
                    length=json_payload["length"],
                    fps=json_payload["fps"],
                    segments=json_payload["segments"],
                )
            else:
                label_output_path = output_dir / "sliced_labels" / f"{output_file_name}_label.npy"
                np.save(label_output_path, sliced_label)
    print("Slicing completed.")

    #     break  # Remove this break to process all files

    # # load example of sliced label
    # example_label_path = "/fs/nexus-projects/PhysicsFall/retarget/amass_results/smplh_to_smpl_sliced/sliced_labels/005259_chunk0_label.npy"
    # example_label = np.load(example_label_path, allow_pickle=True).item()

    # # print label_index
    # print(f"Label index: {example_label['label_index']}")
