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
from src.utils.audio_module.utility import compute_beats
from src.preprocessing.utils import slice_audio, slice_data_by_time, slice_dict_data, normalize_data_length, load_pickle_data, process_smpl_data
from src.preprocessing.label_format import write_compact_label
from src.preprocessing.mirror import (
    mirror_label_payload,
    mirror_smpl_motion_dict,
    mirror_text,
)
from tqdm import tqdm
from src.skeleton.smpl_fk import SMPLModel

motorica_dance_style_dict = {
"gLH": "hip_hop",
"gKR": "krump",
"gPO": "popping",
"gLO": "locking",
"gJZ": "jazz",
"gCH": "charleston",
"gTP": "tap",
"gCA": "casual",
}

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
    # for i in range(total_len):
    #     current_label = label[i, label_idx]
    #     if current_label != initial_label:
    #         counter = 0
    #         initial_label = current_label
    #     output[i] = counter
    #     counter += 1
    # return np.array(output)

def add_aist_description(label_chunks, time_chunks, description):
    output = []
    for i in range(len(label_chunks)):
        label_chunk = label_chunks[i]
        if label_chunk.shape[1] == 1:
            label_chunk = np.concatenate([label_chunk, np.repeat([''], label_chunk.shape[0]).reshape(-1, 1)], axis=-1)
        
        if description is None:
            label_chunk = np.concatenate([label_chunk, np.repeat('', label_chunk.shape[0]).reshape(-1, 1)], axis=-1)
            output.append(label_chunk)
            continue
        start = str(time_chunks[i]['start'])
        end = str(time_chunks[i]['end'])
        key = start + '_' + end

        if key in description.keys():
            T = len(label_chunk)
            label_chunk = np.concatenate([label_chunk, np.repeat(description[key], label_chunk.shape[0]).reshape(T, -1)], axis=-1)
        else:
            label_chunk = np.concatenate([label_chunk, np.repeat('', label_chunk.shape[0]).reshape(-1, 1)], axis=-1)
        output.append(label_chunk)
    return output

def add_motorica_description(label_chunks, description, dance_style):
    output = []
    for label_chunk in label_chunks:
        T = label_chunk.shape[0]
        concat_label_list = []
        for i in range(T):
            if label_chunk[i, 0].lower() == 'random':
                if label_chunk[i, 1].lower() in ['intro windup', 'random']:
                    label_chunk[i, 0] = dance_style
            if "FrameLabel" in label_chunk[i, 0]:
                label_chunk[i, 0] = label_chunk[i, 0].replace(" FrameLabel", "")
            if "HipHop" in label_chunk[i, 0]:
                label_chunk[i, 0] = label_chunk[i, 0].replace("HipHop", "Hip_Hop")
            concat_label_list.append(str(label_chunk[i, 0] + ': ' + label_chunk[i, 1]).lower())
        
        lower_case_label = []
        added_description = []
        idx_counter = {}
        for j, concat_label in enumerate(concat_label_list):
            if concat_label not in idx_counter.keys():
                idx_counter[concat_label] = description[concat_label][-1]
            lower_case_label.append(np.array([label_chunk[j, 0].lower(), label_chunk[j, 1].lower()]))
            if concat_label in description.keys():
                idx = int((idx_counter[concat_label] / 10) % (len(description[concat_label])-1))
                added_description.append(description[concat_label][idx])
            else:
                added_description.append('')
        if concat_label == 'charleston: opposites':
            test=1
        label_chunk = np.concatenate([np.array(lower_case_label).reshape(-1, 2), np.array(added_description).reshape(-1, 1)], axis=-1)
        output.append(label_chunk)
        for key in idx_counter.keys():
            description[key][-1] += 1
    return output

def read_label_segment_txt(path):
    """Read a label segment .txt file into a dictionary.

    Expected format (one key-value per line, 'key: value'):
        genre: popping
        label: tutting
        start: 1524
        end: 1623
        fps: 30
        description: ...
    """
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            key = key.strip()
            value = value.strip()
            if key in ("start", "end", "fps"):
                value = int(value)
            out[key] = value
    return out


def load_and_collect_label_segments(label_path):
    label_lists = os.listdir(label_path)
    label_lists = [f for f in label_lists if f.endswith('.txt')]
    sample_dicts = {}
    for label_list in label_lists:
        id = label_list.split('.')[0].split('_')[:-1]
        id = '_'.join(id)
        if id not in sample_dicts.keys():
            sample_dicts[id] = []
        sample_dicts[id].append(label_list)

    output = {}
    for id in sample_dicts.keys():
        sample_dicts[id].sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        output[id] = []
        for sample_lists in sample_dicts[id]:
            segment_path = os.path.join(label_path, sample_lists)
            output[id].append(read_label_segment_txt(segment_path))

    return output

def concat_label_dicts(label_dicts):
    output = {}
    for key in label_dicts.keys():
        label_list = label_dicts[key]
        start_frame = label_list[0]['start']
        end_frame = label_list[-1]['end']
        label = np.empty((end_frame - start_frame + 1, 3), dtype='<U1000')
        label.fill('')
        for label_item in label_list:
            start_frame = label_item['start']
            end_frame = label_item['end'] + 1
            label[start_frame:end_frame, 0] = label_item['genre']
            label[start_frame:end_frame, 1] = label_item['label']
            label[start_frame:end_frame, 2] = label_item['description']
        output[key] = label
    return output


def load_json_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    AUDIO_FEATURES='jukebox' # jukebox
    LABEL_ONLY=False
    AUDIO=True
    MOTION=True
    LABEL=True
    LABEL_TXT_FORM = True
    DATA_TYPE = 'motorica'  # AIST or motorica
    DATA_REPRESENTATION = 'smpl' # smpl or motorica
    LABEL_FORMAT = 'json'  # 'json' or 'npy'
    MIRROR_AUGMENT = (DATA_TYPE in ('motorica', 'AIST') and DATA_REPRESENTATION == 'smpl')
    STANDARD_AUDIO_FPS = 30
    STANDARD_MOTION_FPS = 30
    STANDARD_LABEL_FPS = 30

    # ORIGINAL_MOTION_FPS = 30 # 30 for AIST and 120 for Motorica
    WINDOW_SIZE = 5 # Window size in seconds
    STRIDE = 1 # Stride in seconds

    ignore_list = []
    movement_music_length_list = {}

    if DATA_TYPE == 'AIST':
        ORIGINAL_MOTION_FPS = 30
        root_dir = '/mnt/hdd/Dataset/AIST'
        output_dir = './data/AIST'
        wav_dir = os.path.join(root_dir, 'wavs')
        kp_dir = os.path.join(root_dir, 'prediction_result')
        if DATA_REPRESENTATION == 'smpl':
            kp_dir = os.path.join(root_dir, 'motions')
            ORIGINAL_MOTION_FPS = 60
            fk = SMPLModel(num_joints=24, smpl_model_path='./data/smpl/SMPL_NEUTRAL.pkl')
    elif DATA_TYPE == 'motorica':
        ORIGINAL_MOTION_FPS = 120
        root_dir = '/mnt/hdd/Dataset/Motorica'
        output_dir = './data/motorica'
        wav_dir = os.path.join(root_dir, 'motorica_dance_dataset', 'wav')
        kp_dir = os.path.join(root_dir, 'motorica_dance_dataset', 'synced_motion')
        if DATA_REPRESENTATION == 'smpl':
            kp_dir = os.path.join(root_dir, 'motorica_dance_dataset', 'synced_motion_smpl')
            ORIGINAL_MOTION_FPS = 30
            fk = SMPLModel(num_joints=24, smpl_model_path='./data/smpl/SMPL_NEUTRAL.pkl')
        movement_music_length_list_path = os.path.join(root_dir, 'movement_music_length.csv')
        import pandas as pd
        movement_music_length_df = pd.read_csv(movement_music_length_list_path)
        csv_data = movement_music_length_df.values.tolist()
        for i in range(len(csv_data)):
            movement_music_length_list[csv_data[i][0]] = {'movement_length': csv_data[i][1], 'music_length': csv_data[i][2]}
    else:
        raise ValueError(f"Invalid DATA_TYPE: {DATA_TYPE}")

    if os.path.exists(os.path.join(root_dir, 'ignore_list.txt')):
        with open(os.path.join(root_dir, 'ignore_list.txt'), 'r') as f:
            ignore_list = f.read().splitlines()

    if not os.path.exists(os.path.join(output_dir, 'sliced_audio')):
        os.makedirs(os.path.join(output_dir, 'sliced_audio'))
    if not os.path.exists(os.path.join(output_dir, 'sliced_beats')):
        os.makedirs(os.path.join(output_dir, 'sliced_beats'))
    if not os.path.exists(os.path.join(output_dir, 'sliced_motion')):
        os.makedirs(os.path.join(output_dir, 'sliced_motion'))
    if DATA_REPRESENTATION == 'smpl':
        if not os.path.exists(os.path.join(output_dir, 'sliced_motion_smpl')):
            os.makedirs(os.path.join(output_dir, 'sliced_motion_smpl'))
    if not os.path.exists(os.path.join(output_dir, 'sliced_labels')):
        os.makedirs(os.path.join(output_dir, 'sliced_labels'))

    # LABEL_TXT_FORM is Motorica-specific: it parses the per-segment .txt
    # annotations under motorica_dance_dataset/label_segment/. AIST and
    # HumanML3D have their own label sources further down the loop, so for
    # those datasets this block is skipped and the per-DATA_TYPE branches
    # at line ~381 take over.
    use_label_txt_form = LABEL_TXT_FORM and DATA_TYPE == 'motorica'
    if use_label_txt_form:
        total_label_dicts = load_and_collect_label_segments(os.path.join(root_dir, 'motorica_dance_dataset', 'label_segment'))
        total_label_dicts = concat_label_dicts(total_label_dicts)

    wavs = Path(wav_dir).glob("*.wav")
    wavs = sorted([wav.stem for wav in wavs])
    kps = Path(kp_dir).glob("*.npy")
    if DATA_TYPE=='AIST' and DATA_REPRESENTATION=='smpl':
        kps = Path(kp_dir).glob("*.pkl")

    kps = sorted(list(kps))
    print(f"Found {len(wavs)} wav files and {len(kps)} keypoint files.")
    refs = kps

    if LABEL:
        if DATA_TYPE=='AIST':
            description_path = os.path.join(root_dir, 'description.pkl')
            if os.path.exists(description_path):
                description_dict = load_pickle_data(description_path)
            else:
                description_dict = {}

    counter = 0
    for ref in tqdm(refs):
        name = Path(ref).stem
        if name in ignore_list:
            continue
        # Load Motion Data
        if DATA_TYPE == 'AIST':
            if DATA_REPRESENTATION == 'smpl':
                kp_data = process_smpl_data(kp_dir, name, fk, data_set='AIST')
            else:
                kp_data = np.load(os.path.join(kp_dir, name + '_motorica.npy'), allow_pickle=True)[()]
                kp_data['file_name'] = kp_data['file_name'][0]
                kp_data['motion_data'] = kp_data['motion_data'].squeeze(0)#.cpu().numpy()
                kp_data['motion_positions'] = kp_data['motion_positions'].squeeze(0)#.cpu().numpy()
        elif DATA_TYPE == 'motorica':
            if DATA_REPRESENTATION == 'motorica':
                kp_data = np.load(os.path.join(kp_dir, name + '.npy'), allow_pickle=True)[()]
            elif DATA_REPRESENTATION == 'smpl':
                kp_data = process_smpl_data(kp_dir, name, fk)
        elif DATA_TYPE == 'HumanML3D':
            # strip the name from the "ref"
            path_to_file = str(Path(ref).parent)
            kp_data = process_smpl_data(path_to_file, name, fk)
           
        ORIGINAL_MOTION_FPS = kp_data['fps']


        if AUDIO:
            # Load Audio Data
            if name not in wavs:
                print(f"Warning: {name} not in wavs")
                continue
            wav_path = os.path.join(wav_dir, f"{name}.wav")
            audio, sr = librosa.load(wav_path, sr=44100)
        else:
            # create an audio placeholder that looks like a silent audio
            audio = np.zeros((int(len(kp_data['motion_data']) / ORIGINAL_MOTION_FPS * 44100),), dtype=np.float32)
            sr = 44100


        # Load Label Data
        if LABEL:
            label = None
            if use_label_txt_form:
                label = total_label_dicts[name]
            elif DATA_TYPE=='motorica':
                label_path = os.path.join(root_dir, 'motorica_dance_dataset', 'full_label', f"{name}.npy")
                label = np.load(label_path, allow_pickle=True)[()]
            elif DATA_TYPE=='AIST':
                STANDARD_LABEL_FPS = 30
                label_path = os.path.join(root_dir, 'genre_id.json')
                with open(label_path, 'r') as f:
                    label_dict = json.load(f)
                try:
                    label = label_dict[name.split('_')[4]]
                except:
                    for key in label_dict.keys():
                        if key in name:
                            label = label_dict[key]
                            break
                    if label is None:
                        label = 'unknown'
                        print(f"Warning: {name} has no label")
                        continue
                ratio = STANDARD_MOTION_FPS // STANDARD_LABEL_FPS
                fps_adjustment_ratio = ORIGINAL_MOTION_FPS // STANDARD_LABEL_FPS
                label = np.repeat(label.lower(), len(kp_data['motion_data']) // fps_adjustment_ratio, axis=0)[:, np.newaxis]
                if not name in description_dict.keys():
                    description = None
                else:
                    description = description_dict[name]
                
        # Synchronize audio and motion length
        if name in movement_music_length_list:
            motion_len = kp_data['motion_data'].shape[0] / ORIGINAL_MOTION_FPS
            min_time = min(movement_music_length_list[name]['movement_length'], movement_music_length_list[name]['music_length'], motion_len)
            audio = audio[:int(min_time * sr)]
            kp_data = slice_dict_data(kp_data, 0, int(min_time * ORIGINAL_MOTION_FPS))

        # Adjust the label length and motion length to be the same
        if LABEL:
            label_length = int(label.shape[0] * ORIGINAL_MOTION_FPS / STANDARD_LABEL_FPS)
            motion_len = kp_data['motion_data'].shape[0]
            if label_length > motion_len:
                label = label[:int(motion_len * STANDARD_LABEL_FPS / ORIGINAL_MOTION_FPS)]
            elif label_length < motion_len:
                kp_data = slice_dict_data(kp_data, 0, label_length)
                audio = audio[:int(label_length * sr / ORIGINAL_MOTION_FPS)]

        tempo, beat_times = compute_beats([audio, sr])
        # Slice audio 
        audio_chunks, time_chunks, beat_chunks = slice_audio(audio, beat_times, sr, WINDOW_SIZE, STRIDE)
        if len(audio_chunks) == 0:
            print(f"Warning: {name} has no audio chunks")
            continue

        d_type = 'dict'
        motion_chunks = slice_data_by_time(kp_data, fps=ORIGINAL_MOTION_FPS, time_chunks=time_chunks, data_type=d_type)
        if len(motion_chunks) != len(audio_chunks):
            print(f"Warning: {name} has a different number of motion and audio chunks")
            continue

  
        if LABEL:
            label_chunks = []
            if label is not None:
                label_index = generate_index(label)
                label_chunks = slice_data_by_time(label, fps=STANDARD_LABEL_FPS, time_chunks=time_chunks, data_type='list')
                if DATA_TYPE=='AIST':
                    label_chunks = add_aist_description(label_chunks, time_chunks, description)
                if DATA_TYPE=='motorica':
                    splitted_name = name.split('_')
                    for i in range(len(splitted_name)):
                        if splitted_name[i] in motorica_dance_style_dict.keys():
                            dance_style = motorica_dance_style_dict[splitted_name[i]]
                            break
                    # label_chunks = add_motorica_description(label_chunks, description, dance_style)
                label_index_chunks = slice_data_by_time(label_index, fps=STANDARD_LABEL_FPS, time_chunks=time_chunks, data_type='list')
                if len(label_chunks) != len(motion_chunks):
                    print(f"Warning: {name} has a different number of label and motion chunks")
                    continue

            for i, label in enumerate(label_chunks):
                # label_chunk, label_fps = normalize_label_length(label, STANDARD_TIME * STANDARD_MOTION_FPS, STANDARD_LABEL_FPS)
                ratio = STANDARD_MOTION_FPS // STANDARD_LABEL_FPS
                label_chunk = np.repeat(label, ratio, axis=0)
                label_index_chunk = np.repeat(label_index_chunks[i], ratio, axis=0)
                # Adjust the repeated values to create continuous counter
                for j in range(len(label_index_chunk)):
                    original_idx = j // ratio
                    offset = j % ratio
                    label_index_chunk[j] = label_index_chunks[i][original_idx] * ratio + offset

                if LABEL_FORMAT == 'json':
                    label_save_path = os.path.join(output_dir, 'sliced_labels', f"{name}_chunk{i}_label.json")
                    write_compact_label(
                        label_save_path,
                        length=label_chunk.shape[0],
                        fps=STANDARD_MOTION_FPS,
                        per_frame_label=label_chunk,
                    )
                    if MIRROR_AUGMENT:
                        from src.preprocessing.label_format import read_compact_label
                        mirrored_payload = mirror_label_payload(read_compact_label(label_save_path))
                        mirror_save_path = os.path.join(output_dir, 'sliced_labels', f"{name}_chunk{i}_M_label.json")
                        write_compact_label(
                            mirror_save_path,
                            length=mirrored_payload['length'],
                            fps=mirrored_payload['fps'],
                            segments=mirrored_payload['segments'],
                        )
                else:
                    label_save_path = os.path.join(output_dir, 'sliced_labels', f"{name}_chunk{i}_label.npy")
                    output = {'data': label_chunk, 'label_index': label_index_chunk, 'current_fps': STANDARD_MOTION_FPS, 'target_fps': STANDARD_MOTION_FPS}
                    np.save(label_save_path, output)
                    if MIRROR_AUGMENT:
                        # Mirror text in-place on a copy of the per-frame matrix.
                        mirrored_data = np.array(
                            [
                                [row[0], mirror_text(str(row[1])), mirror_text(str(row[2]))]
                                for row in label_chunk
                            ],
                            dtype=label_chunk.dtype,
                        )
                        mirror_save_path = os.path.join(output_dir, 'sliced_labels', f"{name}_chunk{i}_M_label.npy")
                        np.save(
                            mirror_save_path,
                            {
                                'data': mirrored_data,
                                'label_index': label_index_chunk,
                                'current_fps': STANDARD_MOTION_FPS,
                                'target_fps': STANDARD_MOTION_FPS,
                            },
                        )

        if LABEL_ONLY:
            continue

        for i in range(len(motion_chunks)):
            if AUDIO:
                # Save audio chunk to wav file
                audio_path = os.path.join(output_dir, 'sliced_audio', f"{name}_chunk{i}.wav")
                sf.write(audio_path, audio_chunks[i], sr)

                # Save beat chunk to npy file
                beat_path = os.path.join(output_dir, 'sliced_beats', f"{name}_chunk{i}.npy")
                np.save(beat_path, beat_chunks[i])

            if MOTION:
                # Resample motion chunk to fixed size
                motion_len= int(STANDARD_MOTION_FPS * WINDOW_SIZE)
                d_type = 'dict'
                motion_chunk, motion_fps = normalize_data_length(motion_chunks[i], motion_len, ORIGINAL_MOTION_FPS, d_type)
                motion = {'motion': motion_chunk, 'current_fps': STANDARD_MOTION_FPS, 'target_fps': STANDARD_MOTION_FPS}
                if DATA_REPRESENTATION == 'smpl':
                    np.save(os.path.join(output_dir, 'sliced_motion_smpl', f"{name}_chunk{i}_motion.npy"), motion)
                    if MIRROR_AUGMENT:
                        mirrored_motion = {
                            'motion': mirror_smpl_motion_dict(motion_chunk),
                            'current_fps': STANDARD_MOTION_FPS,
                            'target_fps': STANDARD_MOTION_FPS,
                        }
                        np.save(
                            os.path.join(output_dir, 'sliced_motion_smpl', f"{name}_chunk{i}_M_motion.npy"),
                            mirrored_motion,
                        )
                else:
                    np.save(os.path.join(output_dir, 'sliced_motion', f"{name}_chunk{i}_motion.npy"), motion)

    if AUDIO_FEATURES == 'jukebox':
        print("Now run data_prep_jukebox.py script to generate audio features")