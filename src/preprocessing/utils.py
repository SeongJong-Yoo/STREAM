import numpy as np
import torch
from copy import deepcopy
import pickle
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import os

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.utility import resample

def adjust_floor_height(data):
    motion_joints = data['motion_positions']
    motion_data = data['motion_data']

    T, J, D = motion_joints.shape
    l_toe_h = motion_joints[0, 10, 1]
    r_toe_h = motion_joints[0, 11, 1]

    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h) / 2
    else:
        height = min(l_toe_h, r_toe_h)

    motion_joints = motion_joints - np.array([0, height, 0])
    motion_data[..., 1] = motion_data[..., 1] - height
    data['motion_positions'] = motion_joints
    data['motion_data'] = motion_data
    return data

def process_smpl_data(kp_dir, name, fk, data_set='AIST', data=None):
    kp_data = {'file_name': name}
    if data is None:
        if os.path.exists(os.path.join(kp_dir, name + '.npy')):
            smpl_data = np.load(os.path.join(kp_dir, name + '.npy'), allow_pickle=True)[()]
        elif os.path.exists(os.path.join(kp_dir, name + '.pkl')):
            smpl_data = load_pickle_data(os.path.join(kp_dir, name + '.pkl'))
        else:
            raise FileNotFoundError(f"SMPL data not found for {name}")
    else:
        smpl_data = data
    
    trans = smpl_data['smpl_trans']
    rot = smpl_data['smpl_poses']
    scale = smpl_data['smpl_scaling']
    if isinstance(trans, torch.Tensor):
        trans = trans.detach().cpu().numpy()
    if isinstance(rot, torch.Tensor):
        rot = rot.detach().cpu().numpy()
    if isinstance(scale, torch.Tensor):
        scale = scale.detach().cpu().numpy()
    if data_set == 'AIST':
        trans = trans / scale

    T = trans.shape[0]
    rot = rot.reshape(T, -1, 3)
    J = rot.shape[1]
    rot = R.from_rotvec(rot.reshape(T*J, -1)).as_matrix().reshape(T, J, 9)
    if 'joint_locations' in smpl_data:
        motion_positions = smpl_data['joint_locations']
    else:
        fk_input = np.concatenate([trans, rot.reshape(T, -1)], axis=-1)
        motion_positions = fk(fk_input).detach().cpu().numpy()
    trans = np.concatenate([trans, np.zeros((T, 6))], axis=-1)
    motion_data = np.concatenate([trans[:, None, :], rot], axis=1)
    kp_data['motion_data'] = motion_data
    if 'fps' in smpl_data:
        kp_data['fps'] = smpl_data['fps']
    else:
        kp_data['fps'] = 60
    kp_data['motion_positions'] = motion_positions
    kp_data['scale'] = scale
    # kp_data = adjust_floor_height(kp_data)
    return kp_data


def normalize_data_length(data, target_len, src_fps, data_type='list'):
    if data_type == 'list':
        current_len = data.shape[0]
        target_fps = target_len / current_len * src_fps

        data = resample(data, target_fps, src_fps, target_len)
    elif data_type == 'dict':
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if not isinstance(value, np.ndarray):
                continue
            if len(value.shape) < 2:
                continue
            current_len = value.shape[0]
            target_fps = target_len / current_len * src_fps 
            data[key] = resample(value, target_fps, src_fps, target_len)
        data['fps'] = target_fps
    return data, target_fps

def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

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

def slice_label(labels, start_frame, end_frame, description_idx):
    sliced_label = labels[start_frame:end_frame]
    description = sliced_label[:, 2]
    total_num_of_descriptions = len(description[0].split(';'))
    cur_idx = description_idx % total_num_of_descriptions
    # split each description by ";" and select the one at description_idx
    description = [desc.split(';')[cur_idx].strip() for desc in description]
    description = np.array(description)
    sliced_label[:, 2] = description
    sliced_label_dict = {
        'data': sliced_label,
        'label_index' : None,
        'current_fps': 30,
        'target_fps': 30
    }
    return sliced_label

def slice_audio(data, beat_times, FPS, window_size, stride):
    origin_beat_time = deepcopy(beat_times)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    total_len = data.shape[0]
    ws_len = int(window_size * FPS)
    s_len = int(stride * FPS)
    num_chunks = max(1, (total_len - ws_len) // s_len + 1)

    sliced_data = []
    time_chunks = []
    beat_chunks = []
    for i in range(num_chunks):
        start = i * stride
        end = start + window_size
        time_chunk = {'start': start, 'end': end}
        sliced_data.append(data[int(start * FPS):int(start * FPS) + ws_len])
        time_chunks.append(time_chunk)

        beat_start = np.where(origin_beat_time >= start)[0]
        beat_end = np.where(origin_beat_time >= end)[0]
        if len(beat_start) == 0:
            beat_chunks.append([])
            continue
        if len(beat_end) == 0:
            beat_chunk = origin_beat_time[beat_start[0]:]
        else:
            beat_chunk = origin_beat_time[beat_start[0]:beat_end[0]]
        beat_chunks.append(beat_chunk - start)

    return sliced_data, time_chunks, beat_chunks

def slice_data_by_time(data, fps, time_chunks, data_type='list'):
    chunks = []
    for chunk in time_chunks:
        start, end = int(chunk['start'] * fps), int(chunk['end'] * fps)
        if data_type == 'list':
            if len(data) < end:
                break
            chunk = data[start:end]
        elif data_type == 'dict':
            chunk = slice_dict_data(data, start, end)
            if chunk is None:
                break

        chunks.append(chunk)
    return chunks

def slice_dict_data(data, start, end):
    output = deepcopy(data)

    original_len = data['motion_data'].shape[0]
    if original_len < end:
        return None
    
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if not isinstance(value, np.ndarray):
            continue
        if len(value.shape) > 1:
            output[key] = value[start:end]

    return output

def slice_data(data, fps, window_time=5, stack=True):
        """
        Slice the data into chunks of 5 seconds
        """
        window_length = int(window_time * fps)
        total_length = data.shape[0]
        total_time = total_length / fps
        num_bins = int(total_time // window_time)

        output = []
        for i in range(num_bins):
            start_idx = i * window_length
            end_idx = (i + 1) * window_length
            output.append(data[start_idx:end_idx])
        if stack:
            output = np.stack(output)
        return output
