import numpy as np
from copy import deepcopy
import sys
import os
from pathlib import Path
import torch
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.geometry import rot_matrix_to_6d, euler_angles_to_matrix, rot_6d_to_matrix

def align_pelvis_origin(motion, l_hip, r_hip, l_shoulder, r_shoulder):
    root = motion[0, l_hip, :]
    x_axis = -motion[0, r_hip, :] + motion[0, l_hip, :]
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = (motion[0, l_shoulder, :] + motion[0, r_shoulder, :]) / 2 - (motion[0, l_hip, :] + motion[0, r_hip, :]) / 2
    z_axis = z_axis - np.dot(x_axis, z_axis) * x_axis
    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[:, 0] = x_axis / np.linalg.norm(x_axis)
    rotation_matrix[:, 1] = y_axis / np.linalg.norm(y_axis)
    rotation_matrix[:, 2] = z_axis / np.linalg.norm(z_axis)

    motion = motion - root
    motion = motion @ rotation_matrix
    return motion

# def compute_norm_rot(motion, idx):
#     T = motion.shape[0]
#     l_hip, r_hip, l_shoulder, r_shoulder = idx['l_hip'], idx['r_hip'], idx['l_shoulder'], idx['r_shoulder']
#     x_axis = -motion[:, r_hip, :] + motion[:, l_hip, :]
#     x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
#     y_axis = (motion[:, l_shoulder, :] + motion[:, r_shoulder, :]) / 2 - (motion[:, l_hip, :] + motion[:, r_hip, :]) / 2
#     y_axis = y_axis - np.sum(y_axis * x_axis, axis=-1, keepdims=True) * x_axis
#     z_axis = np.cross(x_axis, y_axis)

#     rotation_matrix = np.zeros((T, 3, 3))
#     rotation_matrix[:, :, 0] = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
#     rotation_matrix[:, :, 1] = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
#     rotation_matrix[:, :, 2] = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)

#     comp = np.array([[  0.9800666,  0.0000000,  0.1986693],
#                      [   0.0000000,  1.0000000,  0.0000000],
#                      [  -0.1986693,  0.0000000,  0.9800666]])
#     rotation_matrix = rotation_matrix @ comp

#     return rotation_matrix

def compute_norm_rot(motion, idx):
    T = motion.shape[0]
    l_hip, r_hip, l_shoulder, r_shoulder = idx['l_hip'], idx['r_hip'], idx['l_shoulder'], idx['r_shoulder']
    x_axis = -motion[:, r_hip, :] + motion[:, l_hip, :]
    x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
    z_axis = (motion[:, l_shoulder, :] + motion[:, r_shoulder, :]) / 2 - (motion[:, l_hip, :] + motion[:, r_hip, :]) / 2
    z_axis = z_axis - np.sum(z_axis * x_axis, axis=-1, keepdims=True) * x_axis
    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.zeros((T, 3, 3))
    rotation_matrix[:, :, 0] = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
    rotation_matrix[:, :, 1] = y_axis / np.linalg.norm(y_axis, axis=-1, keepdims=True)
    rotation_matrix[:, :, 2] = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)

    # comp = np.array([[0.9800666, -0.1986693,  0.0000000],
    #                  [0.1986693,  0.9800666,  0.0000000],
    #                  [0.0000000,  0.0000000,  1.0000000]])
    # comp = np.array([[  0.9800666,  0.0000000,  0.1986693],
    #                  [   0.0000000,  1.0000000,  0.0000000],
    #                  [  -0.1986693,  0.0000000,  0.9800666]])
    rotation_matrix = rotation_matrix# @ comp

    return rotation_matrix

def apply_transform(motion, t_g, R, data_type='joints'):
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion.astype(np.float32))
    if data_type == 'joints':
        motion = motion - t_g[:, np.newaxis, :]
        motion = motion @ R # rotation_matrix = R_B^-1
        ref_t = deepcopy(t_g[0])
        ref_R = deepcopy(R[0])

        rot_matrix = R.transpose(0, 2, 1) @ ref_R
        t_global = (t_g - ref_t) @ ref_R

    elif data_type == 'tree':
        motion = torch.bmm(torch.tensor(R, dtype=torch.float32).transpose(2, 1), motion)
        motion = rot_matrix_to_6d(motion)#.cpu().numpy()
        ref_t = deepcopy(t_g[0])
        ref_R = deepcopy(R[0])
        # ref_R = torch.eye(3)

        rot_matrix = ref_R.transpose(1, 0) @ R
        t_global = np.einsum('ji, bi->bj', ref_R.transpose(1, 0), t_g)# - ref_t)

    return t_global, rot_matrix, motion

def relative_pelvis_origin(motion, idx, data_type, skeleton=None, fk=None):
    """
    Input:
        motion: (T, J, D) Time x Joint x Dimension
        idx: Dictionary of joint indices (root, l_hip, r_hip, l_shoulder, r_shoulder)
        data_type:
            'joints': 3D joints locations
            'tree': Kinematic tree based on global translation and rotation with 3D euler angles
    Output:
        relative_motion: (t_global, R_global, joint_local)
            t_global: (T, 3) Global translation based on left hip
            R_global: (T, 6) Global rotation based on hip and shoulder coordinates represented in 6D rotation
            joint_local: (T, J, 3) Local joint coordinates
    """
    ROT_REPRESENTATION=False
    root_idx = idx['root']
    T, J, D = motion.shape
    if D==9:
        ROT_REPRESENTATION=True
    t_g = motion[:, root_idx]
    if ROT_REPRESENTATION:
        t_g = t_g[:, :3]
    if data_type=='joints':
        motion_joints = deepcopy(motion)
    elif data_type=='tree':
        if fk is None or skeleton is None:
            raise ValueError("fk and skeleton are required for tree data type")
        
        keypoint_data = torch.tensor(motion, dtype=torch.float32)
        keypoint_data = keypoint_data.reshape(keypoint_data.shape[0], -1)
        motion_joints = fk.forward(keypoint_data, skeleton=skeleton).detach().cpu().numpy()
        if ROT_REPRESENTATION:
            motion_tree = motion[:, 1:].reshape(T, -1, 3, 3)
        else:
            motion_tree = euler_angles_to_matrix(torch.tensor(motion[:, 1:], dtype=torch.float32), skeleton['Hips']['order'])
        motion = deepcopy(motion_tree[:, 0])  # Root rotation (Hips)
        motion_tree = rot_matrix_to_6d(motion_tree[:, 1:])

    rotation_matrix = compute_norm_rot(motion_joints, idx)
    t_global, rot_matrix, motion = apply_transform(motion, t_g, rotation_matrix, data_type=data_type)    
    R_global = rot_matrix_to_6d(rot_matrix)

    if data_type == 'joints':
        return np.concatenate([t_global, R_global, motion.reshape(T, -1)], axis=-1)
    elif data_type == 'tree':
        motion_tree = np.concatenate([motion_tree[:, :9], motion_tree[:, 11:14], motion_tree[:, 16:18]], axis=1) # Total 14 joints, removing 5 redundant joints
        motion = np.concatenate([motion, motion_tree.reshape(T, -1)], axis=-1)            # Total 15 joints: 1 root + 14 limbs
        # return np.concatenate([t_global, R_global, np.zeros((T, 3)), motion], axis=-1)    # (3 + 6 + 3 + 15 * 6)
        return np.concatenate([t_global, R_global, motion], axis=-1)    # (3 + 6 + 3 + 15 * 6)


def absolute_process(motion, root_idx, data_type='tree'):
    T, J, D = motion.shape
    if data_type=='tree':
        if D==9:
            t_process = motion[0, root_idx, :3]
            t_g = motion[:, 0, :3] - t_process
            motion_tree = motion[:, 1:, :].reshape(T, -1, 3, 3)
            motion_tree = rot_matrix_to_6d(motion_tree)
            motion_tree = np.concatenate([motion_tree[:, :10], motion_tree[:, 12:15], motion_tree[:, 17:19]], axis=1) # Total 14 joints, removing 5 redundant joints
            motion = np.concatenate([t_g, motion_tree.reshape(T, -1)], axis=-1)
        else:
            raise ValueError(f"Invalid data shape: {motion.shape}")
    elif data_type=='joints':
        t_process = np.zeros(3)
        motion = motion
    preprocessing = {
        'rotation': np.eye(3),
        'translation': t_process
    }
    return motion, preprocessing

def find_face_direction(motion, idx):
    l_hip, r_hip, l_shoulder, r_shoulder = idx['l_hip'], idx['r_hip'], idx['l_shoulder'], idx['r_shoulder']
    x_axis = motion[l_hip, :] - motion[r_hip, :] 
    x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
    z_axis = (motion[l_shoulder, :] + motion[r_shoulder, :]) / 2 - (motion[l_hip, :] + motion[r_hip, :]) / 2
    z_axis = z_axis - np.sum(z_axis * x_axis, axis=-1, keepdims=True) * x_axis
    y_axis = np.cross(z_axis, x_axis)
    face_dir = y_axis[1:]
    face_dir = face_dir / np.linalg.norm(face_dir, axis=-1, keepdims=True)

    face_dir = np.concatenate([y_axis[0:1], y_axis[2:]], axis=0)
    face_dir = face_dir / np.linalg.norm(face_dir, axis=-1, keepdims=True)
    angle = np.arctan2(face_dir[0], face_dir[1])
    angle = np.minimum(np.pi - angle, -angle)
    rot_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
    return rot_matrix.astype(np.float32)

def compute_floor(foot, threshold=0.02, axis='y'):
    T, J, D = foot.shape
    if axis == 'y':
        foot = foot[:, :, 1].squeeze().reshape(-1)
    elif axis== 'z':
        foot = foot[:, :, 2].squeeze().reshape(-1)

    idx = np.argmin(foot)
    lowest_value = foot[idx]

    foot = foot - lowest_value
    average_mask = np.abs(foot) < threshold
    average = lowest_value + np.mean(foot[average_mask])
    return average, idx // J, idx % J


def face_forward(motion, idx, data_type, fk, skeleton=None):
    T, J, D = motion.shape
    if data_type == 'tree':
        if D==9:
            motion_tree = motion[:, 1:, :].reshape(T, -1, 3, 3)
            # Adjust the global floor height to the lowest foot position to zero
            if 'foot_idx' in idx:
                motion_joints = fk.forward(deepcopy(motion).reshape(T, -1), skeleton=skeleton).detach().cpu().numpy()
                foot = motion_joints[:, idx['foot_idx']]
                floor_min, frame_idx, foot_idx = compute_floor(foot)
                t_process = motion[0, 0, :3]# - [0, floor_min, 0]
                first_frame_joints = motion_joints[0]
            else:
                floor_min = 0
                frame_idx = None
                first_frame = torch.tensor(motion[0:1], dtype=torch.float32).reshape(1, -1)
                first_frame_joints = fk.forward(first_frame, skeleton=skeleton).squeeze().detach().cpu().numpy()
                t_process = motion[0, 0, :3]

            gap = first_frame_joints[0] - motion[0, 0, :3]
            if frame_idx is not None:
                # gap[1] += motion_joints[frame_idx, 0, 1] - motion[frame_idx, 0, 1]
                gap[1] -= (first_frame_joints[0, 1] - floor_min)#floor_min
            first_frame_joints -= t_process
            t_g = motion[:, 0, :3] - t_process
            R = find_face_direction(first_frame_joints, idx)
            motion_tree[:, 0] = np.einsum('ij, tjk -> tik', R, motion_tree[:, 0])
            t_g = np.einsum('ij, tj -> ti', R, t_g) - gap
            motion_tree = rot_matrix_to_6d(motion_tree)
            if J < 21: # SMPL: 24 joints
                motion_tree = np.concatenate([motion_tree[:, :10], motion_tree[:, 12:15], motion_tree[:, 17:19]], axis=1) # Total 14 joints, removing 5 redundant joints
            motion = np.concatenate([t_g, motion_tree.reshape(T, -1)], axis=-1)
            preprocessing = {
                'rotation': R.T,
                'translation': R.T @ gap.astype(np.float32) + t_process.astype(np.float32),
            }
    else:
        raise ValueError(f"Invalid data type: {data_type}") #TODO: Implement for joints data type

    return motion, preprocessing


def motion_preprocessing(motion, data_type='joints', method='absolute', dataset='AIST', skeleton=None, fk=None):
    """
    Preprocess the motion data.
    Inputs:
        motion: (T, D) Batch x Time x Dimension
        data_type:
            'joints': 3D joints locations
            'tree': Kinematic tree based on global translation and rotation with 3D euler angles
        method: Method for preprocessing
            'relative': Relative motion (TODO: Not implemented)
            'relative_pelvis_origin': Relative pelvis origin and hip is aligned to x-axis
            'pelvis_origin': Pelvis origin and hip is aligned to x-axis
            'absolute': No preprocessing
            'norm_pelvis_origin': (TODO: Not implemented)
        dataset: 'smpl' or 'motorica'
    Output:
        motion: (T, D) Preprocessed motion data
    """
    preprocessing = {
        'rotation': np.eye(3),
        'translation': np.zeros(3),
    }
    if dataset=='smpl':
        idx = {'root': 0, 'l_hip': 1, 'r_hip':2, 'l_shoulder':13, 'r_shoulder':14}
        idx['foot_idx'] = [7, 8, 10, 11]
    elif dataset=='motorica':
        idx = {'root': 0, 'l_hip': 2, 'r_hip':3, 'l_shoulder':8, 'r_shoulder':9}
        # idx['foot_idx'] = [15, 16]
    if method == 'relative_pelvis_origin':
        pass
        # motion, preprocessing = relative_pelvis_origin(motion, idx, data_type, skeleton=skeleton, fk=fk)
    elif method == 'pelvis_origin': # Pelvis origin and hip is aligned to x-axis
        pass
        # motion, preprocessing = align_pelvis_origin(motion, idx, data_type) # AIST keypoints order based 
    elif method == 'absolute':
        motion, preprocessing = absolute_process(motion, idx['root'], data_type)
    elif method == 'face_forward':
        motion, preprocessing = face_forward(motion, idx, data_type, fk=fk, skeleton=skeleton)
    else:
        raise ValueError(f"Invalid motion preprocessing method: {method}")
    return motion.astype(np.float32), preprocessing


def motion_postprocessing(motion, type=None, data_type='joints', preprocessing_R=None, preprocessing_t=None):
    """
    Inputs
        motion: (B, F, D)
        type: str, 'relative_pose' or None
        data_type:
            'joints': 3D joints locations
            'tree': Kinematic tree based on global translation and rotation with 3D euler angles
    Outputs
        motion_rel: (B, F, (g_t (3) + g_R (6) + r_R(15 * 6))
    """
    OUT_TO_SQUEEZE = False
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    elif not isinstance(motion, torch.Tensor):
        raise TypeError(f"Input must be numpy array or torch tensor, got {type(motion)}")

    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
        OUT_TO_SQUEEZE = True

    if type == None:
        if OUT_TO_SQUEEZE:
            return motion.squeeze()
        else:
            return motion
    elif type == 'relative_pose': # Motion: (t_g, 6D R_g, motion_rel), motion_rel: g_t(3), relative_rot (16 * 6)
        t_g = motion[:, :, :3].clone()
        R_g = motion[:, :, 3:9].clone() 
        motion_rel = motion[:, :, 9:].clone()
        if data_type == 'joints':
            motion_rel = rearrange(motion_rel, "b f (j d) -> b f j d", d=3)
        elif data_type == 'tree':
            # t_root = motion_rel[:, :, :3]   # motion_rel: (t_root, 6D R_root, 14 joints motion_rel)
            if motion_rel.shape[-1] % 6 == 0:
                R_root = rot_6d_to_matrix(motion_rel[:, :, :6]) # (B, F, 3, 3)
                motion_rel = motion_rel[:, :, 6:] # (B, F, 14*6)
            else:
                R_root = rot_6d_to_matrix(motion_rel[:, :, 3:9]) # (B, F, 3, 3)
                motion_rel = motion_rel[:, :, 9:] # (B, F, 14*6)

        R_mat = rot_6d_to_matrix(R_g)

        if data_type == 'joints':
            motion_rel = motion_rel @ R_mat + t_g.unsqueeze(-2)
        elif data_type == 'tree':
            R_root = R_mat @ R_root
            R_root = rot_matrix_to_6d(R_root)
            motion_rel = torch.cat([t_g, R_root, motion_rel], dim=-1)

        if OUT_TO_SQUEEZE:
            return motion_rel.squeeze()
        else:
            return motion_rel
        
    elif type=='absolute' :
        return motion
    elif type=='face_forward' and data_type=='tree':
        if preprocessing_R is None or preprocessing_t is None:
            return motion
        if isinstance(preprocessing_R, np.ndarray):
            preprocessing_R = torch.from_numpy(preprocessing_R)
        if isinstance(preprocessing_t, np.ndarray):
            preprocessing_t = torch.from_numpy(preprocessing_t)
        if preprocessing_R.ndim == 2:
            preprocessing_R = preprocessing_R[np.newaxis, ...]
        if preprocessing_t.ndim == 2:
            preprocessing_t = preprocessing_t[np.newaxis, ...]
        if preprocessing_R.shape[-1] == 6:
            preprocessing_R = rot_6d_to_matrix(preprocessing_R)
        
        t_g = motion[:, :, :3].clone()
        R_g = rot_6d_to_matrix(motion[:, :, 3:9].clone())
        
        motion_rel = motion[:, :, 9:].clone()
        adjusted_R = torch.einsum('bij, btjk->btik', preprocessing_R, R_g)
        adjusted_R = rot_matrix_to_6d(adjusted_R)
        adjusted_t = torch.einsum('bji, bti->btj', preprocessing_R, t_g) + preprocessing_t

        motion_rel = torch.cat([adjusted_t, adjusted_R, motion_rel], dim=-1)
        if OUT_TO_SQUEEZE:
            return motion_rel.squeeze()
        else:
            return motion_rel

    else:
        raise ValueError(f"Forward kinematics type {type} not supported")
    