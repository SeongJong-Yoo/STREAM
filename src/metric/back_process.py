import sys
import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.quaternion import *

import torch
import os
import pytorch3d.transforms as pt3d

# n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
# kinematic_chain = t2m_kinematic_chain
# l_idx1, l_idx2 = 5, 8
# face_joint_indx = [2, 1, 17, 16]

# example_data = np.load(os.path.join("datasets/HumanML3D/new_joints", "000021" + '.npy'))
# example_data = example_data.reshape(len(example_data), -1, 3)
# example_data = torch.from_numpy(example_data)
# tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
# tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

l_idx1, l_idx2 = 5, 8
fid_r, fid_l = [8, 11], [7, 10]
r_hip, l_hip = 2, 1
joints_num = 22



""" Get Foot Contacts """
def foot_detect(positions, thres):
    velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    #     feet_l_h = positions[:-1,fid_l,1]
    #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    #     feet_r_h = positions[:-1,fid_r,1]
    #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
    return feet_l, feet_r

def compute_cont6d_params(pose, joints):
    velocity = (joints[1:, 0] - joints[:-1, 0]).clone().detach().cpu().numpy()
    r_rot = pt3d.rotation_6d_to_matrix(pose[:, 0]).detach().cpu().numpy()
    r_rot = R.from_matrix(r_rot).as_quat()
    velocity = qrot_np(r_rot[1:], velocity)

    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

    return r_velocity, velocity, r_rot
    
def get_rifke(joints, r_rot):
    joints[..., 0] -= joints[:, 0:1, 0]
    joints[..., 2] -= joints[:, 0:1, 2]

    joints = qrot_np(np.repeat(r_rot[:, None], joints.shape[1], axis=1), joints)
    return joints

def process_file(pose, joints, feet_thre):
    global_trans = pose[:, :3]
    relative_rots_6d = pose[:, 3:].reshape(global_trans.shape[0], -1, 6)
    '''New ground truth positions'''
    global_positions = joints.clone().detach().cpu().numpy()
    feet_l, feet_r = foot_detect(joints.detach().cpu().numpy(), feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    r_velocity, velocity, r_rot = compute_cont6d_params(relative_rots_6d, joints)
    positions = get_rifke(joints.detach().cpu().numpy(), r_rot)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = relative_rots_6d[:, 1:].reshape(len(relative_rots_6d), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    if isinstance(rot_data, torch.Tensor):
        rot_data = rot_data.detach().cpu().numpy()
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


def back_process(pose, joints):
    data, ground_positions, positions, l_velocity = process_file(pose, joints, 0.002)
    return data[:, :67]
