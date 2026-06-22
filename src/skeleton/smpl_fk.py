# SMPL FK Model adapted from https://github.com/Stanford-TML/EDGE?tab=readme-ov-file

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import pytorch3d.transforms as t3d
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply)
from tqdm import tqdm

smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]

smpl_parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]


def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


class SMPLModel(nn.Module):
    def __init__(self, num_joints=24, smpl_model_path=None, device='cpu'):
        """
        Quaternion-based forward kinematics for SMPL skeleton.

        Args:
            num_joints (int): Number of joints (default 24).
            smpl_model_path: Unused, kept for interface compatibility.
            device (str): Device to place offsets on.
        """
        super().__init__()
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        self.register_buffer('_offsets', torch.Tensor(offsets))
        self._parents = np.array(parents)
        self.num_joints = num_joints
        self._compute_metadata()

        if device != 'cpu':
            self.to(device)

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def check_input(self, data):
        """
        Parse flat input into global translation and per-joint axis-angle rotations.
        Supported last-dimension sizes:
            - 75:  3 (translation) + 24 * 3 (axis-angle)
            - 147: 3 (translation) + 24 * 6 (6D rotation)
            - 219: 3 (translation) + 24 * 9 (rotation matrix)
            - 225: 9 (3 translation + 6 padding) + 24 * 9 (rotation matrix)

        Returns:
            global_translation: (..., 3)
            rotations: (..., 24, 3) axis-angle
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(torch.float32)

        if data.shape[-1] in [75, 147, 219]:
            global_translation = data[..., :3]
            poses = data[..., 3:]
        elif data.shape[-1] == 225:
            global_translation = data[..., :3]
            poses = data[..., 9:]
        else:
            raise ValueError(
                f"Expected last dim to be one of [75, 147, 219, 225], "
                f"got shape {tuple(data.shape)}"
            )

        feat_per_joint = poses.shape[-1] // self.num_joints
        poses = poses.reshape(*poses.shape[:-1], self.num_joints, feat_per_joint)

        if feat_per_joint == 3:
            rotations = poses
        elif feat_per_joint == 6:
            rot_mat = t3d.rotation_6d_to_matrix(poses)
            rotations = t3d.matrix_to_axis_angle(rot_mat)
        elif feat_per_joint == 9:
            rot_mat = poses.reshape(*poses.shape[:-1], 3, 3)
            rotations = t3d.matrix_to_axis_angle(rot_mat)
        else:
            raise ValueError(
                f"Unsupported per-joint feature dim {feat_per_joint}. "
                f"Expected 3 (axis-angle), 6 (6D), or 9 (rotation matrix)."
            )

        return global_translation, rotations

    def _forward_kinematics(self, rotations, root_positions):
        """
        Quaternion-based forward kinematics.
        Args:
            rotations:      (N, L, J, 3) axis-angle
            root_positions: (N, L, 3)
        Returns:
            (N, L, J, 3) world joint positions
        """
        self._offsets = self._offsets.to(rotations.device)
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

    def forward(self, data, return_verts=False, skeleton=None, enable_grad=False):
        """
        Forward pass: parse flat input and run FK.

        Args:
            data: (T, D) or (B, T, D) pose data.
            return_verts (bool): If True, returns (joints, vertices) — vertices
                are not available without a mesh model, so NotImplementedError is raised.
            skeleton: Unused, kept for interface compatibility.
            enable_grad (bool): If True, allows gradients through the FK pass.

        Returns:
            (T, J, 3) or (B, T, J, 3) joint positions.
        """
        if return_verts:
            raise NotImplementedError(
                "Vertex output requires a mesh model (e.g. smplx). "
                "Use return_verts=False for joint positions only."
            )

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(torch.float32)
        # data = data.to(self._offsets.device)

        global_translation, rotations = self.check_input(data)

        added_batch = False
        if global_translation.ndim == 2:
            added_batch = True
            global_translation = global_translation.unsqueeze(0)
            rotations = rotations.unsqueeze(0)

        if enable_grad:
            joint_positions = self._forward_kinematics(rotations, global_translation)
        else:
            with torch.no_grad():
                joint_positions = self._forward_kinematics(rotations, global_translation)

        if added_batch:
            joint_positions = joint_positions.squeeze(0)

        return joint_positions

    def get_joints_name(self):
        return [
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hand",
            "right_hand",
        ]

