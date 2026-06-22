"""Standalone forward kinematics for keypoint data"""
import torch
import numpy as np
import pandas as pd
from pytorch3d.transforms import euler_angles_to_matrix
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.skeleton.utility import get_keypoint_skeleton, get_motorica_skeleton_names, expand_skeleton
from src.visualization.skeleton import visualize_pd_skeleton, create_video_from_keypoints
from src.skeleton.preprocessing import motion_preprocessing, motion_postprocessing
from src.utils.geometry import rot_6d_to_matrix


class ForwardKinematics(nn.Module):
    def __init__(self, skeleton = None, selected_joints = None, normalized_skeleton_path = "./data/normalized_skeleton.pkl"):
        super(ForwardKinematics, self).__init__()
        if skeleton:
            self.skeleton = skeleton
        else:
            self.skeleton = get_keypoint_skeleton(normalized_skeleton_path)

        if selected_joints is not None:
            self.joint_names = selected_joints
        else:
            self.joint_names = self.skeleton.keys()

        self._update_skeleton(self.skeleton)
        
    def _update_skeleton(self, skeleton):
        self.skeleton = skeleton
        (
            self.joint_names,
            self.parents,
            self.offsets,
            self.rotation_orders,
            self.has_position,
        ) = self._parse_skeleton(skeleton)

    def get_joint_order(self):
        return self.joint_names
    

    def _parse_skeleton(self, skeleton):
        # Find root joint
        root = "Hips"

        # Traversal order (BFS)
        joint_names = []
        parents = []
        offsets = []
        rotation_orders = []
        has_position = []
        parent_map = {}

        queue = deque([root])
        while queue:
            joint = queue.popleft()
            if joint not in self.joint_names:
                continue
            joint_names.append(joint)
            info = skeleton[joint]

            # Parent index
            if info["parent"] is None:
                parents.append(-1)
            else:
                parents.append(parent_map[info["parent"]])

            # Store parent index for children
            parent_map[joint] = len(joint_names) - 1

            # Offset
            offsets.append(torch.tensor(info["offsets"], dtype=torch.float32))

            # Rotation order
            rotation_orders.append(info["order"].upper())

            # Position channels
            has_position.append("Xposition" in info["channels"])

            # Add children to queue
            queue.extend(info["children"])

        return (
            joint_names,
            parents,
            torch.stack(offsets),
            rotation_orders,
            has_position,
        )
    

    def _df_to_tensors(self, df):
        # joint_order = ['Hips', 'RightUpLeg','RightLeg','RightFoot','RightToeBase', 'LeftUpLeg', 'LeftLeg','LeftFoot','LeftToeBase', 'Spine','Spine1', 'RightShoulder','RightArm','RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'Neck', 'Head']
        joint_order = get_motorica_skeleton_names()
        selected_joints = ["Hips_Xposition",  "Hips_Yposition",  "Hips_Zposition"] + expand_skeleton(joint_order)
        df = df[selected_joints]
        return torch.tensor(df.values, dtype=torch.float32)


    def forward_df(self, df):
        # TODO: something wrong with this function
        raise NotImplementedError("forward_df is not implemented yet")
        data_tensor = self._df_to_tensors(df)
        print(f'data_tensor shape: {data_tensor.shape}')
        return self.forward(data=data_tensor)

    def _fill_dummy_joints(self, data):
        num_frames = data.shape[0]
        pos = data[:, :3]
        rot = data[:, 3:]
        if rot.shape[-1] % 15 != 0:
            # print(f"Turn off 'fill_dummy at forward' because rot.shape[-1] % 15 != 0: {rot.shape[-1]}")
            return data
        rot = rot.reshape(num_frames, 15, -1)

        if rot.shape[-1] != 3 and rot.shape[-1] != 6: 
            return data

        dim = rot.shape[-1]
        dummy = torch.ones((num_frames, 1, dim), dtype=rot.dtype, device=rot.device)
        rot = torch.cat([rot[:, :10, :], dummy.expand(-1, 2, -1), rot[:, 10:13, :], dummy.expand(-1, 2, -1), rot[:, 13:, :], dummy.expand(-1, 2, -1)], dim=1).reshape(num_frames, -1)

        return torch.cat([pos, rot], dim=-1)
    
    def check_input(self, data, fill_dummy=False):
        """
        With fill_dummy = True, expected data input shapes has 14 joints. 
        expected data input shapes: (batch_size x num_frames, D) where D is one of [66, 129, 192, 198]
        66 = 3 (translation) + 21 * 3 (Euler joints)
        129 = 3 (translation) + 21 * 6 (6D rep joints)
        192 = 3 (translation) + 21 * 9 (rot mat joints)  
        198 = 9 (3 translation + 6 zero padding) + 21 * 9 (rot mat joints)
        """
        if fill_dummy:
            data = self._fill_dummy_joints(data)

        seq_length, feature_size  = data.shape
        expected_dims = [66, 129, 192, 198]
        if data.dim() != 2 or feature_size not in expected_dims:
            raise ValueError(f"Expected input data to have shape (batch_size x num_frames, D) where D is one of {expected_dims}, got shape {tuple(data.shape)}")
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        
        if feature_size == 198:
            pos = data[:, :3]
            rot = data[:, 9:].reshape(seq_length, 21, -1)
        else:
            pos = data[:, :3]
            rot = data[:, 3:].reshape(seq_length, 21, -1)


        if rot.shape[-1] == 3:
            rot = euler_angles_to_matrix(rot, self.rotation_orders[0])
        elif rot.shape[-1] == 6:
            rot = rot_6d_to_matrix(rot)
        elif rot.shape[-1] == 9:
            rot = rot.reshape(seq_length, 21, 3, 3)
        else:
            raise ValueError(f"Expected rotation tensor to have shape (num_frames, 19, 3, 3), got {rot.shape}")
        
        identity = torch.eye(3).repeat(seq_length, 1, 1)
        rot[:, 10] = identity
        rot[:, 11] = identity
        rot[:, 15] = identity
        rot[:, 16] = identity
        rot[:, 19] = identity
        rot[:, 20] = identity
        

        return pos, rot



    # A differentiable forward kinematics function
    def forward(self, data: torch.Tensor, fill_dummy: bool=False, adjust=None, skeleton=None):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if skeleton:
            self._update_skeleton(skeleton)
        pos, rot_values = self.check_input(data, fill_dummy=fill_dummy)
        num_frames = pos.shape[0]
        num_joints = len(self.joint_names)
        device = pos.device
        dtype = pos.dtype

        
        global_pos_list = []
        global_rot_list = []
        self.offsets = self.offsets.to(device=device, dtype=dtype)
        for j, joint in enumerate(self.joint_names): # we have parsed joint name in hierarchy order
            if joint == "Hips":
                assert j == 0, f"Expected root joint to be at index 0, got {j}"
                # processing root joint
                global_pos_list.append(pos)
                global_rot_list.append(rot_values[:, j,:])
            else:
                # processing other joints
                parent = self.parents[j]
                # compute rotations
                parent_rot = global_rot_list[parent]
                global_rot = torch.bmm(parent_rot, rot_values[:, j,:,:]) # R_G = R_P * R_L
                global_rot_list.append(global_rot)

                # compute positions
                batch_size = pos.shape[0]
                #pos shape:(num_frame, 3), offset shape:(3,)
                local_pose = self.offsets[j].expand(batch_size, -1)
                #parent_rot: torch.Size([num_frame, 3, 3]); local_pose: torch.Size([num_frame, 3])
                rotated_offset = torch.bmm(global_rot_list[parent], local_pose.unsqueeze(-1)).squeeze(-1) 
                global_pos = global_pos_list[parent] + rotated_offset
                global_pos_list.append(global_pos)

        global_pos = torch.stack(global_pos_list, dim=1) # joint order in self.joint_names
        return global_pos
    
    def convert_to_dataframe(self, positions):
        """Convert output tensor back to DataFrame format"""
        """position shape: (num_frames, num_joints (19), 3)"""
        columns = []
        data = {}
        if isinstance(positions, torch.Tensor):
            if positions.device.type == 'cuda':
                positions = positions.detach().cpu()
            positions = positions.numpy()
        # check position shape
        assert positions.shape[1] == len(self.joint_names), f"Expected position shape to be (num_frames, num_joints (21)), got {positions.shape}"

        for j, joint in enumerate(self.joint_names):
            pos = positions[:, j]
            data[f"{joint}_Xposition"] = pos[:, 0]
            data[f"{joint}_Yposition"] = pos[:, 1]
            data[f"{joint}_Zposition"] = pos[:, 2]
        return pd.DataFrame(data)

def main():
    from copy import deepcopy
    # data_dir = Path("./data/AIST_beats/sliced_motion/gBR_sFM_cAll_d04_mBR2_ch03_chunk13_motion.npy")
    # data_dir = Path("./data/motorica/sliced_motion/kthstreet_gPO_sFM_cAll_d01_mPO_ch01_luco100bpm_001_chunk66_motion.npy")
    data_dir = Path("./data/motorica/sliced_motion/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_chunk0_motion.npy")
    # data_dir = Path("./data/motorica_beats_10sec/sliced_motion/kthstreet_gLH_sFM_cAll_d02_mLH_ch01_yougonnaregretit_002_chunk2_motion.npy")
    # data_dir = Path("./kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.npy")
    # data_dir = Path("./data/motorica_beats/sliced_motion/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_chunk0_motion.npy")
    # data_dir = Path("/mnt/hdd/Dataset/Motorica/motorica_dance_dataset/synced_motion/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.npy")
    # data_dir = Path("./beacon.npy")
    # data_dir = Path("/mnt/hdd/Dataset/Motorica/motorica_dance_dataset/synced_motion/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.npy")
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found.")
    
    data = np.load(data_dir, allow_pickle=True)[()]
    # current_fps = data['fps']
    current_fps = data['current_fps']
    data = data['motion']
    file_name = data["file_name"]
    skeleton = data["skeleton"]
    if "scale" in data:
        scale = data["scale"]
    else:
        scale = 1.0
    motion_data = data["motion_data"]
    motion_data_order = data["motion_data_order"]
    motion_positions = data["motion_positions"]
    motion_positions_order = data["motion_positions_order"]

    fk = ForwardKinematics(skeleton, selected_joints=get_motorica_skeleton_names())
    gt = deepcopy(motion_data)
    motion_data = motion_preprocessing(motion_data, 
                                       data_type='tree', 
                                       method='face_forward', 
                                       dataset='motorica', 
                                       fk=fk, 
                                       skeleton=skeleton)
    
    motion_data = motion_postprocessing(motion_data[np.newaxis, :, :], type='face_forward', data_type='tree').squeeze(0)

    keypoint_data = torch.tensor(motion_data, dtype=torch.float32)
    # if len(keypoint_data.shape) == 3:
    #     print(f'keypoint_data shape: {keypoint_data.shape}') # (num_frames, 20, 3)
    #     keypoint_data = keypoint_data.reshape(-1, 60)
    keypoint_loc = fk.forward(keypoint_data.reshape(keypoint_data.shape[0], -1), fill_dummy=True, skeleton=skeleton)
    # print(f'keypoint_loc shape: {keypoint_loc.shape}') # (num_frames, 19, 3)
    # convert to df for visualization
    keypoint_df = fk.convert_to_dataframe(keypoint_loc)

    gt_data = torch.tensor(gt, dtype=torch.float32)
    gt_data = gt_data.reshape(gt.shape[0], -1)
    gt_loc = fk.forward(gt_data, fill_dummy=True, skeleton=skeleton)
    gt_df = fk.convert_to_dataframe(gt_loc)

    # print(f'keypoint_df: {keypoint_df.head()}') # (num_frames, 57)

    downsample_rate = 1
    create_video_from_keypoints(keypoints=keypoint_loc.numpy()[::downsample_rate], 
                                output_path="./videos/keypoint_visualization.mp4", 
                                skeleton=skeleton, 
                                link_type='motorica',
                                fps=current_fps//downsample_rate,
                                max_frames=600,
                                flipped=True,
                                gt=gt_loc[::downsample_rate]
                                )
    print("Keypoint visualization video saved to ./videos/keypoint_visualization.mp4")

    # # # visualize a frame
    # frame_num = 0
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax = visualize_pd_skeleton(ax, frame_num, keypoint_df, skeleton=skeleton, gt=gt_df)
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    # plt.savefig("./images/keypoint_visualization.png")
    # plt.close()
    # print("Keypoint visualization saved to ./images/keypoint_visualization.png")

if __name__ == "__main__":
    main()