import smplx
import torch
import torch.nn as nn
import pytorch3d.transforms as t3d
import numpy as np


class SMPLModel(nn.Module):
    def __init__(self, num_joints=24, smpl_model_path=None, device='cpu'):
        """
        Initialize the SMPLModel class.

        Args:
            smpl_model_path (str): Path to the SMPL model files.
            device (str, optional): Device to run the model on. Defaults to "cpu".
        """
        super(SMPLModel, self).__init__()
        if smpl_model_path is None:
            smpl_model_path = './data/smpl/SMPL_NEUTRAL.pkl'
        self.smpl_model_path = smpl_model_path
        self.device = device
        self.smpl_model = smplx.create(
            model_path=self.smpl_model_path,
            model_type='smpl',
            # batch_size=1,
        ).to(self.device)
        
        # Set SMPL model to evaluation mode for consistent inference
        self.smpl_model.eval()
        
        # Disable gradients for SMPL parameters to save memory
        for param in self.smpl_model.parameters():
            param.requires_grad = False
            
        self.num_joints = num_joints

    def ensure_eval_mode(self):
        """Ensure SMPL model is in evaluation mode."""
        if self.smpl_model.training:
            self.smpl_model.eval()
            print("Warning: SMPL model was in training mode, switched to eval mode")

    def check_input(self, data):
        """
        Check if the input data is valid and return the global translation and poses.
        Input data shape has to be one of the following:
            - 75: 3 (translation) + 24 * 3 (Euler joints)
            - 147: 3 (translation) + 24 * 6 (6D rep joints)
            - 219: 3 (translation) + 24 * 9 (rot mat joints)
            - 225: 9 (3 translation + 6 zero padding) + 24 * 9 (rot mat joints)
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
            raise ValueError(f"Expected input data to have shape (batch_size, D) where D is one of [75, 147, 219, 225], got shape {tuple(data.shape)}")
        
        if poses.shape[-1] in [144, 216]:
            if poses.ndim == 2:
                poses = poses.reshape(poses.shape[0], 24, -1)
            elif poses.ndim == 3:
                poses = poses.reshape(poses.shape[0], poses.shape[1], 24, -1)

        if poses.shape[-1] == 6:
            if poses.ndim == 3:
                poses = t3d.rotation_6d_to_matrix(poses).reshape(poses.shape[0], 24, -1)
            elif poses.ndim == 4:
                poses = t3d.rotation_6d_to_matrix(poses).reshape(poses.shape[0], poses.shape[1], 24, -1)
            else:
                raise ValueError(f"Expected poses shape (batch_size, 24, 6) or (batch_size, num_frames, 24, 6), got {poses.shape}")
        
        return global_translation, poses

    def forward(self, data, return_verts=False, skeleton=None, enable_grad=False):
        """
        Forward pass of the SMPL model.

        Args:
            data: Input pose data
            return_verts (bool): Whether to return vertices
            skeleton: Skeleton information (unused in SMPL)
            enable_grad (bool): If True, allows gradients to flow through SMPL forward pass
                               for loss computation while keeping SMPL parameters frozen.
                               If False, uses no_grad for pure inference.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_joints * 3) containing the joint locations.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(torch.float32)
        data = data.to(self.device)
        global_translation, poses = self.check_input(data)
        batch_size = poses.shape[0]

        reshaped = False
        if global_translation.ndim == 3:
            reshaped = True
            B, T = global_translation.shape[:2]
            global_translation = global_translation.reshape(B*T, 3)
            poses = poses.reshape(B*T, self.num_joints, -1)
        # # check input
        # assert global_translation.shape == (batch_size, 3), (
        #     f"Expected global_translation shape (batch_size, 3), got {global_translation.shape}"
        # )

        if poses.shape[-1] == 72:
            pose2rot = True
        elif poses.shape[-1] == 9:
            pose2rot = False
        else:
            raise ValueError(f"Expected poses shape (batch_size, 72) or (batch_size, 216), got {poses.shape}")

        if pose2rot:
            smpl_body_pose = poses[..., 3:]
            smpl_root_rot = poses[..., :3]
        else:
            smpl_body_pose = poses[:, 1:]
            smpl_root_rot = poses[:, :1]

        # Forward pass through the SMPL model - ensure eval mode
        self.ensure_eval_mode()
        
        # Choose whether to enable gradients based on use case
        if enable_grad:
            # Allow gradients to flow for loss computation, but SMPL params are frozen
            smpl_output = self.smpl_model.to(data.device)(
                betas=self.smpl_model.betas,
                global_orient=smpl_root_rot,
                body_pose=smpl_body_pose,
                transl=global_translation,
                batch_size=batch_size,
                return_verts=return_verts,
                pose2rot=pose2rot,
            )
        else:
            with torch.no_grad():
                smpl_output = self.smpl_model.to(data.device)(
                    betas=self.smpl_model.betas,
                    global_orient=smpl_root_rot,
                    body_pose=smpl_body_pose,
                    transl=global_translation,
                    batch_size=batch_size,
                    return_verts=return_verts,
                    pose2rot=pose2rot,
                )
        smpl_joints_loc = smpl_output.joints  # (batch_size, num_joints, 3)
        smpl_joints_loc = smpl_joints_loc[:, :24, :]
        if reshaped:
            smpl_joints_loc = smpl_joints_loc.reshape(B, T, self.num_joints, -1)
        if return_verts:
            smpl_verts = smpl_output.vertices  # (batch_size, num_verts, 3)
            if reshaped:
                smpl_verts = smpl_verts.reshape(B, T, smpl_verts.shape[1], -1)
            return smpl_joints_loc, smpl_verts
        else:
            return smpl_joints_loc
    

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

if __name__ == '__main__':
    POST_PROCESSING = False
    GT_COMPARE = False
    DATASET = 'HumanML3D'
    import os
    import numpy as np
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.skeleton.utility import get_keypoint_skeleton, get_motorica_skeleton_names, expand_skeleton
    from src.visualization.skeleton import visualize_pd_skeleton, create_video_from_keypoints, create_2D_video_from_keypoints
    from src.skeleton.preprocessing import motion_preprocessing, motion_postprocessing
    from src.utils.geometry import rot_6d_to_matrix
    from src.skeleton.forward_kinematics import ForwardKinematics

    if DATASET == 'motorica':
        root_path = './data/motorica/'
        id = 'kthjazz_gCH_sFM_cAll_d02_mCH_ch01_charlestonchaserswabashblues_004_chunk84_motion'
        # id = 'kthstreet_gLH_sFM_cAll_d01_mLH_ch01_feelingsvstech_001_chunk145_motion'
        # id = 'kthstreet_gPO_sFM_cAll_d02_mPO_ch01_justus2_002_chunk217_motion'
    elif DATASET == 'AIST':
        root_path = './data/AIST'
        id = 'gBR_sBM_cAll_d05_mBR0_ch04_chunk2_motion'
    elif DATASET == 'HumanML3D':
        root_path = './data/HumanML3D'
        id = '005706_chunk2_motion'

    # Data loading
    smpl_data = np.load(os.path.join(root_path, 'sliced_motion_smpl', f'{id}.npy'), allow_pickle=True)[()]
    smpl_kp_positions = smpl_data['motion']['motion_positions']
    smpl_motion_data = smpl_data['motion']['motion_data']

    if DATASET == 'motorica':
        motorica_data = np.load(os.path.join(root_path, 'sliced_motion', f'{id}.npy'), allow_pickle=True)[()]
        motorica_motion_data = motorica_data['motion']['motion_data']
        motorica_kp_positions = motorica_data['motion']['motion_positions']

    smpl_fk = SMPLModel(num_joints=24)
    smpl_motion_data, preprocessing = motion_preprocessing(smpl_motion_data, 
                                       data_type='tree', 
                                       method='face_forward', 
                                       dataset='smpl', 
                                       fk=smpl_fk)
    
    if POST_PROCESSING:
        smpl_motion_data = motion_postprocessing(smpl_motion_data, 
                                                type='face_forward', 
                                                data_type='tree', 
                                                preprocessing_R=preprocessing['rotation'], 
                                                preprocessing_t=preprocessing['translation'])
    smpl_kp = smpl_fk.forward(smpl_motion_data.reshape(150, -1))

    if GT_COMPARE and DATASET == 'motorica':
        skeleton = motorica_data['motion']['skeleton']
        motorica_fk = ForwardKinematics(skeleton, selected_joints=get_motorica_skeleton_names())
        motorica_motion_data, preprocessing = motion_preprocessing(motorica_motion_data, 
                                        data_type='tree', 
                                        method='face_forward', 
                                        dataset='motorica', 
                                        fk=motorica_fk)
        motorica_kp = motorica_fk.forward(motorica_motion_data.reshape(150, -1))
        create_video_from_keypoints(keypoints=smpl_kp, 
                                    gt=motorica_kp,
                                    output_path="./videos/smpl_visualization.mp4", 
                                    skeleton=get_keypoint_skeleton(), 
                                    link_type='smpl',
                                    gt_link_type='motorica',
                                    fps=smpl_data['current_fps'],
                                    max_frames=300,
                                    flipped=True)

    else:
        create_video_from_keypoints(keypoints=smpl_kp, 
                                    gt=smpl_kp_positions,
                                    output_path="./videos/smpl_visualization.mp4", 
                                    link_type='smpl',
                                    gt_link_type='smpl',
                                    fps=smpl_data['current_fps'],
                                    max_frames=300,
                                    flipped=True)