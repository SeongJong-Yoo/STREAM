# BSD License

# For fairmotion software

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by Ruilong Li

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import torch
import sys
sys.path.append("/fs/nexus-projects/PhysicsFall/editable_dance_project/src/losses")
from feature_extractor import utils as feat_utils

def get_motorica_joints_names():
    return [
        'Hips', 
        'Spine', 
        'LeftUpLeg', 
        'RightUpLeg', 
        'Spine1', 
        'LeftLeg',
        'RightLeg', 
        'Neck', 
        'LeftShoulder', 
        'RightShoulder', 
        'LeftFoot', 
        'RightFoot', 
        'Head', 
        'LeftArm', 
        'RightArm', 
        'LeftForeArm', 
        'RightForeArm', 
        'LeftHand', 
        'RightHand'
    ]

SMPL_to_Keypoint = {
    "root": "Hips",
    "lhip": "LeftUpLeg",
    "rhip": "RightUpLeg",
    "belly": "Spine",
    "lknee": "LeftLeg",
    "rknee": "RightLeg",
    "spine": "Spine",
    "lankle": "LeftFoot",
    "rankle": "RightFoot",
    "chest": "Spine1",
    "ltoes": None,
    "rtoes": None,
    "neck": "Neck",
    "linshoulder": None,
    "rinshoulder": None,
    "head": "Head",
    "lshoulder": "LeftArm",
    "rshoulder": "RightArm",
    "lelbow": "LeftForeArm",
    "relbow": "RightForeArm",
    "lwrist": "LeftHand",
    "rwrist": "RightHand",
    "lhand": None,
    "rhand": None,
}

# convert any number of SMPL joint name to the corresponding motorica joint name.
# if the joint name is not in the SMPL_to_Keypoint dictionary, it will be returned as is.
def convert_SMPL_to_motorica(joint_names):
    return [SMPL_to_Keypoint[j] if j in SMPL_to_Keypoint else j for j in joint_names]
    



def extract_manual_features(positions):
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3) 
    features = []
    f = ManualFeatures(positions)
    for _ in range(1, positions.shape[0]):
        pose_features = []
        pose_features.append(
            f.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8 * f.hl)
        )
        pose_features.append(
            f.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8 * f.hl)
        )
        pose_features.append(
            f.f_nplane("chest", "neck", "neck", "rwrist", 0.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("chest", "neck", "neck", "lwrist", 0.2 * f.hl)
        )
        pose_features.append(
            f.f_move("belly", "chest", "chest", "rwrist", 1.8 * f.hl)
        )
        pose_features.append(
            f.f_move("belly", "chest", "chest", "lwrist", 1.8 * f.hl)
        )
        pose_features.append(
            f.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110])
        )
        pose_features.append(
            f.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110])
        )
        pose_features.append(
            f.f_nplane(
                "lshoulder", "rshoulder", "lwrist", "rwrist", 2.5 * f.sw
            )
        )
        pose_features.append(
            f.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4 * f.hl)
        )
        pose_features.append(
            f.f_move("rwrist", "root", "lwrist", "root", 1.4 * f.hl)
        )
        pose_features.append(
            f.f_move("lwrist", "root", "rwrist", "root", 1.4 * f.hl)
        )
        pose_features.append(f.f_fast("rwrist", 2.5 * f.hl))
        pose_features.append(f.f_fast("lwrist", 2.5 * f.hl))
        # unfortunately, the following features are not available in the motorica dataset as we don't have toes
        # pose_features.append(
        #     f.f_plane("root", "lhip", "ltoes", "rankle", 0.38 * f.hl)
        # )
        # pose_features.append(
        #     f.f_plane("root", "rhip", "rtoes", "lankle", 0.38 * f.hl)
        # )
        pose_features.append(
            f.f_nplane("zero", "y_unit", "y_min", "rankle", 1.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("zero", "y_unit", "y_min", "lankle", 1.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("lhip", "rhip", "lankle", "rankle", 2.1 * f.hw)
        )
        pose_features.append(
            f.f_angle("rknee", "rhip", "rknee", "rankle", [0, 110])
        )
        pose_features.append(
            f.f_angle("lknee", "lhip", "lknee", "lankle", [0, 110])
        )
        pose_features.append(f.f_fast("rankle", 2.5 * f.hl))
        pose_features.append(f.f_fast("lankle", 2.5 * f.hl))
        pose_features.append(
            f.f_angle("neck", "root", "rshoulder", "relbow", [25, 180])
        )
        pose_features.append(
            f.f_angle("neck", "root", "lshoulder", "lelbow", [25, 180])
        )
        pose_features.append(
            f.f_angle("neck", "root", "rhip", "rknee", [50, 180])
        )
        pose_features.append(
            f.f_angle("neck", "root", "lhip", "lknee", [50, 180])
        )
        pose_features.append(
            f.f_plane("rankle", "neck", "lankle", "root", 0.5 * f.hl)
        )
        pose_features.append(
            f.f_angle("neck", "root", "zero", "y_unit", [70, 110])
        )
        pose_features.append(
            f.f_nplane("zero", "minus_y_unit", "y_min", "rwrist", -1.2 * f.hl)
        )
        pose_features.append(
            f.f_nplane("zero", "minus_y_unit", "y_min", "lwrist", -1.2 * f.hl)
        )
        pose_features.append(f.f_fast("root", 2.3 * f.hl))
        features.append(pose_features)
        f.next_frame()
    features = torch.tensor(features, dtype=torch.float32).mean(axis=0)
    return features


class ManualFeatures:
    def __init__(self, positions, joint_names=get_motorica_joints_names()):
        self.positions = positions
        self.joint_names = joint_names
        self.frame_num = 1

        # humerus length
        self.hl = feat_utils.distance_between_points(
            [ 3.7924845e-02,  5.1255649e-01, -5.0576066e-04],  # "lshoulder" "LeftShoulder",
            [0.449883, 0.512556, -0.000499],  # LeftForeArm "lelbow"
        )
        # shoulder width
        self.sw = feat_utils.distance_between_points(
            [0.037925, 0.512556, -0.000506],  # "lshoulder"
            [-0.037939, 0.512556, -0.000506],  # "rshoulder"
        )
        # hip width
        self.hw = feat_utils.distance_between_points(
            [0.094875, 0.000000, 0.000000],  # "lhip"
            [-0.094875, 0.000000, 0.000000],  # "rhip"
        )

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        if j == "y_unit":
            return [0, 1, 0]
        elif j == "minus_y_unit":
            return [0, -1, 0]
        elif j == "zero":
            return [0, 0, 0]
        elif j == "y_min":
            return [
                0,
                min(
                    [y for (_, y, _) in self.positions[self.frame_num]]
                ),
                0,
            ]
        return self.positions[self.frame_num][
            self.joint_names.index(j)
        ]

    def transform_and_fetch_prev_position(self, j):
        return self.positions[self.frame_num - 1][
            self.joint_names.index(j)
        ]

    def f_move(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = convert_SMPL_to_motorica([j1, j2, j3, j4])
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.velocity_direction_above_threshold(
            j1, j1_prev, j2, j2_prev, j3, j3_prev, range
        )

    def f_nmove(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = convert_SMPL_to_motorica([j1, j2, j3, j4])
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.velocity_direction_above_threshold_normal(
            j1, j1_prev, j2, j3, j4, j4_prev, range
        )

    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = convert_SMPL_to_motorica([j1, j2, j3, j4])
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.distance_from_plane(j1, j2, j3, j4, threshold)

    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = convert_SMPL_to_motorica([j1, j2, j3, j4])
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.distance_from_plane_normal(j1, j2, j3, j4, threshold)

    def f_angle(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = convert_SMPL_to_motorica([j1, j2, j3, j4])
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.angle_within_range(j1, j2, j3, j4, range)

    def f_fast(self, j1, threshold):
        j1 = convert_SMPL_to_motorica([j1])[0]
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return feat_utils.velocity_above_threshold(j1, j1_prev, threshold)


if __name__ == "__main__":
    def expand_joint_name(joint_name):
        return [joint_name + "_Xposition", joint_name + "_Yposition", joint_name + "_Zposition"]
    
    def print_format_for_array(array):
        return '[' + ', '.join([f'{x:.6f}' for x in array]) + ']'

    sys.path.append("/fs/nexus-projects/PhysicsFall/editable_dance_project")
    from src.skeleton.forward_kinematics import ForwardKinematics
    from src.visualization.skeleton import visualize_pd_skeleton
    from matplotlib import pyplot as plt

    fk = ForwardKinematics(normalized_skeleton_path = "/fs/nexus-projects/PhysicsFall/editable_dance_project/data/normalized_skeleton.pkl")
    dummpy_inputs = torch.zeros((1,60))
    positions = fk(dummpy_inputs)
    pos_df = fk.convert_to_dataframe(positions)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    visualize_pd_skeleton(ax,0,pos_df)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=90, azim=-90)
    # plt.savefig("rest_pose.png")
    # exit()

    l_shoulder_name = 'LeftShoulder'
    l_elbow_name = 'LeftForeArm'
    r_shoulder_name = 'RightShoulder'
    l_hip_name = "LeftUpLeg"
    r_hip_name = "RightUpLeg"

    l_shoulder_position = pos_df.loc[0, expand_joint_name(l_shoulder_name)].to_numpy()
    l_elbow_position = pos_df.loc[0, expand_joint_name(l_elbow_name)].to_numpy()
    r_shoulder_position = pos_df.loc[0, expand_joint_name(r_shoulder_name)].to_numpy()
    l_hip_position = pos_df.loc[0, expand_joint_name(l_hip_name)].to_numpy()
    r_hip_position = pos_df.loc[0, expand_joint_name(r_hip_name)].to_numpy()
    
    print(f'{l_shoulder_name} position: {print_format_for_array(l_shoulder_position)}')
    print(f'{l_elbow_name} (left elbow) position: {print_format_for_array(l_elbow_position)}')
    print(f'{r_shoulder_name} position: {print_format_for_array(r_shoulder_position)}')
    print(f'{l_hip_name} (left hip) position: {print_format_for_array(l_hip_position)}')
    print(f'{r_hip_name} (right hip) position: {print_format_for_array(r_hip_position)}')

    from pathlib import Path
    file_path = Path("/fs/nexus-projects/PhysicsFall/data/AIST++/prediction_result/gBR_sBM_cAll_d04_mBR3_ch09_motorica.npy")
    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist.")
    
    data_dict = np.load(file_path, allow_pickle=True).item()
    motion_positions = data_dict["motion_positions"].squeeze().cpu()
    print(f'motion_positions shape: {motion_positions.shape}')
    fps = data_dict["fps"]
    kinetic_feature_vector = extract_manual_features(motion_positions)
    print(f'kinetic_feature_vector shape: {kinetic_feature_vector.shape}')





