import numpy as np
import pandas as pd
import pickle
from pathlib import Path

def get_keypoint_skeleton(skeleton_path = "/fs/nexus-projects/PhysicsFall/editable_dance_project/data/normalized_skeleton.pkl"):
    skeleton_path = Path(skeleton_path)
    if not skeleton_path.exists():
        raise FileNotFoundError(f"{skeleton_path} not found. Please run the script 'compute_normalize_skeleton.py' to generate it.")
    with open(skeleton_path, 'rb') as f:
        skeleton = pickle.load(f)
    return skeleton["skeleton"]

def get_motorica_skeleton_names():
    return [
        'Hips',             # 0
        'Spine',            # 1
        'LeftUpLeg',        # 2
        'RightUpLeg',       # 3
        'Spine1',           # 4
        'LeftLeg',          # 5
        'RightLeg',         # 6
        'Neck',             # 7
        'LeftShoulder',     # 8
        'RightShoulder',    # 9
        'LeftFoot',         # 10 -> Redundant
        'RightFoot',        # 11 -> Redundant
        'Head',             # 12
        'LeftArm',          # 13
        'RightArm',         # 14
        'LeftToeBase',      # 15 -> Redundant
        'RightToeBase',     # 16 -> Redundant
        'LeftForeArm',      # 17
        'RightForeArm',     # 18
        'LeftHand',         # 19 -> Redundant
        'RightHand',        # 20 -> Redundant
    ]


def expand_skeleton(skeleton: list, order = "XYZ"):
    """
    Expands a list of skeleton joints into a list of joint-axis combinations.

    Each joint in the input list is expanded into three elements, one for each
    axis of rotation (X, Y, Z).

    Args:
        skeleton (list): A list of joint names.

    Returns:
        list: A list of joint-axis combinations in the format "{joint}_{axis}rotation".
    """
    # check if the order is valid
    if len(order) != 3:
        raise ValueError("The order must be a string of length 3")
    if not all([axis in "XYZ" for axis in order]):
        raise ValueError("The order must contain only 'X', 'Y', and 'Z'")
    expanded_skeleton = [
        f"{joint}_{axis}rotation" for joint in skeleton for axis in order
    ]
    return expanded_skeleton