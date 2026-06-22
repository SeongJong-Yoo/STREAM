
import numpy as np


def distance_between_points(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def distance_from_plane(a, b, c, p, threshold):
    ba = np.array(b) - np.array(a)
    ca = np.array(c) - np.array(a)
    cross = np.cross(ca, ba)

    pa = np.array(p) - np.array(a)
    return np.dot(cross, pa) / np.linalg.norm(cross) > threshold


def distance_from_plane_normal(n1, n2, a, p, threshold):
    normal = np.array(n2) - np.array(n1)
    pa = np.array(p) - np.array(a)
    return np.dot(normal, pa) / np.linalg.norm(normal) > threshold


def angle_within_range(j1, j2, k1, k2, range):
    j = np.array(j2) - np.array(j1)
    k = np.array(k2) - np.array(k1)

    angle = np.arccos(np.dot(j, k) / (np.linalg.norm(j) * np.linalg.norm(k)))
    angle = np.degrees(angle)

    if angle > range[0] and angle < range[1]:
        return True
    else:
        return False


def velocity_direction_above_threshold(
    j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1 / 120
):
    velocity = (
        np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    )
    direction = np.array(j2) - np.array(j1)

    velocity_along_direction = np.dot(velocity, direction) / np.linalg.norm(
        direction
    )
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_direction_above_threshold_normal(
    j1, j1_prev, j2, j3, p, p_prev, threshold, time_per_frame=1 / 120
):
    velocity = (
        np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    )
    j31 = np.array(j3) - np.array(j1)
    j21 = np.array(j2) - np.array(j1)
    direction = np.cross(j31, j21)

    velocity_along_direction = np.dot(velocity, direction) / np.linalg.norm(
        direction
    )
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_above_threshold(p, p_prev, threshold, time_per_frame=1 / 120):
    velocity = np.linalg.norm(np.array(p) - np.array(p_prev)) / time_per_frame
    return velocity > threshold


def calc_average_velocity(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    return np.linalg.norm(average_velocity / (current_window * frame_time))


def calc_average_acceleration(
    positions, i, joint_idx, sliding_window, frame_time
):
    current_window = 0
    average_acceleration = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j + 1 >= len(positions):
            continue
        v2 = (
            positions[i + j + 1][joint_idx] - positions[i + j][joint_idx]
        ) / frame_time
        v1 = (
            positions[i + j][joint_idx]
            - positions[i + j - 1][joint_idx] / frame_time
        )
        average_acceleration += (v2 - v1) / frame_time
        current_window += 1
    return np.linalg.norm(average_acceleration / current_window)


def calc_average_velocity_horizontal(
    positions, i, joint_idx, sliding_window, frame_time, up_vec="z"
):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    if up_vec == "y":
        average_velocity = np.array(
            [average_velocity[0], average_velocity[2]]
        ) / (current_window * frame_time)
    elif up_vec == "z":
        average_velocity = np.array(
            [average_velocity[0], average_velocity[1]]
        ) / (current_window * frame_time)
    else:
        raise NotImplementedError
    return np.linalg.norm(average_velocity)


def calc_average_velocity_vertical(
    positions, i, joint_idx, sliding_window, frame_time, up_vec
):
    current_window = 0
    average_velocity = np.zeros(len(positions[0][joint_idx]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    if up_vec == "y":
        average_velocity = np.array([average_velocity[1]]) / (
            current_window * frame_time
        )
    elif up_vec == "z":
        average_velocity = np.array([average_velocity[2]]) / (
            current_window * frame_time
        )
    else:
        raise NotImplementedError
    return np.linalg.norm(average_velocity)



def extract_kinetic_features(positions):
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3) 
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        feature_vector = np.hstack(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = np.array(kinetic_feature_vector, dtype=np.float32)
    return kinetic_feature_vector


class KineticFeatures:
    def __init__(
        self, positions, frame_time=1./60, up_vec="y", sliding_window=2
    ):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint):
        average_kinetic_energy = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
            average_kinetic_energy += average_velocity ** 2
        average_kinetic_energy = average_kinetic_energy / (
            len(self.positions) - 1.0
        )
        return average_kinetic_energy

    def average_kinetic_energy_horizontal(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_horizontal(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_kinetic_energy_vertical(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = calc_average_velocity_vertical(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_energy_expenditure(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            val += calc_average_acceleration(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
        val = val / (len(self.positions) - 1.0)
        return val

if __name__ == "__main__":
    from pathlib import Path
    file_path = Path("/fs/nexus-projects/PhysicsFall/data/AIST++/prediction_result/gBR_sBM_cAll_d04_mBR3_ch09_motorica.npy")
    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist.")
    
    data_dict = np.load(file_path, allow_pickle=True).item()
    motion_positions = data_dict["motion_positions"].squeeze()
    print(f'motion_positions shape: {motion_positions.shape}')
    fps = data_dict["fps"]
    motion_positions = motion_positions.cpu().numpy()
    kinetic_feature_vector = extract_kinetic_features(motion_positions)
    print(f'kinetic_feature_vector shape: {kinetic_feature_vector.shape}')