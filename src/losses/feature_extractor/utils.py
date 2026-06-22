# BSD License

# For fairmotion software

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

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
import torch


def distance_between_points(a, b):
    return torch.linalg.norm(torch.tensor(a) - torch.tensor(b))


def distance_from_plane(a, b, c, p, threshold):
    ba = torch.tensor(b) - torch.tensor(a)
    ca = torch.tensor(c) - torch.tensor(a)
    cross = torch.cross(ca, ba)

    pa = torch.tensor(p) - torch.tensor(a)
    return torch.dot(cross, pa) / torch.linalg.norm(cross) > threshold


def distance_from_plane_normal(n1, n2, a, p, threshold):
    normal = torch.tensor(n2, dtype=torch.float32) - torch.tensor(n1,dtype=torch.float32)
    pa = torch.tensor(p) - torch.tensor(a)
    return torch.dot(normal, pa) / torch.linalg.norm(normal) > threshold


def angle_within_range(j1, j2, k1, k2, range):
    j = torch.tensor(j2, dtype=torch.float32) - torch.tensor(j1, dtype=torch.float32)
    k = torch.tensor(k2, dtype=torch.float32) - torch.tensor(k1, dtype=torch.float32)

    angle = torch.acos(torch.dot(j, k) / (torch.linalg.norm(j) * torch.linalg.norm(k)))
    angle = torch.rad2deg(angle)

    return range[0] < angle < range[1]


def velocity_direction_above_threshold(
    j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1 / 120
):
    velocity = (
        torch.tensor(p) - torch.tensor(j1) - (torch.tensor(p_prev) - torch.tensor(j1_prev))
    )
    direction = torch.tensor(j2) - torch.tensor(j1)

    velocity_along_direction = torch.dot(velocity, direction) / torch.linalg.norm(
        direction
    )
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_direction_above_threshold_normal(
    j1, j1_prev, j2, j3, p, p_prev, threshold, time_per_frame=1 / 120
):
    velocity = (
        torch.tensor(p) - torch.tensor(j1) - (torch.tensor(p_prev) - torch.tensor(j1_prev))
    )
    j31 = torch.tensor(j3) - torch.tensor(j1)
    j21 = torch.tensor(j2) - torch.tensor(j1)
    direction = torch.cross(j31, j21)

    velocity_along_direction = torch.dot(velocity, direction) / torch.linalg.norm(
        direction
    )
    velocity_along_direction = velocity_along_direction / time_per_frame
    return velocity_along_direction > threshold


def velocity_above_threshold(p, p_prev, threshold, time_per_frame=1 / 120):
    velocity = torch.linalg.norm(torch.tensor(p) - torch.tensor(p_prev)) / time_per_frame
    return velocity > threshold


def calc_average_velocity(positions, i, joint_idx, sliding_window, frame_time):
    current_window = 0
    device = positions.device
    average_velocity = torch.zeros(len(positions[0][joint_idx])).to(device)
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    return torch.linalg.norm(average_velocity / (current_window * frame_time))


def calc_average_acceleration(
    positions, i, joint_idx, sliding_window, frame_time
):
    current_window = 0
    device = positions.device
    average_acceleration = torch.zeros(len(positions[0][joint_idx])).to(device)
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
    return torch.linalg.norm(average_acceleration / current_window)


def calc_average_velocity_horizontal(
    positions, i, joint_idx, sliding_window, frame_time, up_vec="z"
):
    current_window = 0
    device = positions.device
    average_velocity = torch.zeros(len(positions[0][joint_idx])).to(device)
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (positions[i + j][joint_idx] - positions[i + j - 1][joint_idx])
        current_window += 1
    if up_vec == "y":
        average_velocity = torch.cat(
            (average_velocity[0].unsqueeze(0), average_velocity[2].unsqueeze(0))
        ) / (current_window * frame_time)
    elif up_vec == "z":
        average_velocity = torch.cat(
            (average_velocity[0].unsqueeze(0), average_velocity[1].unsqueeze(0))
        ) / (current_window * frame_time)
    else:
        raise NotImplementedError
    return torch.linalg.norm(average_velocity)


def calc_average_velocity_vertical(
    positions, i, joint_idx, sliding_window, frame_time, up_vec
):
    current_window = 0
    device = positions.device
    average_velocity = torch.zeros(len(positions[0][joint_idx])).to(device)
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        average_velocity += (
            positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        )
        current_window += 1
    if up_vec == "y":
        average_velocity = average_velocity[1].unsqueeze(0) / (
            current_window * frame_time
        )
    elif up_vec == "z":
        average_velocity = average_velocity[2].unsqueeze(0) / (
            current_window * frame_time
        )
    else:
        raise NotImplementedError
    return torch.linalg.norm(average_velocity)