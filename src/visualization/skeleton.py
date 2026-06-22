import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from .utility import LINK_AIST, LINK_MOTORICA, LINK_SMPL
import sys
from pathlib import Path
import os
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import soundfile as sf
from tqdm import tqdm
from copy import deepcopy
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.skeleton.utility import get_keypoint_skeleton, get_motorica_skeleton_names

def plot3d_keypoints(ax, keypoints, gt=None, frame=0, title='Keypoints', link_type='aist', flipped=False):
    """
    Plot the keypoints at a specific frame.
    """
    if link_type == 'aist':
        link = LINK_AIST
    elif link_type == 'motorica':
        link = LINK_MOTORICA
    else:
        raise ValueError(f"Link type {link_type} not supported")
    
    # Calculate global min and max values across all frames
    global_min_x = np.min(keypoints[:, :, 0])
    global_max_x = np.max(keypoints[:, :, 0])
    global_min_y = np.min(keypoints[:, :, 1])
    global_max_y = np.max(keypoints[:, :, 1])
    global_min_z = np.min(keypoints[:, :, 2])
    global_max_z = np.max(keypoints[:, :, 2])
    
    # Calculate global range and midpoints
    global_max_range = np.array([
        global_max_x - global_min_x,
        global_max_y - global_min_y,
        global_max_z - global_min_z
    ]).max() / 2.0
    
    global_mid_x = (global_max_x + global_min_x) * 0.5
    global_mid_y = (global_max_y + global_min_y) * 0.5
    global_mid_z = (global_max_z + global_min_z) * 0.5

    if flipped:
        ax.scatter(keypoints[frame, :, 0], keypoints[frame, :, 2], keypoints[frame, :, 1], c='r', marker='o', label='Predicted')
    else:
        ax.scatter(keypoints[frame, :, 0], keypoints[frame, :, 1], keypoints[frame, :, 2], c='r', marker='o', label='Predicted')

    for i, (x, y, z) in enumerate(keypoints[frame]):
        if flipped:
            ax.text(x, z, y, str(i), fontsize=10)
        else:
            ax.text(x, y, z, str(i), fontsize=10)
        
    for i, j in link:
        if flipped:
            ax.plot([keypoints[frame, i, 0], keypoints[frame, j, 0]],
                    [keypoints[frame, i, 2], keypoints[frame, j, 2]],
                    [keypoints[frame, i, 1], keypoints[frame, j, 1]], c='b')
        else:
            ax.plot([keypoints[frame, i, 0], keypoints[frame, j, 0]],
                    [keypoints[frame, i, 1], keypoints[frame, j, 1]],
                    [keypoints[frame, i, 2], keypoints[frame, j, 2]], c='b')
        
    if gt is not None:
        if flipped:
            ax.scatter(gt[frame, :, 0], gt[frame, :, 2], gt[frame, :, 1], c='g', marker='o', label='Ground Truth')
        else:
            ax.scatter(gt[frame, :, 0], gt[frame, :, 1], gt[frame, :, 2], c='g', marker='o', label='Ground Truth')

        for i, j in link:
            if flipped:
                ax.plot([gt[frame, i, 0], gt[frame, j, 0]],
                        [gt[frame, i, 2], gt[frame, j, 2]],
                        [gt[frame, i, 1], gt[frame, j, 1]], c='k')
            else:
                ax.plot([gt[frame, i, 0], gt[frame, j, 0]],
                        [gt[frame, i, 1], gt[frame, j, 1]],
                        [gt[frame, i, 2], gt[frame, j, 2]], c='k')
        
    ax.legend()
    
    ax.set_xlim(global_mid_x - global_max_range, global_mid_x + global_max_range)
    ax.set_ylim(global_mid_y - global_max_range, global_mid_y + global_max_range)
    ax.set_zlim(global_mid_z - global_max_range, global_mid_z + global_max_range)

    if flipped:
        ax.set_ylim(global_mid_z - global_max_range, global_mid_z + global_max_range)
        ax.set_zlim(global_mid_y - global_max_range, global_mid_y + global_max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
            
    ax.set_title(f"{title}: Frame {frame}")

    return ax

def plot2d_keypoints(keypoints, frame=0, title='Keypoints', save_path=None, link_type='aist'):
    """
    Plot the keypoints at a specific frame.
    """
    if link_type == 'aist':
        link = LINK_AIST
    else:
        raise ValueError(f"Link type {link_type} not supported")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(keypoints[frame, :, 0], keypoints[frame, :, 1], c='r', marker='o')

    for i, j in link:
        # Plot a single line between points (x1,y1) and (x2,y2)
        ax.plot([keypoints[frame, i, 0], keypoints[frame, j, 0]], 
                [keypoints[frame, i, 1], keypoints[frame, j, 1]], c='b')
    for i, (x, y, z) in enumerate(keypoints[frame]):
        ax.text(x, y, str(i), fontsize=10)

    ax.set_title(title + f": Frame {frame}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set equal aspect ratio for all axes
    X = keypoints[frame, :, 0]
    Y = keypoints[frame, :, 1]
    
    # Get the range of each axis
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)

    if save_path:
        plt.savefig(save_path)


def visualize_pd_skeleton(ax, frame: int, df: pd.DataFrame, skeleton=None, title='Keypoints', gt=None, flipped=False):
    if skeleton is None:
        skeleton = get_keypoint_skeleton()
    joint_names = get_motorica_skeleton_names()
    for idx, joint in enumerate(joint_names):
        # ^ In mocaps, Y is the up-right axis
        parent_x = df[f"{joint}_Xposition"].iloc[frame]
        parent_y = df[f"{joint}_Yposition"].iloc[frame]
        parent_z = df[f"{joint}_Zposition"].iloc[frame]
        if flipped:
            parent_y = df[f"{joint}_Zposition"].iloc[frame]
            parent_z = df[f"{joint}_Yposition"].iloc[frame]
        # print(f'joint: {joint}: parent_x: {parent_x}, parent_y: {parent_y}, parent_z: {parent_z}')
        ax.scatter(xs=parent_x, ys=parent_y, zs=parent_z, alpha=0.6, c="y", marker="o", label='Predicted')
        if gt is not None:
            gt_parent_x = gt[f"{joint}_Xposition"].iloc[frame]
            gt_parent_y = gt[f"{joint}_Yposition"].iloc[frame]
            gt_parent_z = gt[f"{joint}_Zposition"].iloc[frame]
            if flipped:
                gt_parent_y = gt[f"{joint}_Zposition"].iloc[frame]
                gt_parent_z = gt[f"{joint}_Yposition"].iloc[frame]
            ax.scatter(xs=gt_parent_x, ys=gt_parent_y, zs=gt_parent_z, alpha=0.6, c="k", marker="o", label='Ground Truth')

        children_to_draw = [
            c for c in skeleton[joint]["children"] if c in joint_names
        ]

        for c in children_to_draw:
            # ^ In mocaps, Y is the up-right axis
            child_x = df[f"{c}_Xposition"].iloc[frame]
            child_y = df[f"{c}_Yposition"].iloc[frame]
            child_z = df[f"{c}_Zposition"].iloc[frame]
            if flipped:
                child_y = df[f"{c}_Zposition"].iloc[frame]
                child_z = df[f"{c}_Yposition"].iloc[frame]
            ax.plot(
                [parent_x, child_x],
                [parent_y, child_y],
                [parent_z, child_z],
                # "k-",
                lw=2,
                c="black",
            )

            if gt is not None:
                gt_child_x = gt[f"{c}_Xposition"].iloc[frame]
                gt_child_y = gt[f"{c}_Yposition"].iloc[frame]
                gt_child_z = gt[f"{c}_Zposition"].iloc[frame]
                if flipped:
                    gt_child_y = gt[f"{c}_Zposition"].iloc[frame]
                    gt_child_z = gt[f"{c}_Yposition"].iloc[frame]
                ax.plot(
                    [gt_parent_x, gt_child_x],
                    [gt_parent_y, gt_child_y],
                    [gt_parent_z, gt_child_z],
                    lw=2,
                    c="green",
                )

        ax.text(
            x=parent_x - 0.01,
            y=parent_y - 0.01,
            z=parent_z - 0.01,
            s=f"{idx}:{joint}",
            fontsize=5,
        )
        # if joint =='LeftToeBase':
        #     # add text of showing the foot coordinate
        #     ax.text(
        #         x=-0.5,
        #         y=0.5,
        #         z=0.5,
        #         s=f"{joint}: ({parent_x:.3f}, {parent_y:.3f}, {parent_z:.3f})")
        # if joint == 'RightToeBase':
        #     ax.text(
        #         x=-0.5,
        #         y=0.5,
        #         z=0.6,
        #         s=f"{joint}: ({parent_x:.3f}, {parent_y:.3f}, {parent_z:.3f})")
    
    # ax.legend()
    ax.set_title(f"{title}: Frame {frame}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.scatter(xs=0, ys=0, zs=0, c='r', marker='o')

    return ax


def _reencode_h264(temp_path, output_path):
    """Re-encode a temp mp4 (mp4v / OpenCV-written) to H.264 / yuv420p so the
    file plays in VSCode's preview pane and standard browsers.

    Prefers /usr/bin/ffmpeg with libx264 (system ffmpeg on most Linux
    distros has a working build), then falls back to whatever ffmpeg is on
    PATH with a sequence of common H.264 encoders. Conda envs sometimes
    ship a broken libopenh264, so we don't trust any single binary.
    Final fallback: keep the cv2 output as-is — it's a valid mp4, just not
    as widely playable.
    """
    import shutil
    import subprocess

    # Candidate (binary, encoder) pairs, tried in order.
    candidates = []
    if os.path.exists("/usr/bin/ffmpeg"):
        candidates.append(("/usr/bin/ffmpeg", "libx264"))
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg and path_ffmpeg != "/usr/bin/ffmpeg":
        # The conda/PATH ffmpeg might still work for someone — try several
        # encoders before giving up.
        for enc in ("libx264", "libopenh264", "h264_nvenc"):
            candidates.append((path_ffmpeg, enc))

    if not candidates:
        # No ffmpeg anywhere. Keep the cv2 output.
        if temp_path != output_path:
            os.replace(temp_path, output_path)
        return

    last_err = None
    for binary, encoder in candidates:
        try:
            subprocess.run(
                [
                    binary, "-y",
                    "-i", temp_path,
                    "-c:v", encoder,
                    "-pix_fmt", "yuv420p",
                    "-loglevel", "error",
                    output_path,
                ],
                check=True,
            )
            if os.path.exists(temp_path) and temp_path != output_path:
                os.remove(temp_path)
            return
        except subprocess.CalledProcessError as e:
            last_err = (binary, encoder, e)
            # Clean up partial output before the next attempt.
            if os.path.exists(output_path) and output_path != temp_path:
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            continue

    print(
        f"WARNING: no working H.264 encoder found (tried {[c[1] for c in candidates]}); "
        f"keeping mp4v file. Last failure: {last_err}"
    )
    if temp_path != output_path:
        os.replace(temp_path, output_path)


def create_video_from_keypoints(keypoints,
                                output_path, 
                                trajectory=None,
                                vertices=None,
                                smpl_model=None,
                                skeleton=None, 
                                fps=30, 
                                title='Keypoints', 
                                audio_path=None, 
                                link_type='aist', 
                                gt=None,
                                gt_vertices=None,
                                gt_link_type=None,
                                max_frames=-1,
                                flipped=False):
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Always write cv2 frames to a temp mp4 (mp4v codec — fast, reliable in
    # OpenCV) and re-encode to H.264 (libx264) at the end. cv2's mp4v output
    # isn't playable in VSCode's preview pane or most browsers; the audio
    # branch already re-encoded with libx264, the no-audio branch did not.
    # This makes the output H.264 in both cases.
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the figure to get dimensions
    fig.canvas.draw()
    # Use buffer_rgba() instead of tostring_rgb()
    img = np.array(fig.canvas.buffer_rgba())
    # Convert RGBA to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    counter = 0

    # Calculate global min and max values across all frames
    global_min_x = np.min(keypoints[:, :, 0])
    global_max_x = np.max(keypoints[:, :, 0])
    global_min_y = np.min(keypoints[:, :, 1])
    global_max_y = np.max(keypoints[:, :, 1])
    global_min_z = np.min(keypoints[:, :, 2])
    global_max_z = np.max(keypoints[:, :, 2])

    if flipped:
        global_min_y = global_min_z
        global_max_y = global_max_z
        global_min_z = np.min(keypoints[:, :, 1])
        global_max_z = np.max(keypoints[:, :, 1])
    
    # Calculate global range and midpoints
    global_max_range = np.array([
        global_max_x - global_min_x,
        global_max_y - global_min_y,
        global_max_z - global_min_z
    ]).max() / 2.0
    
    global_mid_x = (global_max_x + global_min_x) * 0.5
    global_mid_y = (global_max_y + global_min_y) * 0.5
    global_mid_z = (global_max_z + global_min_z) * 0.5

    # Prepare artists once and update per-frame for speed
    pred_scatter = None
    gt_scatter = None
    pred_lines = []
    gt_lines = []
    mesh_collection = None
    mesh_collection_gt = None
    trajectory_scatter = None
    trajectory_points = []  # Store accumulated trajectory points

    def _update_scatter3d(sc, pts, flipped=False):
        if flipped:
            sc._offsets3d = (pts[:, 0], pts[:, 2], pts[:, 1])
        else:
            sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

    def _update_line3d(line, p1, p2, flipped=False):
        if flipped:
            line.set_data([p1[0], p2[0]], [p1[2], p2[2]])
            line.set_3d_properties([p1[1], p2[1]])
        else:
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])

    # Determine link sets
    if link_type is None:
        link = None
    elif isinstance(keypoints, pd.DataFrame):
        link = None  # handled in visualize_pd_skeleton fallback
    elif link_type.lower() == 'aist':
        link = LINK_AIST
    elif link_type.lower() == 'motorica':
        link = LINK_MOTORICA
    elif link_type.lower() == 'smpl':
        link = LINK_SMPL
    else:
        link = None

    if gt_link_type is None:
        gt_link = None
    elif gt_link_type and isinstance(gt_link_type, str) and gt_link_type.lower() == 'aist':
        gt_link = LINK_AIST
    elif gt_link_type and isinstance(gt_link_type, str) and gt_link_type.lower() == 'motorica':
        gt_link = LINK_MOTORICA
    elif gt_link_type and isinstance(gt_link_type, str) and gt_link_type.lower() == 'smpl':
        gt_link = LINK_SMPL
    else:
        gt_link = None

    # Initialize artists with the first frame
    if vertices is not None:
        # Mesh for predicted vertices
        faces = smpl_model.smpl_model.faces
        if flipped:
            # Apply coordinate flip: swap Y and Z coordinates
            flipped_vertices = vertices[0].copy()
            flipped_vertices[:, [1, 2]] = flipped_vertices[:, [2, 1]]
            mesh_collection = Poly3DCollection(flipped_vertices[faces], alpha=0.01)
        else:
            mesh_collection = Poly3DCollection(vertices[0][faces], alpha=0.01)
        mesh_collection.set_edgecolor((0, 0, 0))
        mesh_collection.set_facecolor((1.0, 1.0, 0.9))
        ax.add_collection3d(mesh_collection)
        # Predicted keypoints scatter
        if flipped:
            pred_scatter = ax.scatter(keypoints[0, :, 0], keypoints[0, :, 2], keypoints[0, :, 1], color="r", s=1, alpha=1, label='Predicted')
        else:
            pred_scatter = ax.scatter(keypoints[0, :, 0], keypoints[0, :, 1], keypoints[0, :, 2], color="r", s=1, alpha=1, label='Predicted')
        # Ground-truth
        if gt is not None:
            if flipped:
                gt_scatter = ax.scatter(gt[0, :, 0], gt[0, :, 2], gt[0, :, 1], color='g', s=1, alpha=1, label='Ground Truth')
            else:
                gt_scatter = ax.scatter(gt[0, :, 0], gt[0, :, 1], gt[0, :, 2], color='g', s=1, alpha=1, label='Ground Truth')
            if gt_link is not None:
                for (i, j) in gt_link:
                    if flipped:
                        line, = ax.plot([gt[0, i, 0], gt[0, j, 0]],
                                        [gt[0, i, 2], gt[0, j, 2]],
                                        [gt[0, i, 1], gt[0, j, 1]], c='k')
                    else:
                        line, = ax.plot([gt[0, i, 0], gt[0, j, 0]],
                                        [gt[0, i, 1], gt[0, j, 1]],
                                        [gt[0, i, 2], gt[0, j, 2]], c='k')
                    gt_lines.append(line)
        if gt_vertices is not None:
            if flipped:
                # Apply coordinate flip: swap Y and Z coordinates
                flipped_gt_vertices = gt_vertices[0].copy()
                flipped_gt_vertices[:, [1, 2]] = flipped_gt_vertices[:, [2, 1]]
                mesh_collection_gt = Poly3DCollection(flipped_gt_vertices[faces], alpha=0.01)
            else:
                mesh_collection_gt = Poly3DCollection(gt_vertices[0][faces], alpha=0.01)
            mesh_collection_gt.set_edgecolor((0, 0, 0))
            mesh_collection_gt.set_facecolor((1.0, 1.0, 0.9))
            ax.add_collection3d(mesh_collection_gt)
        
        # Trajectory visualization - start with empty scatter
        if trajectory is not None:
            trajectory_scatter = ax.scatter([], [], [], color='g', marker='o', s=20, alpha=0.8, label='Trajectory')
    elif isinstance(keypoints, pd.DataFrame):
        # Fallback: slower path reused, but at least avoid recreating figure each frame
        visualize_pd_skeleton(ax, frame=0, df=keypoints, skeleton=skeleton, gt=gt, flipped=flipped)
    else:
        ax.scatter([0, 0], [0, 0], [0, 0], c='k', s=10)
        # Generic keypoints path
        if flipped:
            pred_scatter = ax.scatter(keypoints[0, :, 0], keypoints[0, :, 2], keypoints[0, :, 1], c='r', marker='o', label='Predicted', s=5)
        else:
            pred_scatter = ax.scatter(keypoints[0, :, 0], keypoints[0, :, 1], keypoints[0, :, 2], c='r', marker='o', label='Predicted', s=5)
        if link is not None:
            for (i, j) in link:
                if flipped:
                    line, = ax.plot([keypoints[0, i, 0], keypoints[0, j, 0]],
                                    [keypoints[0, i, 2], keypoints[0, j, 2]],
                                    [keypoints[0, i, 1], keypoints[0, j, 1]], c='b')
                else:
                    line, = ax.plot([keypoints[0, i, 0], keypoints[0, j, 0]],
                                    [keypoints[0, i, 1], keypoints[0, j, 1]],
                                    [keypoints[0, i, 2], keypoints[0, j, 2]], c='b')
                pred_lines.append(line)
        if gt is not None:
            if flipped:
                gt_scatter = ax.scatter(gt[0, :, 0], gt[0, :, 2], gt[0, :, 1], c='g', marker='o', label='Ground Truth', s=5)
            else:
                gt_scatter = ax.scatter(gt[0, :, 0], gt[0, :, 1], gt[0, :, 2], c='g', marker='o', label='Ground Truth', s=5)
            if gt_link is not None:
                for (i, j) in gt_link:
                    if flipped:
                        line, = ax.plot([gt[0, i, 0], gt[0, j, 0]],
                                        [gt[0, i, 2], gt[0, j, 2]],
                                        [gt[0, i, 1], gt[0, j, 1]], c='k')
                    else:
                        line, = ax.plot([gt[0, i, 0], gt[0, j, 0]],
                                        [gt[0, i, 1], gt[0, j, 1]],
                                        [gt[0, i, 2], gt[0, j, 2]], c='k')
                    gt_lines.append(line)
        
        # Trajectory visualization for generic keypoints (vertices is None case) - start with empty scatter
        if trajectory is not None:
            trajectory_scatter = ax.scatter([], [], [], color='g', marker='o', s=20, alpha=0.8, label='Trajectory')

    # Set axes properties once
    ax.set_xlim(global_mid_x - global_max_range, global_mid_x + global_max_range)
    ax.set_ylim(global_mid_y - global_max_range, global_mid_y + global_max_range)
    ax.set_zlim(global_mid_z - global_max_range, global_mid_z + global_max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='upper right')

    # Main rendering loop: update artists only
    total_frames = len(keypoints) if max_frames == -1 else min(len(keypoints), max_frames)
    for frame in tqdm(range(total_frames)):
        if vertices is not None:
            # Update mesh vertices
            if mesh_collection is not None:
                faces = smpl_model.smpl_model.faces
                if flipped:
                    # Apply coordinate flip: swap Y and Z coordinates
                    flipped_vertices = vertices[frame].copy()
                    flipped_vertices[:, [1, 2]] = flipped_vertices[:, [2, 1]]
                    mesh_collection.set_verts(flipped_vertices[faces])
                else:
                    mesh_collection.set_verts(vertices[frame][faces])
            # Update predicted scatter
            if pred_scatter is not None:
                _update_scatter3d(pred_scatter, keypoints[frame], flipped=flipped)
            # Update ground truth scatter and lines
            if gt is not None and gt_scatter is not None:
                _update_scatter3d(gt_scatter, gt[frame], flipped=flipped)
                if gt_link is not None and gt_lines:
                    for line, (i, j) in zip(gt_lines, gt_link):
                        _update_line3d(line, gt[frame, i], gt[frame, j], flipped=flipped)
            # Update gt mesh
            if gt_vertices is not None and mesh_collection_gt is not None:
                if flipped:
                    # Apply coordinate flip: swap Y and Z coordinates
                    flipped_gt_vertices = gt_vertices[frame].copy()
                    flipped_gt_vertices[:, [1, 2]] = flipped_gt_vertices[:, [2, 1]]
                    mesh_collection_gt.set_verts(flipped_gt_vertices[faces])
                else:
                    mesh_collection_gt.set_verts(gt_vertices[frame][faces])
            
            # Update trajectory - accumulate points up to current frame
            if trajectory is not None and trajectory_scatter is not None:
                trajectory_points.append(trajectory[frame])
                if flipped:
                    # Apply coordinate flip: swap Y and Z coordinates
                    flipped_trajectory = np.array(trajectory_points)
                    flipped_trajectory[:, [1, 2]] = flipped_trajectory[:, [2, 1]]
                    # trajectory_scatter._offsets3d = (flipped_trajectory[:, 0], flipped_trajectory[:, 2], flipped_trajectory[:, 1])
                else:
                    trajectory_array = np.array(trajectory_points)
                    # trajectory_scatter._offsets3d = (trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2])
        elif isinstance(keypoints, pd.DataFrame):
            # Slower fallback path: clear axes contents, redraw skeleton for this frame
            ax.cla()
            ax.set_xlim(global_mid_x - global_max_range, global_mid_x + global_max_range)
            ax.set_ylim(global_mid_y - global_max_range, global_mid_y + global_max_range)
            ax.set_zlim(global_mid_z - global_max_range, global_mid_z + global_max_range)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"{title}: Frame {frame}")
            visualize_pd_skeleton(ax, frame=frame, df=keypoints, skeleton=skeleton, gt=gt, flipped=flipped)
        else:
            # Update generic keypoints
            if pred_scatter is not None:
                _update_scatter3d(pred_scatter, keypoints[frame], flipped=flipped)
            if link is not None and pred_lines:
                for line, (i, j) in zip(pred_lines, link):
                    _update_line3d(line, keypoints[frame, i], keypoints[frame, j], flipped=flipped)
            if gt is not None and gt_scatter is not None:
                _update_scatter3d(gt_scatter, gt[frame], flipped=flipped)
                if gt_link is not None and gt_lines:
                    for line, (i, j) in zip(gt_lines, gt_link):
                        _update_line3d(line, gt[frame, i], gt[frame, j], flipped=flipped)
            
            # Update trajectory - accumulate points up to current frame
            if trajectory is not None and trajectory_scatter is not None:
                trajectory_points.append(trajectory[frame])
                if flipped:
                    # Apply coordinate flip: swap Y and Z coordinates
                    flipped_trajectory = np.array(trajectory_points)
                    # flipped_trajectory[:, [1, 2]] = flipped_trajectory[:, [2, 1]]
                    trajectory_scatter._offsets3d = (flipped_trajectory[:, 0], flipped_trajectory[:, 2], flipped_trajectory[:, 1])
                else:
                    trajectory_array = np.array(trajectory_points)
                    trajectory_scatter._offsets3d = (trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2])

        # Draw updated frame
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img)
        counter += 1

    # Clean up
    video_writer.release()
    plt.close(fig)

    # Add audio if provided using moviepy (required 1.0.3 version, recent one is not working)
    if audio_path and os.path.exists(audio_path):
        try:
            REMOVE = False
            total_time = len(keypoints) / fps
            wav, sr = sf.read(audio_path)
            audio_length = len(wav) / sr
            if audio_length > total_time:
                REMOVE=True
                sf.write(audio_path.replace('.wav', '_processed.wav'), wav, sr)
                audio_path = audio_path.replace('.wav', '_processed.wav')
            video_clip = VideoFileClip(temp_video_path)
            audio_clip = AudioFileClip(audio_path)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_path, codec='libx264')

            # Remove temporary video file
            if temp_video_path != output_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if REMOVE:
                os.remove(audio_path)

            print(f"Video with audio saved to {output_path}")
        except Exception as e:
            print(f"Error adding audio: {e}")
            # On audio failure, fall back to the H.264 re-encode below.
            _reencode_h264(temp_video_path, output_path)
    else:
        # No audio: re-encode the cv2 mp4v output to H.264 so VSCode's
        # preview pane and browsers can play it back.
        _reencode_h264(temp_video_path, output_path)
        if audio_path:
            print(f"Audio file {audio_path} not found. Video saved without audio.")
        else:
            print("Video saved without audio.")



def create_2D_video_from_keypoints(keypoints, 
                                output_path, 
                                vertices=None,
                                smpl_model=None,
                                skeleton=None, 
                                fps=30, 
                                title='Keypoints', 
                                audio_path=None, 
                                link_type='aist', 
                                gt=None,
                                gt_vertices=None,
                                gt_link_type=None,
                                max_frames=-1,
                                flipped=False):
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Always write through a temp file and re-encode to H.264 (see notes in
    # create_video_from_keypoints).
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Draw the figure to get dimensions
    fig.canvas.draw()
    # Use buffer_rgba() instead of tostring_rgb()
    img = np.array(fig.canvas.buffer_rgba())
    # Convert RGBA to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    counter = 0

    # Calculate global min and max values across all frames (x-y plane only)
    global_min_x = np.min(keypoints[:, :, 0])
    global_max_x = np.max(keypoints[:, :, 0])
    global_min_y = np.min(keypoints[:, :, 1])
    global_max_y = np.max(keypoints[:, :, 1])
    
    # Calculate global range and midpoints for x-y plane
    global_max_range = np.array([
        global_max_x - global_min_x,
        global_max_y - global_min_y
    ]).max() / 2.0
    
    global_mid_x = (global_max_x + global_min_x) * 0.5
    global_mid_y = (global_max_y + global_min_y) * 0.5

    # Prepare artists once and update per-frame for speed
    pred_scatter = None
    gt_scatter = None
    pred_lines = []
    gt_lines = []
    mesh_collection = None
    mesh_collection_gt = None

    def _update_scatter2d(sc, pts):
        # Update scatter plot for x-y plane (2D)
        sc.set_offsets(pts[:, [0, 1]])

    def _update_line2d(line, p1, p2):
        # Update line plot for x-y plane (2D)
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])

    # Determine link sets
    if link_type is None:
        link = None
    elif isinstance(keypoints, pd.DataFrame):
        link = None  # handled in visualize_pd_skeleton fallback
    elif link_type.lower() == 'aist':
        link = LINK_AIST
    elif link_type.lower() == 'motorica':
        link = LINK_MOTORICA
    elif link_type.lower() == 'smpl':
        link = LINK_SMPL
    else:
        link = None

    if gt_link_type is None:
        gt_link = None
    elif gt_link_type and isinstance(gt_link_type, str) and gt_link_type.lower() == 'aist':
        gt_link = LINK_AIST
    elif gt_link_type and isinstance(gt_link_type, str) and gt_link_type.lower() == 'motorica':
        gt_link = LINK_MOTORICA
    elif gt_link_type and isinstance(gt_link_type, str) and gt_link_type.lower() == 'smpl':
        gt_link = LINK_SMPL
    else:
        gt_link = None

    # Initialize artists with the first frame
    if vertices is not None:
        # Note: Mesh visualization is not supported in 2D x-y plane
        # Predicted keypoints scatter (x-y plane)
        pred_scatter = ax.scatter(keypoints[0, :, 0], keypoints[0, :, 1], color="r", s=1, alpha=1, label='Predicted')
        # Ground-truth
        if gt is not None:
            gt_scatter = ax.scatter(gt[0, :, 0], gt[0, :, 1], color='g', s=1, alpha=1, label='Ground Truth')
            if gt_link is not None:
                for (i, j) in gt_link:
                    line, = ax.plot([gt[0, i, 0], gt[0, j, 0]],
                                    [gt[0, i, 1], gt[0, j, 1]], c='k')
                    gt_lines.append(line)
    elif isinstance(keypoints, pd.DataFrame):
        # Fallback: slower path reused, but at least avoid recreating figure each frame
        visualize_pd_skeleton(ax, frame=0, df=keypoints, skeleton=skeleton, gt=gt, flipped=flipped)
    else:
        ax.scatter([0, 0], [0, 0], c='k', s=10)
        # Generic keypoints path (x-y plane)
        pred_scatter = ax.scatter(keypoints[0, :, 0], keypoints[0, :, 1], c='r', marker='o', label='Predicted', s=5)
        if link is not None:
            for (i, j) in link:
                line, = ax.plot([keypoints[0, i, 0], keypoints[0, j, 0]],
                                [keypoints[0, i, 1], keypoints[0, j, 1]], c='b')
                pred_lines.append(line)
        if gt is not None:
            gt_scatter = ax.scatter(gt[0, :, 0], gt[0, :, 1], c='g', marker='o', label='Ground Truth', s=5)
            if gt_link is not None:
                for (i, j) in gt_link:
                    line, = ax.plot([gt[0, i, 0], gt[0, j, 0]],
                                        [gt[0, i, 1], gt[0, j, 1]], c='k')
                    gt_lines.append(line)

    # Set axes properties once (x-y plane)
    ax.set_xlim(global_mid_x - global_max_range, global_mid_x + global_max_range)
    ax.set_ylim(global_mid_y - global_max_range, global_mid_y + global_max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Main rendering loop: update artists only
    total_frames = len(keypoints) if max_frames == -1 else min(len(keypoints), max_frames)
    for frame in tqdm(range(total_frames)):
        if vertices is not None:
            # Note: Mesh visualization is not supported in 2D x-y plane
            # Update predicted scatter
            if pred_scatter is not None:
                _update_scatter2d(pred_scatter, keypoints[frame])
            # Update ground truth scatter and lines
            if gt is not None and gt_scatter is not None:
                _update_scatter2d(gt_scatter, gt[frame])
                if gt_link is not None and gt_lines:
                    for line, (i, j) in zip(gt_lines, gt_link):
                        _update_line2d(line, gt[frame, i], gt[frame, j])
        elif isinstance(keypoints, pd.DataFrame):
            # Slower fallback path: clear axes contents, redraw skeleton for this frame
            ax.cla()
            ax.set_xlim(global_mid_x - global_max_range, global_mid_x + global_max_range)
            ax.set_ylim(global_mid_y - global_max_range, global_mid_y + global_max_range)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f"{title}: Frame {frame}")
            ax.grid(True, alpha=0.3)
            visualize_pd_skeleton(ax, frame=frame, df=keypoints, skeleton=skeleton, gt=gt, flipped=flipped)
        else:
            # Update generic keypoints (x-y plane)
            if pred_scatter is not None:
                _update_scatter2d(pred_scatter, keypoints[frame])
            if link is not None and pred_lines:
                for line, (i, j) in zip(pred_lines, link):
                    _update_line2d(line, keypoints[frame, i], keypoints[frame, j])
            if gt is not None and gt_scatter is not None:
                _update_scatter2d(gt_scatter, gt[frame])
                if gt_link is not None and gt_lines:
                    for line, (i, j) in zip(gt_lines, gt_link):
                        _update_line2d(line, gt[frame, i], gt[frame, j])

        # Draw updated frame
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video_writer.write(img)
        counter += 1

    # Clean up
    video_writer.release()
    plt.close(fig)

    # Add audio if provided using moviepy (required 1.0.3 version, recent one is not working)
    if audio_path and os.path.exists(audio_path):
        try:
            REMOVE = False
            total_time = len(keypoints) / fps
            wav, sr = sf.read(audio_path)
            audio_length = len(wav) / sr
            if audio_length > total_time:
                REMOVE=True
                sf.write(audio_path.replace('.wav', '_processed.wav'), wav, sr)
                audio_path = audio_path.replace('.wav', '_processed.wav')
            video_clip = VideoFileClip(temp_video_path)
            audio_clip = AudioFileClip(audio_path)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(output_path, codec='libx264')

            # Remove temporary video file
            if temp_video_path != output_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if REMOVE:
                os.remove(audio_path)

            print(f"Video with audio saved to {output_path}")
        except Exception as e:
            print(f"Error adding audio: {e}")
            # On audio failure, fall back to the H.264 re-encode below.
            _reencode_h264(temp_video_path, output_path)
    else:
        # No audio: re-encode the cv2 mp4v output to H.264 so VSCode's
        # preview pane and browsers can play it back.
        _reencode_h264(temp_video_path, output_path)
        if audio_path:
            print(f"Audio file {audio_path} not found. Video saved without audio.")
        else:
            print("Video saved without audio.")

def create_video_from_images(image_pattern, output_path, fps=30, cleanup=True):
    """
    Create a video from a sequence of images.
    
    Args:
        image_pattern (str): Pattern for image files (e.g., './images/frame_{}.png')
        output_path (str): Path where the video will be saved
        fps (int): Frames per second for the output video
        cleanup (bool): Whether to delete the source images after video creation
    """
    # Get the first image to determine video size
    first_img_path = image_pattern.format(0)
    if not os.path.exists(first_img_path):
        raise FileNotFoundError(f"No images found matching pattern: {image_pattern}")
    
    img_array = []
    frame_count = 0
    
    # Keep reading images until we don't find the next one
    while True:
        img_path = image_pattern.format(frame_count)
        if not os.path.exists(img_path):
            break
            
        img = cv2.imread(img_path)
        if frame_count == 0:  # Get size from first frame
            height, width, layers = img.shape
            size = (width, height)
            
        img_array.append(img)
        frame_count += 1
    
    if frame_count == 0:
        raise ValueError("No frames were loaded")
        
    # Create video writer
    out = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, size)
    
    # Write frames to video
    for img in img_array:
        out.write(img)
    out.release()
    
    # Cleanup source images if requested
    if cleanup:
        for i in range(frame_count):
            img_path = image_pattern.format(i)
            if os.path.exists(img_path):
                os.remove(img_path)


# if __name__ == "__main__":
#     id2 = 'gBR_sBM_cAll_d04_mBR0_ch05_chunk0'
#     data1 = np.load('data/AIST/sliced_motion/' + id2 + '_motion.npy')
#     data1 = motion_preprocessing(data1, 'norm_pelvis_origin')

#     create_video_from_keypoints(data1, './videos/test.mp4', fps=30, title=id2, audio_path='./data/AIST/sliced_audio/' + id2 + '.wav')