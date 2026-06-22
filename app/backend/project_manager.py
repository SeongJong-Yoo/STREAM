"""Manages loading / saving a dance-design-studio project folder.

Data folder layout::

    <folder>/
        audio.wav
        label/            # per-segment .txt files
        audio_features/   # optional – computed on first load
        sliced_audio/     # created during prep
        sliced_labels/    # created during prep
        sliced_audio_features/  # created during prep

Model folder layout::

    <folder>/
        config.yaml
        checkpoints/last.ckpt
"""

import os
import json
import numpy as np
from pathlib import Path
from copy import deepcopy

from app.backend.utils.label_io import (
    load_all_segments,
    save_all_segments,
    write_label_segment,
    rewrite_all_segments,
)


class ProjectManager:
    def __init__(self):
        self.folder = None
        self.segments = []          # list of label segment dicts
        self.model_config = None    # OmegaConf config
        self.model_path = None      # path to .ckpt
        self.audio_path = None
        self.fps = 30               # label / motion fps
        self.motion_data = None     # generated SMPL params  (T, 147)
        self.joint_data = None      # FK joints              (T, 24, 3)
        self.vert_data = None       # FK vertices            (T, V, 3)
        self.faces = None           # SMPL face indices

    # ------------------------------------------------------------------ load
    def load_data(self, folder):
        """Load a data folder (audio + labels). Returns error string or None."""
        folder = Path(folder)
        if not folder.is_dir():
            return f"Not a directory: {folder}"

        audio = folder / "audio.wav"
        if not audio.exists():
            return "Missing audio.wav"

        label_dir = folder / "label"
        if not label_dir.is_dir():
            return "Missing label/ folder"

        self.folder = folder
        self.audio_path = str(audio)
        self.segments = load_all_segments(label_dir)
        if self.segments:
            self.fps = self.segments[0].get("fps", 30)

        return None  # success

    # -------------------------------------------------------------- helpers
    @property
    def label_dir(self):
        return self.folder / "label" if self.folder else None

    @property
    def audio_features_dir(self):
        return self.folder / "sliced_audio_features" if self.folder else None

    @property
    def needs_audio_prep(self):
        d = self.audio_features_dir
        return d is None or not d.exists() or not any(d.iterdir())

    @property
    def total_duration(self):
        """Duration in seconds based on label segments."""
        if not self.segments:
            return 0.0
        last = max(s.get("end", 0) for s in self.segments)
        return last / self.fps

    @property
    def total_frames(self):
        if not self.segments:
            return 0
        return max(s.get("end", 0) for s in self.segments)

    # -------------------------------------------------------- label editing
    def update_segment(self, index, genre=None, label=None, description=None):
        """Update a single segment in memory and on disk."""
        seg = self.segments[index]
        if genre is not None:
            seg["genre"] = genre
        if label is not None:
            seg["label"] = label
        if description is not None:
            seg["description"] = description
        write_label_segment(seg["_path"], seg)

    def update_segments(self, indices, genre=None, label=None, description=None):
        """Batch-update multiple segments."""
        for idx in indices:
            self.update_segment(idx, genre, label, description)

    def apply_edit_with_range(self, edit_start, edit_end, genre, label, description):
        """Split / merge segments so that [edit_start, edit_end) gets new properties.

        Parameters
        ----------
        edit_start, edit_end : int  – frame numbers
        genre, label, description : str

        Returns the list of new segment indices that were affected (for partial regen).
        """
        if edit_start >= edit_end:
            return []

        new_segments = []
        for seg in self.segments:
            s, e = seg.get("start", 0), seg.get("end", 0)

            # No overlap — keep as-is
            if e <= edit_start or s >= edit_end:
                new_segments.append(seg)
                continue

            # Left remainder
            if s < edit_start:
                left = deepcopy(seg)
                left["end"] = edit_start
                new_segments.append(left)

            # Right remainder
            if e > edit_end:
                right = deepcopy(seg)
                right["start"] = edit_end
                new_segments.append(right)

            # (the overlapping portion is consumed — not appended)

        # Create the new edit segment
        new_seg = {
            "genre": genre,
            "label": label,
            "description": description,
            "start": edit_start,
            "end": edit_end,
            "fps": self.fps,
        }
        new_segments.append(new_seg)

        # Sort by start frame
        new_segments.sort(key=lambda s: s.get("start", 0))
        self.segments = new_segments

        # Persist: rewrite all label files
        rewrite_all_segments(self.label_dir, self.segments)

        # Return indices of segments that overlap the edited range (for partial regen)
        affected = []
        for i, seg in enumerate(self.segments):
            s, e = seg.get("start", 0), seg.get("end", 0)
            if s < edit_end and e > edit_start:
                affected.append(i)
        return affected

    # --------------------------------------------------------------- save
    def save_results(self, path=None):
        """Save SMPL params and FK joints (not mesh)."""
        if self.motion_data is None:
            return "No generated motion to save."
        if path is None:
            path = self.folder
        path = Path(path)
        output = {
            "smpl_params": self.motion_data,       # (T, 147)
            "joints": self.joint_data,              # (T, 24, 3)
        }
        np.save(str(path / "result.npy"), output)
        return None
