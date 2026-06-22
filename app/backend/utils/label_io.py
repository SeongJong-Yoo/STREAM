"""Read / write the per-segment label .txt files used by the dance pipeline."""

import os
import json
from pathlib import Path


def read_label_segment(path):
    """Read a single label segment .txt file.

    Returns dict with keys: genre, label, start, end, fps, description.
    """
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            key = key.strip()
            value = value.strip()
            if key in ("start", "end", "fps"):
                value = int(value)
            out[key] = value
    return out


def write_label_segment(path, segment):
    """Write a label segment dict back to a .txt file."""
    with open(path, "w") as f:
        for key in ("genre", "label", "start", "end", "fps", "description"):
            if key in segment:
                f.write(f"{key}: {segment[key]}\n")


def load_all_segments(label_dir):
    """Load every .txt segment in *label_dir*, sorted by start frame.

    Returns a list of dicts, each augmented with ``_path`` (source file).
    """
    label_dir = Path(label_dir)
    segments = []
    for f in sorted(label_dir.iterdir()):
        if f.suffix != ".txt":
            continue
        seg = read_label_segment(f)
        seg["_path"] = str(f)
        segments.append(seg)
    # Sort by start frame
    segments.sort(key=lambda s: s.get("start", 0))
    return segments


def save_all_segments(label_dir, segments):
    """Write every segment back to its ``_path`` (or a generated name)."""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    for i, seg in enumerate(segments):
        path = seg.get("_path")
        if path is None:
            # Generate a filename based on index
            path = str(label_dir / f"segment_{i}.txt")
            seg["_path"] = path
        write_label_segment(path, seg)


def rewrite_all_segments(label_dir, segments):
    """Delete all .txt in label_dir, then re-write segments with sequential names.

    Each segment dict is updated with a fresh ``_path``.
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing .txt files
    for f in label_dir.iterdir():
        if f.suffix == ".txt":
            f.unlink()

    # Write sequentially
    for i, seg in enumerate(segments):
        path = label_dir / f"segment_{i:03d}.txt"
        seg["_path"] = str(path)
        write_label_segment(str(path), seg)


def load_label_descriptions(json_path):
    """Load the genre -> label -> [descriptions] mapping from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)
