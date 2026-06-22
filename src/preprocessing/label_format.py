"""Compact JSON label format for sliced label chunks.

The legacy format wrote a `(T, 3)` `<U1000` numpy matrix per chunk where every
frame stored a copy of `[genre, label, description]`. Most chunks contain only
1–4 distinct segments yet the matrix repeats the same long description string
~150 times. This module replaces that with a JSON file that stores one entry
per segment plus chunk metadata, and provides helpers to rebuild the per-frame
representation that the dataloader expects.

Schema::

    {
        "fps": 30,
        "length": 150,
        "segments": [
            {"start": 0, "end": 50, "genre": "popping",
             "label": "arm wave", "description": "..."},
            ...
        ],
        # HumanML3D may carry a list of candidate descriptions per segment;
        # in that case "description" is replaced by "descriptions": [...].
    }
"""
from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np


SegmentDescription = Union[str, List[str]]


# Maximum string length the per-frame label matrix can hold without numpy
# silently truncating. Must match the U-width used by ``expand_segments_to_per_frame``
# (currently "<U1000"). Anything that keys CLIP cache lookups on label or
# description strings must apply the same truncation, otherwise the cache
# key written by precompute_clip_cache.py won't match the key the dataloader
# computes at runtime, and we'll see spurious "CLIP cache miss" warnings.
LABEL_STRING_MAX_LEN = 1000


def truncate_for_label_array(text: str) -> str:
    """Match the silent truncation that ``np.array(..., dtype='<U1000')``
    applies. Use this anywhere we build a CLIP cache key from a label or
    description string, so precompute and runtime agree on the key."""
    if not isinstance(text, str):
        text = str(text)
    return text[:LABEL_STRING_MAX_LEN]


def _segments_from_per_frame_label(label: np.ndarray) -> List[dict]:
    """Collapse a per-frame `(T, 3)` label matrix into segment dicts.

    A new segment starts whenever any of the three columns changes.
    """
    if label.ndim != 2 or label.shape[1] != 3:
        raise ValueError(
            f"Expected (T, 3) label matrix, got shape {label.shape}"
        )
    segments: List[dict] = []
    start = 0
    cur = (str(label[0, 0]), str(label[0, 1]), str(label[0, 2]))
    for i in range(1, label.shape[0]):
        row = (str(label[i, 0]), str(label[i, 1]), str(label[i, 2]))
        if row != cur:
            segments.append(
                {
                    "start": start,
                    "end": i,
                    "genre": cur[0],
                    "label": cur[1],
                    "description": cur[2],
                }
            )
            start = i
            cur = row
    segments.append(
        {
            "start": start,
            "end": label.shape[0],
            "genre": cur[0],
            "label": cur[1],
            "description": cur[2],
        }
    )
    return segments


def write_compact_label(
    save_path: str,
    *,
    length: int,
    fps: int,
    segments: Optional[Sequence[dict]] = None,
    per_frame_label: Optional[np.ndarray] = None,
) -> None:
    """Write a chunk label JSON.

    Provide either ``segments`` directly, or ``per_frame_label`` (shape
    `(T, 3)`) to be collapsed automatically. Each segment must carry
    ``start``, ``end``, ``genre``, ``label``, and one of ``description`` or
    ``descriptions``.
    """
    if segments is None:
        if per_frame_label is None:
            raise ValueError(
                "Provide either segments or per_frame_label"
            )
        segments = _segments_from_per_frame_label(per_frame_label)

    payload = {
        "fps": int(fps),
        "length": int(length),
        "segments": [
            _normalize_segment(seg) for seg in segments
        ],
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False)


def _normalize_segment(seg: dict) -> dict:
    out = {
        "start": int(seg["start"]),
        "end": int(seg["end"]),
        "genre": str(seg.get("genre", "")),
        "label": str(seg.get("label", "")),
    }
    if "descriptions" in seg:
        descs = seg["descriptions"]
        if isinstance(descs, str):
            descs = [descs]
        out["descriptions"] = [str(d) for d in descs]
    else:
        out["description"] = str(seg.get("description", ""))
    return out


def read_compact_label(path: str) -> dict:
    """Load a compact JSON label file."""
    with open(path, "r") as f:
        return json.load(f)


def _pick_description(seg: dict, description_idx: int = 0) -> str:
    """Resolve a single description string from a segment.

    For HumanML3D segments that carry multiple candidates the index is used
    cyclically; the cycle is per-segment so that a chunk with two segments
    each containing five descriptions does not get a degenerate index.
    """
    if "descriptions" in seg:
        descs = seg["descriptions"]
        if not descs:
            return ""
        return descs[description_idx % len(descs)]
    return seg.get("description", "")


def expand_segments_to_per_frame(
    payload: dict,
    *,
    description_idx: int = 0,
    dtype: str = "<U1000",
) -> np.ndarray:
    """Reconstruct the legacy `(T, 3)` per-frame label matrix.

    Note on the ``<U1000`` default: numpy silently truncates strings longer
    than the dtype's width. The downstream consumer
    (``class_label_to_embedding``) relies on ``.item()`` semantics, which
    only works on numpy fixed-width string scalars, not Python ``str``
    objects, so we can't switch to ``dtype=object`` without touching that
    consumer. Anything that keys CLIP cache lookups on these strings must
    truncate identically — see ``LABEL_STRING_MAX_LEN`` below.
    """
    length = int(payload["length"])
    segments = payload["segments"]
    out = np.empty((length, 3), dtype=dtype)
    out.fill("")
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        genre = str(seg.get("genre", ""))
        label = str(seg.get("label", ""))
        desc = _pick_description(seg, description_idx=description_idx)
        out[start:end, 0] = genre
        out[start:end, 1] = label
        out[start:end, 2] = desc
    return out


def label_index_from_segments(payload: dict) -> np.ndarray:
    """Reproduce the per-frame label_index used by the legacy format.

    Each segment maps to a 0→1 ramp over its duration (matching
    ``generate_index`` in the original preprocessing scripts).
    """
    length = int(payload["length"])
    segments = payload["segments"]
    out = np.zeros(length, dtype=np.float64)
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])
        n = end - start
        if n <= 0:
            continue
        if n == 1:
            out[start:end] = 0.0
        else:
            out[start:end] = np.arange(n) / (n - 1)
    return out


def label_chunk_record(
    payload: dict,
    *,
    description_idx: int = 0,
    dtype: str = "<U1000",
) -> dict:
    """Return a dict matching the legacy npy `output` record shape.

    Output keys: ``data`` (T,3 strings), ``label_index`` (T,), ``current_fps``,
    ``target_fps``. Compatible drop-in for the dataloader.
    """
    data = expand_segments_to_per_frame(
        payload, description_idx=description_idx, dtype=dtype
    )
    return {
        "data": data,
        "label_index": label_index_from_segments(payload),
        "current_fps": int(payload["fps"]),
        "target_fps": int(payload["fps"]),
    }


def num_description_candidates(payload: dict) -> int:
    """Largest number of description candidates across segments.

    HumanML3D-style files can vary per segment; the dataloader uses this to
    cycle through epochs without ever indexing past the available list.
    """
    n = 1
    for seg in payload.get("segments", []):
        if "descriptions" in seg and seg["descriptions"]:
            n = max(n, len(seg["descriptions"]))
    return n
