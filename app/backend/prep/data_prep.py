"""Audio slicing + Jukebox feature extraction for the dance design studio.

Mirrors the pipelines in:
  - src/preprocessing/inference_data_prep.py  (slice audio, labels, motion)
  - src/preprocessing/vae_data_prep_jukebox.py (jukebox feature extraction)
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.utils import slice_audio


# ------------------------------------------------------------------ slicing

def slice_project(folder, progress_callback=None):
    """Slice audio for a project folder.

    Creates:
        sliced_audio/

    Labels and beats are no longer sliced here — labels are encoded
    via CLIP on the fly in model_runner, and beats are unused.

    Parameters
    ----------
    folder : str or Path
    progress_callback : callable(str, float)  (message, 0-1 fraction)

    Returns error string or None.
    """
    folder = Path(folder)
    audio_path = folder / "audio.wav"

    WINDOW = 5.0      # seconds
    STRIDE = 2.5      # seconds

    if not audio_path.exists():
        return "Missing audio.wav"

    if progress_callback:
        progress_callback("Loading audio...", 0.0)

    audio, sr = librosa.load(str(audio_path), sr=44100)

    # Slice audio (beat_times not needed, pass empty array)
    audio_chunks, time_chunks, _ = slice_audio(
        audio, np.array([]), sr, WINDOW, STRIDE
    )
    num_chunks = len(audio_chunks)

    # Prepare output dir
    (folder / "sliced_audio").mkdir(parents=True, exist_ok=True)

    for i in range(num_chunks):
        sf.write(str(folder / "sliced_audio" / f"chunk{i}.wav"), audio_chunks[i], sr)
        if progress_callback:
            progress_callback("Slicing audio...", (i + 1) / num_chunks * 0.4)

    if progress_callback:
        progress_callback("Slicing complete.", 0.4)
    return None


# --------------------------------------------------------- jukebox features

def extract_jukebox_features(folder, progress_callback=None):
    """Extract Jukebox features for all sliced audio chunks.

    Reads from  <folder>/sliced_audio/
    Writes to   <folder>/sliced_audio_features/
    """
    import jukemirlib

    folder = Path(folder)
    src = folder / "sliced_audio"
    dest = folder / "sliced_audio_features"
    dest.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(src.glob("*.wav"))
    total = len(audio_files)
    if total == 0:
        return "No sliced audio files found. Run slice_project first."

    for i, audio_file in enumerate(audio_files):
        name = audio_file.stem
        save_path = dest / f"{name}_audio.npy"
        if save_path.exists():
            if progress_callback:
                progress_callback(f"Skipping {name} (cached)", (i + 1) / total)
            continue

        audio = jukemirlib.load_audio(audio_file)
        reps = jukemirlib.extract(audio, layers=[66], downsample_target_rate=30)[66]
        # Target 150 frames for 5s at 30fps
        target_len = 150
        if reps.shape[0] > target_len:
            reps = reps[:target_len]
        elif reps.shape[0] < target_len:
            pad = np.zeros((target_len - reps.shape[0], reps.shape[1]), dtype=reps.dtype)
            reps = np.concatenate([reps, pad], axis=0)

        output = {"audio": reps, "current_fps": 30, "target_fps": 30}
        np.save(str(save_path), output)
        if progress_callback:
            progress_callback(f"Jukebox: {name}", (i + 1) / total)

    return None


def prepare_project(folder, progress_callback=None):
    """Full prep pipeline: slice + jukebox features."""
    def _cb(msg, frac):
        if progress_callback:
            progress_callback(msg, frac * 0.4)

    err = slice_project(folder, progress_callback=_cb)
    if err:
        return err

    def _cb2(msg, frac):
        if progress_callback:
            progress_callback(msg, 0.4 + frac * 0.6)

    err = extract_jukebox_features(folder, progress_callback=_cb2)
    return err
