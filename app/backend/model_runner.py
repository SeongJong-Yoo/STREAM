"""Model loading and long-range generation pipeline.

Adapted from long_range_generation.py for the design studio.
Supports full generation and partial regeneration of edited chunks.
"""

import os
import re
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf
from scipy.signal import butter, filtfilt
import pytorch3d.transforms as t3d

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.skeleton.smpl_fk import SMPLModel
from src.skeleton.smpl_fk_smplx import SMPLModel as SMPLMeshModel
from src.skeleton.preprocessing import motion_preprocessing
from src.models.pl_module.cross_energy_edit_dance_joint import CrossEnergyEditDanceJoint
from src.utils.utility import CLIPLabelConverter, T5LabelConverter
from src.dataloader.utility import Normalizer

# Import stitching utilities from the long_range script
from long_range_generation import stitch, align_motion


class ModelRunner:
    def __init__(self):
        self.model = None
        self.config = None
        self.fk = None
        self.fk_mesh = None
        self.normalizer = None
        self.normalizer_latent = None
        self.label_converter = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Per-chunk storage for partial regeneration
        self._chunk_motions = []       # list of (1, T, D) tensors per chunk
        self._chunk_normalized = []    # list of normalized motion tensors per chunk
        self._feature_files = []       # ordered list of feature file paths
        self._num_chunks = 0
        self._folder = None

    def load_model(self, config_path, ckpt_path, progress_callback=None):
        """Load model from config.yaml + checkpoint.

        Returns error string or None.
        """
        if progress_callback:
            progress_callback("Loading config...", 0.0)

        config = OmegaConf.load(config_path)
        config.trainer.devices = 1

        model_name = config.model.name
        if model_name == "vae":
            config.model.VAE.pretrained_vae = ckpt_path
        elif "joint" in model_name:
            config.model.pretrained_energy = ckpt_path
        elif "diffusion" in model_name or "flow" in model_name:
            if hasattr(config.model, "Diffusion"):
                config.model.Diffusion.pretrained_diffusion = ckpt_path
            else:
                config.model.pretrained_energy = ckpt_path

        config.data.trained_dataset = deepcopy(config.data.dataset)
        self.config = config

        if progress_callback:
            progress_callback("Setting up normalizer...", 0.2)

        # FK models
        self.fk = SMPLModel()
        self.fk_mesh = SMPLMeshModel()

        # Normalizer
        root_dir = "./data/"
        normalizer_path = self._get_normalizer_path(config, root_dir, "smpl")
        if os.path.exists(normalizer_path):
            self.normalizer = Normalizer(path=normalizer_path)
        else:
            return f"Normalizer not found at {normalizer_path}"

        if config.model.name == "cross_latent_diffusion":
            lat_path = self._get_normalizer_path(config, root_dir, "latent")
            if os.path.exists(lat_path):
                self.normalizer_latent = Normalizer(path=lat_path)

        # Label converter — match what the checkpoint was trained with.
        # Both converters share the encode_text / class_label_to_embedding
        # API used downstream, so only the construction differs.
        label_converter_name = getattr(config.data, 'label_converter', 'clip')
        cache_dir = os.path.join('./data', getattr(config.data, 'dataset', ['motorica'])[0])
        if label_converter_name == 't5':
            text_cache_path = os.path.join(cache_dir, 'label_t5_cache.pkl')
            self.label_converter = T5LabelConverter(text_cache_path=text_cache_path)
        else:
            text_cache_path = os.path.join(cache_dir, 'label_clip_cache.pkl')
            self.label_converter = CLIPLabelConverter(text_cache_path=text_cache_path)

        if progress_callback:
            progress_callback("Loading model weights...", 0.4)

        dataset_info = {
            "normalizer": self.normalizer,
            "normalizer_latent": self.normalizer_latent,
            "skeleton_dict": None,
        }

        if model_name == "vae":
            self.model = EditDance(config, dataset_info, mode="inference")
        elif model_name == "cross_latent_diffusion":
            self.model = CrossEnergyEditDance(config, dataset_info, mode="inference")
        elif "joint" in model_name:
            self.model = CrossEnergyEditDanceJoint(config, dataset_info, mode="inference")
        else:
            return f"Unsupported model: {model_name}"

        self.model.eval().to(self.device)
        if progress_callback:
            progress_callback("Model loaded.", 1.0)
        return None

    # ============================================================ GENERATE

    def _prepare_feature_files(self, folder):
        """Scan and sort audio feature files."""
        folder = Path(folder)
        feature_dir = folder / "sliced_audio_features"
        files = sorted(
            feature_dir.glob("*_audio.npy"),
            key=lambda p: int(re.search(r"chunk(\d+)", p.stem).group(1))
            if re.search(r"chunk(\d+)", p.stem)
            else float("inf"),
        )
        return files

    def _build_chunk_labels(self, segments, chunk_start, chunk_frames, motion_fps):
        """Build per-frame label array and label_index for a chunk directly
        from the live segment list.

        Parameters
        ----------
        segments : list[dict]   – project segments with genre/label/description/start/end
        chunk_start : int       – first frame of this chunk (global frame index)
        chunk_frames : int      – number of frames in the chunk (e.g. 150)
        motion_fps : int

        Returns (label_arr, label_index) where
            label_arr  : np.ndarray (chunk_frames, 3) dtype str
            label_index: np.ndarray (chunk_frames,) dtype float
        """
        chunk_end = chunk_start + chunk_frames

        # Build per-frame (T, 3) string array: [genre, label, description]
        # T5 path uses object dtype so long descriptions aren't silently
        # truncated by numpy's fixed-width Unicode dtype. CLIP path keeps
        # <U500 (CLIP's tokenizer caps at 77 tokens anyway).
        if isinstance(self.label_converter, T5LabelConverter):
            label_arr = np.empty((chunk_frames, 3), dtype=object)
        else:
            label_arr = np.empty((chunk_frames, 3), dtype="<U500")
        label_arr.fill("")

        for seg in segments:
            s = seg.get("start", 0)
            e = seg.get("end", 0)
            # Overlap with this chunk?
            if s >= chunk_end or e <= chunk_start:
                continue
            # Clamp to chunk boundaries (relative indices)
            rel_s = max(s - chunk_start, 0)
            rel_e = min(e - chunk_start, chunk_frames)
            label_arr[rel_s:rel_e, 0] = seg.get("genre", "")
            label_arr[rel_s:rel_e, 1] = seg.get("label", "")
            label_arr[rel_s:rel_e, 2] = seg.get("description", "")

        # Build label_index: 0-to-1 ramp within each contiguous label span
        T = chunk_frames
        label_index = np.zeros(T, dtype=np.float32)
        initial = label_arr[0, 1]
        counter = 0
        chunks = []
        for i in range(T):
            cur = label_arr[i, 1]
            if cur != initial:
                chunks.append(np.arange(counter, dtype=np.float32) / max(counter - 1, 1))
                initial = cur
                counter = 0
            counter += 1
        chunks.append(np.arange(counter, dtype=np.float32) / max(counter - 1, 1))
        label_index = np.concatenate(chunks)

        return label_arr, label_index

    def _generate_chunk(self, chunk_idx, folder, segments, prev_normalized=None):
        """Generate a single chunk.

        Returns (motion_tensor, normalized_tensor).
        """
        folder = Path(folder)
        fpath = self._feature_files[chunk_idx]

        LABEL = self.config.model.name != "vae"
        motion_fps = self.config.data.motion_fps
        chunk_time = self.config.data.chunk_time
        body_features = self.config.data.body_features
        num_joints = self.config.data.num_joints
        stride = motion_fps * chunk_time // 2
        chunk_frames = motion_fps * chunk_time

        samples = torch.randn(
            (1, chunk_frames, body_features * num_joints + 3),
            device=self.device,
            dtype=torch.float,
        )

        feat = np.load(str(fpath), allow_pickle=True)[()]
        audio_tensor = torch.from_numpy(feat["audio"]).unsqueeze(0).to(self.device)

        model_input = {
            "audio": audio_tensor,
            "audio_mask": torch.tensor([True], dtype=torch.bool, device=self.device),
            "att_mask": torch.tensor([True], dtype=torch.bool, device=self.device),
            "noise": samples,
        }

        if LABEL:
            chunk_start = chunk_idx * stride
            label_arr, label_idx = self._build_chunk_labels(
                segments, chunk_start, chunk_frames, motion_fps
            )

            # Check if we have any labels at all
            has_labels = np.any(label_arr[:, 0] != "")
            if has_labels:
                attrs, desc = self.label_converter.class_label_to_embedding(label_arr)
                model_input["attributes"] = (
                    torch.from_numpy(attrs).unsqueeze(0).to(self.device)
                )
                if desc is not None:
                    model_input["description"] = (
                        torch.from_numpy(desc).unsqueeze(0).to(self.device)
                    )
                else:
                    model_input["description"] = torch.zeros_like(
                        model_input["attributes"]
                    )
                model_input["label_index"] = (
                    torch.from_numpy(label_idx).unsqueeze(0).to(self.device)
                )
            else:
                # No label coverage — unconditional generation. Embedding dim
                # depends on the converter (512 CLIP, 768 T5).
                T = chunk_frames
                emb_dim = self.label_converter.embedding_dim
                model_input["attributes"] = torch.zeros(
                    (1, T, emb_dim), device=self.device, dtype=torch.float
                )
                model_input["description"] = torch.zeros(
                    (1, T, emb_dim), device=self.device, dtype=torch.float
                )
                model_input["label_index"] = torch.zeros(
                    (1, T), device=self.device, dtype=torch.float
                )

        prefix_kwargs = {}
        if prev_normalized is not None:
            prefix_kwargs = {
                "prefix_motion": prev_normalized[:, -stride:, :],
                "prefix_frames": stride,
                "blend_frames": 20,
                "foot_optim": True,
            }

        result = self.model.forward(model_input, **prefix_kwargs)
        motion_recon = result["motion"]
        norm_motion = result.get("normalized_motion", None)

        return motion_recon, norm_motion

    @staticmethod
    def _smooth_motion(motion, fps=30, cutoff=0.5):
        """Apply a low-pass Butterworth filter to stitched SMPL params.

        Parameters
        ----------
        motion : np.ndarray (T, 147)
            3 translation + 24*6 rotation-6D.
        fps : int
            Motion frame rate.
        cutoff : float
            Cutoff frequency in Hz.

        Returns
        -------
        np.ndarray (T, 147)  – smoothed motion with valid rotations.
        """
        T = motion.shape[0]
        if T < 12:
            return motion

        # Design 2nd-order Butterworth low-pass
        nyq = fps / 2.0
        b, a = butter(2, cutoff / nyq, btype="low")

        smoothed = np.empty_like(motion)

        # Filter translation (dims 0-2)
        for d in range(3):
            smoothed[:, d] = filtfilt(b, a, motion[:, d])

        # Filter rotation 6D (dims 3-146) per column
        for d in range(3, motion.shape[1]):
            smoothed[:, d] = filtfilt(b, a, motion[:, d])

        # Re-project 6D rotations to valid rotation matrices via SVD
        rot6d = torch.from_numpy(smoothed[:, 3:].reshape(T, -1, 6)).float()
        mat = t3d.rotation_6d_to_matrix(rot6d)            # (T, 24, 3, 3)
        U, _, Vh = torch.linalg.svd(mat)
        # Correct reflections to ensure det = +1
        det = torch.det(U @ Vh)
        sign = torch.ones_like(Vh)
        sign[:, :, -1, :] *= det.unsqueeze(-1)
        mat_valid = U @ (sign * Vh)
        rot6d_valid = t3d.matrix_to_rotation_6d(mat_valid) # (T, 24, 6)
        smoothed[:, 3:] = rot6d_valid.reshape(T, -1).numpy()

        return smoothed

    def _stitch_and_fk(self, smooth=False, cutoff=0.5):
        """Stitch all stored chunks and run FK for joints + verts.

        Parameters
        ----------
        smooth : bool
            Apply low-pass Butterworth filter for smoother motion.
        cutoff : float
            Filter cutoff frequency in Hz (only used when smooth=True).
        """
        all_motions = torch.cat(self._chunk_motions, dim=0).detach().cpu().numpy()
        stitched = stitch(all_motions, self.fk, interpolate=True)

        if smooth:
            print("Applying Low-pass filter")
            fps = self.config.data.motion_fps
            stitched = self._smooth_motion(stitched, fps=fps, cutoff=cutoff)

        joints, verts = self.fk_mesh.forward(
            torch.from_numpy(stitched).float(), return_verts=True
        )
        joints = joints.detach().cpu().numpy()
        verts = verts.detach().cpu().numpy()
        faces = np.array(self.fk_mesh.smpl_model.faces, dtype=np.int32)

        return stitched, joints, verts, faces

    def generate(self, folder, segments, progress_callback=None,
                 smooth=True, cutoff=10):
        """Full generation: all chunks sequentially.

        Returns (smpl_params, joints, verts, faces) or raises.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        self._folder = str(folder)
        self._feature_files = self._prepare_feature_files(folder)
        self._num_chunks = len(self._feature_files)

        if self._num_chunks == 0:
            raise FileNotFoundError("No audio features found. Run data prep first.")

        self._chunk_motions = []
        self._chunk_normalized = []
        prev_normalized = None

        for i in range(self._num_chunks):
            motion, norm = self._generate_chunk(i, folder, segments, prev_normalized)
            self._chunk_motions.append(motion)
            self._chunk_normalized.append(norm)
            prev_normalized = norm

            if progress_callback:
                progress_callback(
                    f"Generating chunk {i+1}/{self._num_chunks}",
                    (i + 1) / self._num_chunks,
                )

        return self._stitch_and_fk(smooth=smooth, cutoff=cutoff)

    # ======================================================= PARTIAL REGEN

    def segments_to_chunks(self, segment_indices, segments, fps=30):
        """Map segment indices to overlapping chunk indices.

        Each chunk covers [i*stride_frames, i*stride_frames + window_frames).
        """
        if not self._feature_files:
            return []

        motion_fps = self.config.data.motion_fps
        chunk_time = self.config.data.chunk_time
        stride_time = chunk_time / 2.0  # 2.5s
        window_frames = int(chunk_time * motion_fps)    # 150
        stride_frames = int(stride_time * motion_fps)   # 75

        # Frame range covered by the edited segments
        seg_start = min(segments[i].get("start", 0) for i in segment_indices)
        seg_end = max(segments[i].get("end", 0) for i in segment_indices)

        affected = set()
        for c in range(self._num_chunks):
            c_start = c * stride_frames
            c_end = c_start + window_frames
            # Overlap check
            if c_start < seg_end and c_end > seg_start:
                affected.add(c)

        return sorted(affected)

    def regenerate_partial(self, folder, segments, chunk_indices,
                          progress_callback=None, smooth=True, cutoff=10):
        """Regenerate only the given chunks + 1 neighbor after, then re-stitch.

        Returns (smpl_params, joints, verts, faces).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if not self._chunk_motions:
            raise RuntimeError("No previous generation. Run full generate first.")

        folder = Path(folder)

        # Expand to include 1 neighbor chunk after the last affected chunk
        # for smooth transition out of the edited region
        expanded = set(chunk_indices)
        if chunk_indices:
            last = max(chunk_indices)
            if last + 1 < self._num_chunks:
                expanded.add(last + 1)
        regen_list = sorted(expanded)

        total = len(regen_list)
        for step, c in enumerate(regen_list):
            # Get prefix from the chunk before this one
            if c > 0:
                prev_norm = self._chunk_normalized[c - 1]
            else:
                prev_norm = None

            motion, norm = self._generate_chunk(c, folder, segments, prev_norm)
            self._chunk_motions[c] = motion
            self._chunk_normalized[c] = norm

            if progress_callback:
                progress_callback(
                    f"Regenerating chunk {step+1}/{total}",
                    (step + 1) / total,
                )

        return self._stitch_and_fk(smooth=smooth, cutoff=cutoff)

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _get_normalizer_path(config, root_dir, name="smpl"):
        ref_dataset = config.data.dataset
        if hasattr(config.data, "trained_dataset"):
            ref_dataset = config.data.trained_dataset

        n = "normalizer"
        if name == "smpl":
            n += "_smpl"
        elif name == "latent":
            n += "_latent"
        elif name == "joint":
            n += "_joint"

        if len(ref_dataset) == 1:
            n += "_" + ref_dataset[0].lower()
        elif len(ref_dataset) == 2:
            if "motorica" in ref_dataset and "AIST" in ref_dataset:
                n += "_dance_all"
            elif "motorica" in ref_dataset and "HumanML3D" in ref_dataset:
                n += "_motorica_humanml3d"
        elif len(ref_dataset) == 3:
            n += "_all"

        return os.path.join(root_dir, n + ".npy")
