"""TMR (Text-to-Motion Retrieval) wrapper for clip-similarity scoring.

Handles the namespace conflict between this project's ``src`` and
TMR's ``src`` by temporarily swapping sys.path / sys.modules during
model loading.
"""

import os
import sys
import numpy as np
import torch


class TMRWrapper:
    """Thin wrapper around a loaded TMR model and its utilities."""

    def __init__(self, tmr_dir, run_dir, device, ckpt_name="last"):
        self.device = device
        (self.model, self.text_model, self.normalizer,
         self.collate_fn, self.score_fn, self.j2g_fn) = \
            _load_tmr_components(tmr_dir, run_dir, device, ckpt_name)

    # ── encoding ────────────────────────────────────────────

    def encode_motions(self, joints_fk):
        """Convert FK joints -> guoh3dfeats (263-d), normalize, encode.

        Parameters
        ----------
        joints_fk : Tensor (B, T, 24, 3)

        Returns
        -------
        Tensor (B, latent_dim)
        """
        x_dicts = []
        for j in range(joints_fk.shape[0]):
            feats = self.j2g_fn(joints_fk[j].numpy())   # (T-1, 263)
            motion = torch.from_numpy(feats).float()
            motion = self.normalizer(motion)
            motion = motion.to(self.device)
            x_dicts.append({"x": motion, "length": len(motion)})
        batched = self.collate_fn(x_dicts)
        with torch.inference_mode():
            latent = self.model.encode(batched, sample_mean=True)  # (B, D)
        return latent

    def encode_texts(self, texts):
        """Tokenize texts and encode through TMR.

        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        Tensor (B, latent_dim)
        """
        text_x_dicts = self.text_model(texts)
        batched = self.collate_fn(text_x_dicts)
        with torch.inference_mode():
            latent = self.model.encode(batched, sample_mean=True)  # (B, D)
        return latent

    def score(self, lat_text, lat_motion):
        """Pairwise score matrix in [0, 1].

        Parameters
        ----------
        lat_text   : Tensor (B1, D)
        lat_motion : Tensor (B2, D)

        Returns
        -------
        Tensor (B1, B2)
        """
        return self.score_fn(lat_text, lat_motion)

    def paired_scores(self, joints_fk, texts):
        """Convenience: encode + diagonal similarity for paired data.

        Parameters
        ----------
        joints_fk : Tensor (B, T, 24, 3)
        texts     : list[str]  (length B)

        Returns
        -------
        list[float]  length B – per-sample similarity in [0, 1]
        """
        lat_m = self.encode_motions(joints_fk)
        lat_t = self.encode_texts(texts)
        scores = self.score(lat_t, lat_m)          # (B, B)
        return [scores[j, j].item() for j in range(len(texts))]


# ── private loader ──────────────────────────────────────────

def _load_tmr_components(tmr_dir, run_dir, device, ckpt_name="last"):
    """Load TMR model, text encoder, normalizer, collate, score, j2g."""
    saved_modules = {k: v for k, v in sys.modules.items()
                     if k == 'src' or k.startswith('src.')}
    for k in saved_modules:
        del sys.modules[k]
    saved_path = sys.path.copy()
    saved_cwd = os.getcwd()
    tmr_dir = os.path.abspath(tmr_dir)
    sys.path.insert(0, tmr_dir)
    os.chdir(tmr_dir)

    try:
        import src.prepare  # noqa
        from src.config import read_config
        from src.load import load_model_from_cfg
        from src.data.collate import collate_x_dict as _collate
        from src.model.tmr import get_score_matrix as _score
        from src.guofeats import joints_to_guofeats as _j2g
        from hydra.utils import instantiate

        cfg = read_config(run_dir)
        text_model = instantiate(cfg.data.text_to_token_emb, device=device)
        tmr_model = load_model_from_cfg(cfg, ckpt_name=ckpt_name,
                                        eval_mode=True, device=device)
        normalizer = instantiate(cfg.data.motion_loader.normalizer)

        collate_fn = _collate
        score_fn = _score
        j2g_fn = _j2g
    finally:
        os.chdir(saved_cwd)
        tmr_src = [k for k in sys.modules
                   if k == 'src' or k.startswith('src.')]
        for k in tmr_src:
            del sys.modules[k]
        sys.modules.update(saved_modules)
        sys.path = saved_path

    return tmr_model, text_model, normalizer, collate_fn, score_fn, j2g_fn
