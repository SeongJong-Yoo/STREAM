"""Precompute CLIP text embeddings for every label string a dataloader will
ever ask for, and write a `text → (512,) float32` lookup pkl.

The dataloader's CLIPLabelConverter checks this pkl on each ``encode_text``
call before falling through to a live GPU CLIP forward. Walking the dataset
once eliminates per-sample CLIP cost during training without rebuilding LMDB.

Output:
    {data_dir}/label_clip_cache.pkl   # dict[str, np.ndarray(512,) float32]

Usage:
    python -m src.preprocessing.precompute_clip_cache --dataset motorica
    python -m src.preprocessing.precompute_clip_cache --dataset HumanML3D
    python -m src.preprocessing.precompute_clip_cache --dataset AIST

Re-running is safe: existing entries are kept, only missing texts are encoded.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.label_format import (  # noqa: E402
    expand_segments_to_per_frame,
    num_description_candidates,
    read_compact_label,
    truncate_for_label_array,
)


DATASET_DIR = {
    "motorica": "data/motorica",
    "AIST": "data/AIST",
    "HumanML3D": "data/HumanML3D",
}


def _normalize_genre(genre: str) -> str:
    """Mirror the cleanup CLIPLabelConverter.class_label_to_embedding does
    on column 0 before building the concat string."""
    if "FrameLabel" in genre:
        genre = genre.replace(" FrameLabel", "")
    if "HipHop" in genre:
        genre = genre.replace("HipHop", "Hip_Hop")
    return genre


def _texts_from_per_frame(per_frame: np.ndarray, *, truncate: bool = True):
    """Extract the unique label strings + descriptions a chunk would produce.

    ``truncate=True`` matches the silent truncation that the CLIP dataloader's
    per-frame ``<U1000`` numpy array applies. ``truncate=False`` is for the
    T5 path where the dataloader uses ``dtype=object`` and stores full
    strings.
    """
    out = set()
    if per_frame.shape[1] not in (2, 3):
        return out
    for row in per_frame:
        genre = _normalize_genre(str(row[0]))
        label = str(row[1])
        if truncate:
            genre = truncate_for_label_array(genre)
            label = truncate_for_label_array(label)
        if label == "":
            out.add(genre)
        else:
            concat = f"{genre}: {label}"
            out.add(truncate_for_label_array(concat) if truncate else concat)
        if per_frame.shape[1] == 3:
            desc = str(row[2])
            if truncate:
                desc = truncate_for_label_array(desc)
            if desc:
                out.add(desc)
    return out


def _texts_from_json(payload: dict, *, truncate: bool = True):
    """Texts hidden behind a JSON payload's segments (incl. all description
    candidates for the HumanML3D-style ``descriptions: [...]`` shape)."""
    out = set()
    for seg in payload.get("segments", []):
        genre = _normalize_genre(str(seg.get("genre", "")))
        label = str(seg.get("label", ""))
        if truncate:
            genre = truncate_for_label_array(genre)
            label = truncate_for_label_array(label)
        if label == "":
            out.add(genre)
        else:
            concat = f"{genre}: {label}"
            out.add(truncate_for_label_array(concat) if truncate else concat)
        if "descriptions" in seg:
            for d in seg["descriptions"]:
                d = str(d)
                if truncate:
                    d = truncate_for_label_array(d)
                if d:
                    out.add(d)
        elif "description" in seg:
            d = str(seg["description"])
            if truncate:
                d = truncate_for_label_array(d)
            if d:
                out.add(d)
    return out


def collect_dataset_texts(data_dir: str, *, truncate: bool = True):
    """Walk all label files under ``data_dir/sliced_labels`` (json + npy)
    and the legacy ``data_dir/sliced_labels_old`` (npy only) collecting
    every unique text the dataloader could pass to ``encode_text``.

    ``truncate=True`` (default) is for the CLIP path. ``truncate=False`` is
    for T5, which uses ``dtype=object`` in the dataloader's label array and
    therefore preserves full-length strings.
    """
    label_dirs = [
        os.path.join(data_dir, "sliced_labels"),
        os.path.join(data_dir, "sliced_labels_old"),
    ]
    seen_files = set()
    texts = set()

    for ldir in label_dirs:
        if not os.path.isdir(ldir):
            continue
        files = sorted(os.listdir(ldir))
        for f in tqdm(files, desc=f"scan {os.path.basename(ldir)}"):
            full = os.path.join(ldir, f)
            chunk_id_root = f.split("_label.")[0]
            # Prefer the JSON version when both exist for the same chunk.
            if chunk_id_root in seen_files:
                continue
            if f.endswith(".json"):
                payload = read_compact_label(full)
                texts |= _texts_from_json(payload, truncate=truncate)
                seen_files.add(chunk_id_root)
            elif f.endswith(".npy"):
                try:
                    d = np.load(full, allow_pickle=True)[()]
                    per_frame = d.get("data") if isinstance(d, dict) else None
                    if per_frame is None:
                        continue
                    texts |= _texts_from_per_frame(per_frame, truncate=truncate)
                    seen_files.add(chunk_id_root)
                except Exception as e:
                    print(f"WARNING: skipped {full}: {e}")

    # Drop empty strings — encode_text short-circuits "" to a zero embedding,
    # so it never gets queried in the cache.
    texts.discard("")
    return texts


def encode_in_batches_clip(clip_model, device, texts, batch_size=256):
    """Run CLIP text encoder on ``texts`` in batches → dict[str, ndarray(512,)]."""
    import clip

    out = {}
    text_list = list(texts)
    for i in tqdm(range(0, len(text_list), batch_size), desc="CLIP encode"):
        batch = text_list[i : i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens).float()
        feats_np = feats.detach().cpu().numpy().astype(np.float32)
        for s, e in zip(batch, feats_np):
            out[s] = e
    return out


def encode_in_batches_t5(t5_model, tokenizer, device, texts, batch_size=64):
    """Run T5 encoder on ``texts`` in batches → dict[str, ndarray(768,)].

    Mean-pools the encoder's last hidden states over non-pad tokens. No
    truncation: pads to the longest sequence in each batch.
    """
    out = {}
    text_list = list(texts)
    for i in tqdm(range(0, len(text_list), batch_size), desc="T5 encode"):
        batch = text_list[i : i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=False,
            padding=True,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = t5_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, 768)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        feats_np = pooled.detach().cpu().numpy().astype(np.float32)
        for s, e in zip(batch, feats_np):
            out[s] = e
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_DIR.keys()),
        help="Which dataset to scan (uses ./data/<dataset>/sliced_labels[_old]).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the data directory for the given dataset.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output pkl path. Defaults to <data_dir>/label_<encoder>_cache.pkl.",
    )
    parser.add_argument(
        "--encoder",
        default="clip",
        choices=("clip", "t5"),
        help="Which text encoder to precompute embeddings for.",
    )
    parser.add_argument(
        "--t5-model",
        default="t5-base",
        help="HuggingFace T5 model id when --encoder=t5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size (256 for CLIP, 64 for T5).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda or cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Discard any existing cache and re-encode from scratch.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or DATASET_DIR[args.dataset]
    default_out_name = f"label_{args.encoder}_cache.pkl"
    out_path = args.out or os.path.join(data_dir, default_out_name)

    # CLIP path truncates strings to LABEL_STRING_MAX_LEN (1000) so cache keys
    # match the dataloader's <U1000 truncation. T5 path uses dtype=object in
    # the dataloader, so keys must be the full strings.
    truncate = (args.encoder == "clip")
    print(f"Scanning labels under {data_dir} (truncate={truncate}) ...")
    texts = collect_dataset_texts(data_dir, truncate=truncate)
    print(f"Found {len(texts)} unique label/description strings.")

    existing = {}
    if not args.rebuild and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            existing = pickle.load(f)
        print(f"Existing cache has {len(existing)} entries. Skipping those.")

    todo = sorted(texts - set(existing.keys()))
    if not todo:
        print("Nothing to do — cache is already complete.")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.encoder == "clip":
        print(f"Encoding {len(todo)} new texts via CLIP on {device} ...")
        import clip
        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        # Keep CLIP in fp32 — see notes in CLIPLabelConverter.load_and_freeze_clip.
        clip_model = clip_model.float()
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        bs = args.batch_size or 256
        new_entries = encode_in_batches_clip(clip_model, device, todo, batch_size=bs)
    else:
        print(f"Encoding {len(todo)} new texts via T5 ({args.t5_model}) on {device} ...")
        from transformers import T5EncoderModel, T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained(args.t5_model)
        t5_model = T5EncoderModel.from_pretrained(args.t5_model).to(device)
        t5_model.eval()
        for p in t5_model.parameters():
            p.requires_grad = False
        bs = args.batch_size or 64
        new_entries = encode_in_batches_t5(t5_model, tokenizer, device, todo, batch_size=bs)

    merged = dict(existing)
    merged.update(new_entries)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {len(merged)} entries to {out_path} ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()
