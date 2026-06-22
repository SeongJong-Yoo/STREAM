from src.models.architecture.cross_energy_denoiser_joint import CrossEnergyDenoiserJoint
import numpy as np
from copy import deepcopy
import importlib
import os
from pathlib import Path
import pickle
import clip
import torch

def get_model(config, model_name):
    if model_name == 'cross_energy_diffusion_joint':
        return CrossEnergyDenoiserJoint(config)
    else:
        print(f"Model {model_name} not found")  
        return None
    
FOOT_IDX = {
    'motorica': [15, 16],
    'smpl': [7, 8, 10, 11]
}

def compute_foot_contact(motions):
    thresh_height = 0.05  # 10
    verts_feet = motions[:, :, [10, 11], :]  # [bs, max_len, 2, 3]

    foot_min = torch.stack([torch.min(verts_feet[i:i+1, :, :, 1]) for i in range(verts_feet.shape[0])], axis=0)
    while foot_min.ndim != verts_feet.ndim:
        foot_min = foot_min.unsqueeze(-1)
    
    verts_feet = verts_feet - foot_min
    verts_feet_height = verts_feet[:, :, :, 1]  # [bs,  max_len, 2]
    # If feet touch ground in adjacent frames
    feet_contact = torch.logical_and(
        verts_feet_height[:, :-1, :] < thresh_height,
        verts_feet_height[:, 1:, :] < thresh_height
    )  # [bs, max_len - 1, 2]
    return feet_contact

def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def resample(data, target_fps, src_fps, target_len):
    src_len = len(data)

    # Create time points for original and target data
    src_times = np.arange(src_len) / src_fps
    target_times = np.arange(target_len) / target_fps

    data_2d = data.reshape(src_len, -1)
    result = np.zeros((target_len, data_2d.shape[1]))

    for i in range(data_2d.shape[1]):
        result[:, i] = np.interp(target_times, src_times, data_2d[:, i])

    result_shape = list(data.shape)
    result_shape[0] = target_len
    return result.reshape(result_shape)


def read_label_segment_txt(path):
    """Read a label segment .txt file into a dictionary.

    Expected format (one key-value per line, 'key: value'):
        genre: popping
        label: tutting
        start: 1524
        end: 1623
        fps: 30
        description: ...
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


# class that convert label to number and number to label
class LabelConverter:
    def __init__(self, label_list_path = None):
        if label_list_path == None:
            print("INFO: No label list path provided, using text embedding conversion only")
            self.label_list = None
        else:
            self.label_list_path = Path(label_list_path)
            if not self.label_list_path.exists():
                raise FileNotFoundError(f"Label list file not found at {self.label_list_path}")
            self.load_label_list()
            
    def load_label_list(self):
        with open(self.label_list_path, "r") as f:
            self.label_list = [eval(line.strip()) for line in f.readlines()]
        # append [None, None] to the label list
        self.label_list.append([None, None])
        # Remove duplicates from the label list
        self.subclass_label_list = []
        for label in self.label_list:
            if label[1] not in self.subclass_label_list:
                self.subclass_label_list.append(label[1])
        self.subclass_label_dict = {label: i for i, label in enumerate(self.subclass_label_list)}
        self.subclass_number_dict = {i: label for i, label in enumerate(self.subclass_label_list)}

        # get unique class labels while keeping the order
        self.unique_labels = []
        for label in self.label_list:
            if label[1] not in self.unique_labels:
                self.unique_labels.append(label[1])
        self.class_label_dict = {label: i for i, label in enumerate(self.unique_labels)}
        self.class_number_dict = {i: label for i, label in enumerate(self.unique_labels)}

        print(f"Loaded {len(self.label_list)} labels from {self.label_list_path}")
        return self.label_list
    
    def subclass_label_to_number(self, label):
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        if label:
            if label in self.subclass_label_dict:
                return self.subclass_label_dict[label]
            else:
                raise ValueError(f"Label {label} not found in label list")
        else:
            return -1
    
    def subclass_number_to_label(self, number):
        if number in self.subclass_number_dict:
            return self.subclass_number_dict[number]
        else:
            raise ValueError(f"Number {number} not found in label list")
        
    def subclass_label_to_one_hot(self, label):
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        # convert label to one hot vector
        if label in self.subclass_label_dict:
            one_hot = np.zeros(len(self.label_list))
            one_hot[self.subclass_label_dict[label]] = 1
            return one_hot
        else:
            raise ValueError(f"Label {label} not found in label list")
        
    def subclass_one_hot_to_label(self, one_hot):
        # convert one hot vector to label
        if isinstance(one_hot, np.ndarray) and len(one_hot) == len(self.label_list):
            index = np.argmax(one_hot)
            return self.subclass_number_dict[index]
        else:
            raise ValueError(f"One hot vector {one_hot} not valid")
        
    
    def class_label_to_number(self, label):
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        if label in self.class_label_dict:
            return self.class_label_dict[label]
        else:
            raise ValueError(f"Label {label} not found in label list")
    
    def class_number_to_label(self, number):
        if number in self.class_number_dict:
            return self.class_number_dict
        else:
            raise ValueError(f"Number {number} not found in label list")
        
    def class_label_to_one_hot(self, label):
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        if label in self.class_label_dict:
            one_hot = np.zeros(len(self.unique_labels))
            one_hot[self.class_label_dict[label]] = 1
            return one_hot
        else:
            raise ValueError(f"Label {label} not found in label list")
    
    def class_one_hot_to_label(self, one_hot):
        if isinstance(one_hot, np.ndarray) and len(one_hot) == len(self.unique_labels):
            index = np.argmax(one_hot)
            return self.class_number_dict
        else:
            raise ValueError(f"One hot vector {one_hot} not valid")
        
        
class GeminiLabelConverter(LabelConverter):
    def __init__(self, label_list_path=None, embedding_path=None):
        super().__init__(label_list_path)
        if embedding_path == None:
            embedding_path = Path("./data/motorica/dance_technique_embeddings.pkl")
        with open(embedding_path, "rb") as f:
            self.embedding_dict = pickle.load(f)

    def class_label_to_embedding(self, label):
        if label.ndim == 2:
            T, L = label.shape
            if L == 2 or L == 3:
                concat_label = []
                for i in range(T):
                    if "FrameLabel" in label[i, 0]:
                        label[i, 0] = label[i, 0].replace(" FrameLabel", "")
                    if "HipHop" in label[i, 0]:
                        label[i, 0] = label[i, 0].replace("HipHop", "Hip_Hop")
                    concat_label.append(label[i, 0] + ': ' + label[i, 1])
        elif label.ndim == 1:
            concat_label = label
            # label = [label[i, 0] + ': ' + label[i, 1] for i in range(T)]
        embedding = np.array([self.embedding_dict[l] for l in concat_label], dtype=np.float32)
        return embedding

def _to_python_str(x):
    """Pull a Python ``str`` out of either a numpy fixed-width string scalar
    or an object-array element (already a Python ``str``). Single helper so
    ``class_label_to_embedding`` works on both ``<U1000`` and ``dtype=object``
    label arrays — the CLIP path uses the former (silent truncation OK because
    CLIP's tokenizer truncates further anyway), the T5 path uses the latter
    so long descriptions aren't truncated."""
    return x if isinstance(x, str) else x.item()


class CLIPLabelConverter(LabelConverter):
    embedding_dim = 512  # ViT-B/32 text encoder output dim

    def __init__(self, label_list_path=None, embedding_path=None, dance_motion=True, device=None, text_cache_path=None):
        super().__init__(label_list_path)
        # Allow callers to pin CLIP to CPU. Required by lazy-npy dataloaders:
        # DataLoader workers are forked from the parent and PyTorch forbids
        # using CUDA after fork ("Cannot re-initialize CUDA in forked
        # subprocess"). Defaults preserve the previous auto-detect behavior.
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.dance_motion = dance_motion
        if self.dance_motion:
            self.dance_motion_prefix = 'dance '
        else:
            self.dance_motion_prefix = ''
        # Persistent text→embedding cache. Built offline by
        # src/preprocessing/precompute_clip_cache.py and passed in via
        # text_cache_path. Misses fall through to live CLIP encoding so the
        # cache can be partial without breaking the dataloader.
        self.text_cache_path = text_cache_path
        self.text_cache = {}
        if text_cache_path is not None and os.path.exists(text_cache_path):
            try:
                with open(text_cache_path, "rb") as f:
                    self.text_cache = pickle.load(f)
                print(f"INFO: Loaded {len(self.text_cache)} CLIP text embeddings from {text_cache_path}")
            except Exception as e:
                print(f"WARNING: Failed to load CLIP text cache at {text_cache_path}: {e}")
                self.text_cache = {}

        # CLIP is loaded lazily — only when encode_text actually misses the
        # text cache. With a complete cache (built via
        # precompute_clip_cache.py) CLIP never loads, which is what avoids
        # ranks × workers × ~250 MB of GPU OOM under spawn.
        # if embedding_path == None:
        #     embedding_path = Path("./data/motorica/dance_technique_embeddings_clip.pkl")
        self.clip_model = None
        if embedding_path is None:
            self.embedding_dict = {}
        else:
            if embedding_path.exists():
                with open(embedding_path, "rb") as f:
                    self.embedding_dict = pickle.load(f)
            else:
                self.embedding_dict = {}

    def load_and_freeze_clip(self, clip_version='ViT-B/32'):
        clip_cache_dir = os.getenv("CLIP_CACHE_DIR")
        if clip_cache_dir:
            os.makedirs(clip_cache_dir, exist_ok=True)
            clip_model, clip_preprocess = clip.load(
                clip_version,
                device=self.device,
                jit=False,
                download_root=clip_cache_dir,
            )
        else:
            clip_model, clip_preprocess = clip.load(clip_version, device=self.device, jit=False)
        clip_model = clip_model.float()

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text):
        # Cache fast path: only single-string lookups go through the cache.
        # Lists are extremely rare in current callers, but keep the original
        # batched-tokenize behavior available as a fallback.
        if isinstance(raw_text, str):
            # Empty string -> zero embedding. Matches the convention already
            # used for empty descriptions in class_label_to_embedding, and
            # avoids loading CLIP just to encode "" (which would produce a
            # degenerate start/end-token-only vector anyway).
            if raw_text == "":
                return torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)
            cached = self.text_cache.get(raw_text)
            if cached is not None:
                return torch.from_numpy(cached).to(self.device)

        text = clip.tokenize(raw_text, truncate=True)
        text = text.to(self.device)
        if self.clip_model is None:
            preview_src = raw_text if isinstance(raw_text, str) else str(raw_text)
            preview = preview_src[:200]
            ellipsis = "…" if len(preview_src) > 200 else ""
            print(
                f"INFO: CLIP cache miss in worker (pid={os.getpid()}), loading "
                f"CLIP onto {self.device}. First missed text "
                f"(len={len(preview_src)}): {preview + ellipsis!r}"
            )
            self.clip_model = self.load_and_freeze_clip()
        feat_clip_text = self.clip_model.encode_text(text).float()
        feat = feat_clip_text.squeeze()
        # Cache cold misses for the rest of this worker's lifetime. The cache
        # is a dict on a module-local copy; we don't write it back to disk
        # from worker processes (would race across spawn workers + epochs).
        if isinstance(raw_text, str):
            self.text_cache[raw_text] = feat.detach().cpu().numpy().astype(np.float32)
        return feat

    def split_sample(self, data):
        T = len(data)
        current_value = data[0]
        if T==1:
            sliced_data_list = [[data[0], 0, 1]]
            return sliced_data_list
        sliced_data_list = []
        start_idx = 0
        for i in range(1, T):
            if data[i] != current_value:
                sliced_data_list.append([data[i-1], start_idx, i])
                start_idx = i
                current_value = data[i]
        sliced_data_list.append([data[i], start_idx, T])
        return sliced_data_list


    def class_label_to_embedding(self, label, use_description=True):
        if label.ndim == 2:
            T, L = label.shape
            if L == 2 or L == 3:
                concat_label = []
                description = []
                for i in range(T):
                    g = _to_python_str(label[i, 0])
                    l = _to_python_str(label[i, 1])
                    if "FrameLabel" in g:
                        g = g.replace(" FrameLabel", "")
                        label[i, 0] = g
                    if "HipHop" in g:
                        g = g.replace("HipHop", "Hip_Hop")
                        label[i, 0] = g
                    if l == '':
                        concat_label.append(g)
                    else:
                        concat_label.append(g + ': ' + l)
                    if use_description:
                        description.append(_to_python_str(label[i, 2]))
                    else:
                        description.append('')
            else:
                concat_label = label
                description = None
        elif label.ndim == 1:
            concat_label = label

        sliced_data_list = self.split_sample(concat_label)
        label_embedding_array = []
        description_embedding_array = []
        for data, start_idx, end_idx in sliced_data_list:
            label_embedding = self.encode_text(data)
            label_embedding_array.append(np.tile(label_embedding.detach().cpu().numpy().astype(np.float32), (end_idx - start_idx, 1)))

            if use_description:
                if description is None:
                    description_embedding_array = None
                elif description[start_idx] != '':
                    description_embedding = self.encode_text(description[start_idx])
                    description_embedding_array.append(np.tile(description_embedding.detach().cpu().numpy().astype(np.float32), (end_idx - start_idx, 1)))
                else:
                    if len(sliced_data_list) == 1:
                        description_embedding_array = None
                    else:
                        # Zero embedding; uses the converter's own embedding_dim
                        # (512 for CLIP ViT-B/32, 768 for T5-base) so this also
                        # works for the T5 subclass without touching the model.
                        description_embedding_array.append(np.zeros((end_idx-start_idx, self.embedding_dim), dtype=np.float32))

        if not use_description:
            print(f"INFO: No description provided, using zero embedding")
            # description_embedding_array = np.zeros(label_embedding_array.shape[-1], dtype=np.float32)
            description_embedding_array = None
        label_embedding_array = np.concatenate(label_embedding_array, dtype=np.float32)
        if description_embedding_array is not None:
            description_embedding_array = np.concatenate(description_embedding_array, dtype=np.float32)
        return label_embedding_array, description_embedding_array

            
        # if not self.embedding_dict:
        #     for l in concat_label:
        #         self.embedding_dict[l] = self.encode_text(l).detach().cpu().numpy().astype(np.float32)
        # embedding = np.array([self.embedding_dict[l] for l in concat_label], dtype=np.float32)
        # description_embedding = []
        # for d in description:
        #     if d in self.embedding_dict:
        #         description_embedding.append(self.embedding_dict[d])
        #     elif d == '':
        #         description_embedding.append(np.zeros(embedding.shape[-1]).astype(np.float32))
        #     else:
        #         print(f"Description {d} not found in embedding dict")
        #         description_embedding.append(self.encode_text(d).detach().cpu().numpy().astype(np.float32))

        # description_embedding = np.array(description_embedding, dtype=np.float32)
        # return embedding, description_embedding


class T5LabelConverter(CLIPLabelConverter):
    """T5-base text encoder converter.

    Inherits ``class_label_to_embedding`` and the cache plumbing from
    ``CLIPLabelConverter`` — only the underlying encoder changes. T5 has
    no hard token cap in its architecture (uses relative position bias),
    so we don't truncate the input. Long descriptions still pass through
    fully, unlike CLIP which caps at 77 tokens internally.

    The dataloader's ``label_chunk_record`` should be called with
    ``dtype=object`` when this converter is in use, so the per-frame
    label array doesn't silently truncate strings via numpy's fixed-width
    Unicode dtype.
    """

    embedding_dim = 768  # t5-base hidden size

    def __init__(self, model_name='t5-base', device=None, text_cache_path=None):
        # Skip CLIPLabelConverter.__init__ — we don't want it loading CLIP.
        # Reach up to LabelConverter directly.
        LabelConverter.__init__(self, label_list_path=None)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_name = model_name

        # Same persistent text cache mechanism as CLIPLabelConverter.
        self.text_cache_path = text_cache_path
        self.text_cache = {}
        if text_cache_path is not None and os.path.exists(text_cache_path):
            try:
                with open(text_cache_path, "rb") as f:
                    self.text_cache = pickle.load(f)
                print(f"INFO: Loaded {len(self.text_cache)} T5 text embeddings from {text_cache_path}")
            except Exception as e:
                print(f"WARNING: Failed to load T5 text cache at {text_cache_path}: {e}")
                self.text_cache = {}

        # Tokenizer is cheap to load eagerly; the encoder is heavy and
        # follows the same lazy-on-miss pattern as CLIP.
        from transformers import T5TokenizerFast  # local import: not used elsewhere
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.clip_model = None  # repurposed: holds T5EncoderModel; lazy-loaded.

        # Unused attributes inherited from the CLIPLabelConverter contract.
        self.embedding_dict = {}
        self.dance_motion = False
        self.dance_motion_prefix = ""

    def load_and_freeze_clip(self, clip_version=None):
        """Lazy-load T5EncoderModel. Name kept for API compatibility with
        the cache-miss path in ``encode_text``."""
        from transformers import T5EncoderModel
        model = T5EncoderModel.from_pretrained(self.model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model = model.to(self.device)
        return model

    def encode_text(self, raw_text):
        if isinstance(raw_text, str):
            if raw_text == "":
                return torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)
            cached = self.text_cache.get(raw_text)
            if cached is not None:
                return torch.from_numpy(cached).to(self.device)

        # No truncation — T5 was trained at 512 tokens but the model
        # architecture handles arbitrary length via relative position bias.
        # The user explicitly asked not to truncate.
        encoded = self.tokenizer(
            raw_text,
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        if self.clip_model is None:
            preview_src = raw_text if isinstance(raw_text, str) else str(raw_text)
            preview = preview_src[:200]
            ellipsis = "…" if len(preview_src) > 200 else ""
            print(
                f"INFO: T5 cache miss in worker (pid={os.getpid()}), loading "
                f"T5 onto {self.device}. First missed text "
                f"(len={len(preview_src)}, tokens={input_ids.shape[-1]}): "
                f"{preview + ellipsis!r}"
            )
            self.clip_model = self.load_and_freeze_clip()

        with torch.no_grad():
            out = self.clip_model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over non-pad tokens (with padding=False there are no pads
        # in the single-sequence case, but keep the masked mean for safety).
        last_hidden = out.last_hidden_state  # (1, T, 768)
        mask = attention_mask.unsqueeze(-1).float()  # (1, T, 1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        feat = pooled.squeeze(0).float()  # (768,)

        if isinstance(raw_text, str):
            self.text_cache[raw_text] = feat.detach().cpu().numpy().astype(np.float32)
        return feat
