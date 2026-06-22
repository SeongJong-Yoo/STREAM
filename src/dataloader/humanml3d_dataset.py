import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml, os
import random
import numpy as np
import smplx
import pandas as pd
import json
import pickle

from .base_dataset import BaseDataset
from src.skeleton.preprocessing import motion_preprocessing, motion_postprocessing
from src.skeleton.smpl_fk import SMPLModel
import h5py
from src.utils.utility import CLIPLabelConverter, T5LabelConverter
from src.dataloader.glove import GloVe
from src.preprocessing.label_format import (
    label_chunk_record,
    num_description_candidates,
    read_compact_label,
)
class Humanml3D(BaseDataset):
    def __init__(self, config, id=None):
        super().__init__(config)
        
        self.name = 'HumanML3D'
        self.evaluation = getattr(config, 'evaluation', False)

        self.data_dir = os.path.join(self.root_dir, 'HumanML3D')
        with open(os.path.join(self.data_dir, 'val.txt'), 'r') as f:
            self.test_key = f.read().splitlines()
        
        if os.path.exists(os.path.join(self.data_dir, 'ignore_list.txt')):
            with open(os.path.join(self.data_dir, 'ignore_list.txt'), 'r') as f:
                self.ignore_list = f.read().splitlines()

        self.motion_fps = config.data.motion_fps
        self.audio_fps = config.data.audio_fps
        self.ratio = self.audio_fps / self.motion_fps

        self.normalize_data = config.data.normalize_data     

        if self.config.data.motion_preprocessing == 'relative_pelvis_origin':
            self.forward_type = 'relative_pose'
        else:
            self.forward_type = None

        # Initialize label converter
        if config.data.label_converter == 'clip':
            text_cache_path = os.path.join(self.data_dir, 'label_clip_cache.pkl')
            self.label_converter = CLIPLabelConverter(embedding_path=Path(os.path.join(self.data_dir, 'dance_technique_embeddings_clip.pkl')), dance_motion=False, text_cache_path=text_cache_path)
        elif config.data.label_converter == 't5':
            text_cache_path = os.path.join(self.data_dir, 'label_t5_cache.pkl')
            self.label_converter = T5LabelConverter(text_cache_path=text_cache_path)
        else:
            raise ValueError(f"Unknown label converter: {config.data.label_converter}")
        
        if self.lmdb_load:
            lmdb_path = os.path.join(self.data_dir, 'lmdb_dataset')
            if not os.path.exists(lmdb_path):
                self.create_lmdb_dataset(lmdb_path)
            key_path = os.path.join(self.data_dir, 'key_list.txt')
            if not os.path.exists(key_path):
                self.create_key_list_from_lmdb(lmdb_path, key_path)
            with open(key_path, 'r') as f:
                self._data_keys = [line.strip() for line in f.readlines()]
        elif self.lazy_load:
            self.list_data_keys(id)
        else:
            self.load_data(id)
            self._data_keys = list(self._data.keys())

        print(f"HumanML3D: Total Chunks loaded: {len(self._data_keys)}")
        if self.evaluation:
            self.w_vectorizer = GloVe(meta_root='./src/dataloader/glove', prefix='our_vab')

    def data_packing(self, data, id, preprocess=True):
        output = super().data_packing(data, id, preprocess)
        
        if 'label' in data:
            if 'attributes' not in data:
                if self.config.data.label_converter == 'clip' or self.config.data.label_converter == 't5':
                    output['attributes'], output['description'] = self.label_converter.class_label_to_embedding(data['label'])
            else:
                output['attributes'] = data['attributes']
                if 'description' in data:
                    output['description'] = data['description']
            output['label'] = data['label']
            if 'label_index' in data:
                output['label_index'] = data['label_index']
            self.is_att = True

        return output

    def load_single_data(self, id, h5_packing=False):
        output = {}
        motion_file = os.path.join(self.data_dir, 'sliced_motion', f'{id}_motion.npy')
        if self.rep.lower() == 'smpl':
            motion_file = os.path.join(self.data_dir, 'sliced_motion_smpl', f'{id}_motion.npy')

        # Load motion
        motion = np.load(motion_file, allow_pickle=True)[()]
        output['current_motion_fps'] = motion['current_fps']
        output['tgt_motion_fps'] = motion['target_fps']
        output['motion'] = motion['motion']
        if self.use_neutralized_motion is not None:
            output['neutralized_motion'] = motion['motion']
        else:
            output['neutralized_motion'] = None

        output['current_audio_fps'] = self.audio_fps
        output['tgt_audio_fps'] = self.audio_fps

        # if isinstance(motion, np.ndarray):
        #     motion_length = motion.shape[0]
        # elif isinstance(motion, dict):
        #     motion_length = output['motion']['motion_data'].shape[0]
        # else:
        #     raise ValueError(f'Unknown motion type: {type(motion)}')
        
        # HumanML3D doesn't have audio
        # if not self.motion_only:
        #     output['audio'] = np.zeros((motion_length, self.config.data.audio_features), dtype=np.float32)
        #     output['current_audio_fps'] = self.audio_fps
        #     output['tgt_audio_fps'] = self.audio_fps

        # Load label — prefer compact JSON. HumanML3D segments carry a list
        # of candidate descriptions; pick one at random per __getitem__ call
        # so different epochs see different captions (decision (b) in the
        # refactor plan).
        label_dir = os.path.join(self.data_dir, 'sliced_labels')
        label_json = os.path.join(label_dir, f'{id}_label.json')
        label_npy = os.path.join(label_dir, f'{id}_label.npy')
        # T5 path stores full strings — no <U1000 truncation.
        label_dtype = object if self.config.data.label_converter == 't5' else "<U1000"
        if os.path.exists(label_json):
            payload = read_compact_label(label_json)
            n_candidates = num_description_candidates(payload)
            desc_idx = random.randrange(n_candidates) if n_candidates > 1 else 0
            record = label_chunk_record(payload, description_idx=desc_idx, dtype=label_dtype)
            output['label'] = record['data']
            output['label_index'] = record['label_index']
            if self.config.data.use_neutralized_motion:
                output['label_neutral'] = output['label'].copy()
        elif os.path.exists(label_npy):
            output['label'] = np.load(label_npy, allow_pickle=True)[()]['data']
            output['label_index'] = np.load(label_npy, allow_pickle=True)[()]['label_index']
            if self.config.data.use_neutralized_motion:
                output['label_neutral'] = output['label'].copy() # HumanML3D data doesn't have neutralized motion
        else:
            print(f'WARNING: {id} does not have label')
            return None

        return output

    def gen_pose_one_hots(self, key, label):
        key = key.split('_chunk')[0]
        data_path = os.path.join(self.data_dir, 'texts', key + '.txt')
        text = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]
                tokens = line_split[1].split(' ')
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict['caption'] = caption
                text_dict['tokens'] = tokens
                text.append(text_dict)

        ref_text = [label[0, -1]]
        for i in range(len(label)):
            if ref_text[-1] != label[i, -1]:
                ref_text.append(label[i, -1])
        token = None
        for i in range(len(text)):
            for ref in ref_text:
                if ref.lower() in text[i]['caption'].lower():
                    token = text[i]['tokens']
                    ref_text = ref
                    break
            if token is not None:
                break

        if token is None:
            tokens = ['unk/OTHER'] * 198
            sent_len = 0
        else:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (196 + 2 - sent_len) # Max length of the tokens is 196 hard-coded
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_vec, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_vec[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return pos_one_hots, sent_len, ref_text, word_embeddings

    def __getitem__(self, key):
        if self.lmdb_load:
            if not self.evaluation:
                return self._load_lmdb_data(key)
            else:
                data = self._load_lmdb_data(key)
                pos_one_hots, sent_len, ref_text, word_embeddings = self.gen_pose_one_hots(key, data['label'])
                data['pos_one_hots'] = pos_one_hots
                data['sent_len'] = sent_len
                data['text'] = ref_text
                data['word_embeddings'] = word_embeddings
                return data
        else:
            data = super().__getitem__(key)
            if self.evaluation and 'label' in data:
                pos_one_hots, sent_len, ref_text, word_embeddings = self.gen_pose_one_hots(key, data['label'])
                data['pos_one_hots'] = pos_one_hots
                data['sent_len'] = sent_len
                data['text'] = ref_text
                data['word_embeddings'] = word_embeddings
            return data

if __name__ == '__main__':
    from src.utils.train.train_util import load_config
    config = load_config()  
    dataset = Humanml3D(config, evaluation=True)
    data = dataset[0]
    test=1