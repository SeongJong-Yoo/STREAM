from pathlib import Path
import numpy as np
import os
import copy
import torch
from tqdm import tqdm
import pickle
import lmdb

from .base_dataset import BaseDataset
# from .utility import create_h5_dataset
from src.skeleton.preprocessing import motion_preprocessing, motion_postprocessing
from src.skeleton.smpl_fk import SMPLModel
import h5py
from src.utils.utility import LabelConverter, GeminiLabelConverter, CLIPLabelConverter, T5LabelConverter
from src.preprocessing.label_format import (
    label_chunk_record,
    read_compact_label,
)

AIST_TEST_KEY = [
        "mBR0"
        "mLH4",
        "mKR2",
        "mBR0",
        "mLO2",
        "mJB5",
        "mWA0",
        "mJS3",
        "mMH3",
        "mHO5",
        "mPO1",
]

class AISTDataset(BaseDataset):
    def __init__(self, config, id=None):
        super().__init__(config)
        if config.data.beat_based:
            print("INFO: Using AIST_beat based dataset")
            self.beat_based = True

        self.name = 'AIST'
        
        self.data_dir = os.path.join(self.root_dir, 'AIST')
        if self.beat_based:
            self.data_dir = self.data_dir + '_beats'
        # self.ignore_list = []
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

        self.test_key = AIST_TEST_KEY

        # Initialize label converter
        if config.data.label_converter == 'one-hot':
            self.label_converter = LabelConverter()
            self.convert_to_number = np.vectorize(self.label_converter.subclass_label_to_number)
        elif config.data.label_converter == 'gemini':
            self.label_converter = GeminiLabelConverter(label_list_path=Path(os.path.join(self.data_dir, 'class_list.txt')),
                                                        embedding_path=Path(os.path.join(self.data_dir, 'dance_technique_embeddings.pkl')))
            # self.convert_to_embedding = np.vectorize(self.label_converter.class_label_to_embedding)
        elif config.data.label_converter == 'clip':
            text_cache_path = os.path.join(self.data_dir, 'label_clip_cache.pkl')
            self.label_converter = CLIPLabelConverter(embedding_path=Path(os.path.join(self.data_dir, 'dance_technique_embeddings_clip.pkl')), text_cache_path=text_cache_path)
        elif config.data.label_converter == 't5':
            text_cache_path = os.path.join(self.data_dir, 'label_t5_cache.pkl')
            self.label_converter = T5LabelConverter(text_cache_path=text_cache_path)
        else:
            raise ValueError(f"Unknown label converter: {config.data.label_converter}")
        # if config.data.num_attributes != len(self.label_converter.subclass_label_list) - 1:
        #     raise ValueError(f"Number of attributes mismatch: {config.data.num_attributes} != {len(self.label_converter.subclass_label_list) - 1}")
        
        if self.lmdb_load:
            lmdb_path = os.path.join(self.data_dir, 'lmdb_dataset')
            if not os.path.exists(lmdb_path):
                self.create_lmdb_dataset(lmdb_path)
            key_path = os.path.join(self.data_dir, 'key_list.txt')
            if not os.path.exists(key_path):
                self.create_key_list_from_lmdb(lmdb_path, key_path)
            with open(key_path, 'r') as f:
                self._data_keys = [line.strip() for line in f.readlines()]
        elif self.h5_load and id is None:
            h5_path = os.path.join(self.data_dir, 'h5_dataset.h5')
            if not os.path.exists(h5_path):
                self.create_h5_dataset(h5_path)
            key_path = os.path.join(self.data_dir, 'key_list.txt')
            if not os.path.exists(key_path):
                self.create_key_list(key_path)
            with open(key_path, 'r') as f:
                self._data_keys = [line.strip() for line in f.readlines()]
        elif self.lazy_load:
            self.list_data_keys(id)
        else:
            self.load_data(id)
            self._data_keys = list(self._data.keys())

        print(f"AIST: Total Chunks loaded: {len(self._data_keys)}")

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
        output = super().load_single_data(id, h5_packing)
        label_dir = os.path.join(self.data_dir, 'sliced_labels')
        label_json = os.path.join(label_dir, f'{id}_label.json')
        label_npy = os.path.join(label_dir, f'{id}_label.npy')
        # T5 path stores full strings — no <U1000 truncation.
        label_dtype = object if self.config.data.label_converter == 't5' else "<U1000"
        if os.path.exists(label_json):
            payload = read_compact_label(label_json)
            record = label_chunk_record(payload, dtype=label_dtype)
            output['label'] = record['data']
            output['label_index'] = record['label_index']
            if self.config.data.use_neutralized_motion:
                neutral_json = label_json.replace('sliced_labels', 'sliced_labels_neutralized')
                neutral_npy = label_npy.replace('sliced_labels', 'sliced_labels_neutralized')
                if os.path.exists(neutral_json):
                    output['label_neutral'] = label_chunk_record(read_compact_label(neutral_json))
                elif os.path.exists(neutral_npy):
                    output['label_neutral'] = np.load(neutral_npy, allow_pickle=True)[()]
        elif os.path.exists(label_npy):
            output['label'] = np.load(label_npy, allow_pickle=True)[()]['data']
            output['label_index'] = np.load(label_npy, allow_pickle=True)[()]['label_index']
            if self.config.data.use_neutralized_motion:
                output['label_neutral'] = np.load(label_npy.replace('sliced_labels', 'sliced_labels_neutralized'), allow_pickle=True)[()]
        else:
            print(f'WARNING: {id} does not have label')
            return None

        return output

    def __getitem__(self, key):
        if self.lmdb_load:
            return self._load_lmdb_data(key)
        if self.h5_load:
            h5_data = {name: self.h5_file[key][name][()] for name in self.h5_file[key].keys() if name !='motion'}
            h5_data['motion'] = {name: self.h5_file[key]['motion'][name][()] for name in self.h5_file[key]['motion'].keys()}
            h5_data['label'] = h5_data['label'].astype(str)
            data = self.data_packing(h5_data, key, preprocess=True)

            return data
        else:
            return super().__getitem__(key)


if __name__ == '__main__':
    from src.utils.train.train_util import load_config
    config = load_config()  
    dataset = AISTDataset(config)
    data = dataset[0]
    test=1