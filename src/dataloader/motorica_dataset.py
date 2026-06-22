import numpy as np
import os
import copy
from copy import deepcopy
import torch
import h5py
import pickle
import lmdb
from pathlib import Path

from .base_dataset import BaseDataset
from .utility import load_pickle_data, save_pickle_data
from src.skeleton.preprocessing import motion_preprocessing
from tqdm import tqdm

from src.skeleton.preprocessing import motion_postprocessing
from src.skeleton.utility import get_motorica_skeleton_names
from src.skeleton.forward_kinematics import ForwardKinematics
from src.skeleton.smpl_fk import SMPLModel
from src.utils.utility import LabelConverter, GeminiLabelConverter, CLIPLabelConverter, T5LabelConverter, read_label_segment_txt
from src.preprocessing.label_format import (
    label_chunk_record,
    num_description_candidates,
    read_compact_label,
)


MOTORICA_TEST_KEY = [
    "kthjazz_gCH_sFM_cAll_d02_mCH_ch01_whitemanpaulandhisorchestraloisiana_006",
    "kthjazz_gJZ_sFM_cAll_d02_mJZ_ch01_bennygoodmansugarfootstomp_003",
    "kthjazz_gTP_sFM_sngl_d02_015",
    "kthmisc_gCA_sFM_cAll_d01_mCA_ch24",
    "kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001",
    "kthstreet_gLH_sFM_cAll_d01_mLH_ch01_thisisit_001",
    "kthstreet_gLH_sFM_cAll_d02_mLH_ch01_lala_001",
    "kthstreet_gLO_sFM_cAll_d02_mLO_ch01_arethafranklinrocksteady_002",
    "kthstreet_gPO_sFM_cAll_d01_mPO_ch01_bombom_002",
    "kthstreet_gPO_sFM_cAll_d02_mPO_ch01_bombom_001"
]

class MotoricaDataset(BaseDataset):
    def __init__(self, config, id=None):
        super().__init__(config)
        if config.data.beat_based:
            print("INFO: Using Beat based dataset")
            self.beat_based = True

        self.name = 'motorica'
        self.data_dir = os.path.join(self.root_dir, 'motorica')
        if self.beat_based:
            self.data_dir = self.data_dir + '_beats'

        self.motion_fps = config.data.motion_fps
        self.audio_fps = config.data.audio_fps
        self.ratio = self.audio_fps / self.motion_fps
        self.dataset_list = config.data.dataset
        self.model_type = config.model.name
        self.skeleton_dict = {}

        self.normalize_data = config.data.normalize_data
        
        self.use_description = True
        if 'all' in config.data.name.lower():
            self.use_description = False
        
        if self.config.data.motion_preprocessing == 'relative_pelvis_origin':
            self.forward_type = 'relative_pose'
        else:
            self.forward_type = None

        self.test_key = MOTORICA_TEST_KEY

        # Initialize label converter
        if config.data.label_converter == 'one-hot':
            self.label_converter = LabelConverter()
            self.convert_to_number = np.vectorize(self.label_converter.subclass_label_to_number)
        elif config.data.label_converter == 'gemini':
            self.label_converter = GeminiLabelConverter(label_list_path=Path(os.path.join(self.data_dir, 'class_list.txt')),
                                                        embedding_path=Path(os.path.join(self.data_dir, 'dance_technique_embeddings.pkl')))
            # self.convert_to_embedding = np.vectorize(self.label_converter.class_label_to_embedding)
        elif config.data.label_converter == 'clip':
            # self.label_converter = CLIPLabelConverter(embedding_path=Path(os.path.join(self.data_dir, 'dance_technique_embeddings_clip.pkl')))
            # GPU is fine here because the DataLoader uses
            # multiprocessing_context="spawn" (see DataModule._mp_context),
            # so workers initialize their own CUDA contexts.
            text_cache_path = os.path.join(self.data_dir, 'label_clip_cache.pkl')
            self.label_converter = CLIPLabelConverter(text_cache_path=text_cache_path)
        elif config.data.label_converter == 't5':
            # T5-base text encoder. 768-dim. Long descriptions are not
            # truncated — the per-frame label array uses dtype=object on
            # this path (see load_single_data).
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
            self.lmdb_load = False
            self.h5_load = False
            
        if self.rep == 'motorica':
            self.load_skeleton()

        print(f"MOTORICA: Total Chunks loaded: {len(self._data_keys)}")

    def load_skeleton(self):
        skeleton_path = os.path.join(self.data_dir, 'skeleton')
        if not os.path.exists(skeleton_path):
            os.mkdir(skeleton_path)
            motion_path = os.path.join(self.data_dir, 'sliced_motion')
            motion_list = [id for id in os.listdir(motion_path) if id.endswith('_motion.npy')]
            for file in motion_list:
                id = file.split('_motion')[0]
                motion = np.load(os.path.join(motion_path, file), allow_pickle=True)[()]['motion']
                skeleton = motion['skeleton']
                self.skeleton_dict[id] = skeleton
                skeleton_path = os.path.join(skeleton_path, f'{id}_skeleton.pkl')
                save_pickle_data(skeleton, skeleton_path)
        else:
            skeleton_list = [id for id in os.listdir(skeleton_path) if id.endswith('.pkl')]
            for file in skeleton_list:
                id = file.split('_skeleton')[0]
                skeleton = load_pickle_data(os.path.join(skeleton_path, file))
                self.skeleton_dict[id] = skeleton

    def data_packing(self, data, id, preprocess=True):
        output = super().data_packing(data, id, preprocess)
        
        if 'label' in data:
            if 'attributes' not in data:
                if self.config.data.label_converter == 'one-hot':
                    output['attributes'] = self.convert_to_number(data['label'][:, 1])
                elif self.config.data.label_converter == 'gemini':
                    output['attributes'] = self.label_converter.class_label_to_embedding(data['label'])
                elif self.config.data.label_converter == 'clip' or self.config.data.label_converter == 't5':
                    output['attributes'], output['description'] = self.label_converter.class_label_to_embedding(data['label'], use_description=self.use_description)
                    if 'label_neutral' in data:
                        output['attributes_neutral'], output['description_neutral'] = self.label_converter.class_label_to_embedding(data['label_neutral'], use_description=self.use_description)
            else:
                output['attributes'] = data['attributes']
                if 'description' in data:
                    output['description'] = data['description']
                if 'label_neutral' in data:
                    output['attributes_neutral'] = data['attributes_neutral']
                    if 'description_neutral' in data:
                        output['description_neutral'] = data['description_neutral']
            output['label'] = data['label']
            if 'label_index' in data:
                output['label_index'] = data['label_index']
            if 'label_neutral' in data:
                output['label_neutral'] = data['label_neutral']
            self.is_att = True
        
        return output

    def load_single_data(self, id, h5_packing=False):
        output = super().load_single_data(id, h5_packing)
        label_dir = os.path.join(self.data_dir, 'sliced_labels')
        label_json = os.path.join(label_dir, f'{id}_label.json')
        label_npy = os.path.join(label_dir, f'{id}_label.npy')

        # T5 path needs full strings (no <U1000 truncation), so the per-frame
        # array is built on dtype=object. CLIP path keeps <U1000 — its
        # tokenizer truncates further to 77 tokens anyway.
        label_dtype = object if self.config.data.label_converter == 't5' else "<U1000"

        # Compact JSON: preferred path. Reconstruct (T, 3) string array and
        # label_index on the fly so downstream code keeps the same shape
        # contract as the legacy npy.
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
            label_file = os.path.join(label_dir, f'{id}_label.txt')
            if os.path.exists(label_file):
                label_dict = read_label_segment_txt(label_file)
                output['label'] = label_dict['genre'] + ": " + label_dict['label']
                output['length'] = label_dict['end'] - label_dict['start']
                output['attributes'] = self.label_converter.encode_text(output['label']).detach().cpu().numpy().astype(np.float32)
                output['description'] = self.label_converter.encode_text(label_dict['description']).detach().cpu().numpy().astype(np.float32)
            else:
                print(f'WARNING: {id} does not have label')
                return None

        return output

    def __getitem__(self, key):
        if self.lmdb_load:
            return self._load_lmdb_data(key)
        elif self.lazy_load:
            return super().__getitem__(key)
        elif self.h5_load:
            h5_data = {name: self.h5_file[key][name][()] for name in self.h5_file[key].keys() if name !='motion'}
            h5_data['motion'] = {name: self.h5_file[key]['motion'][name][()] for name in self.h5_file[key]['motion'].keys()}
            h5_data['label'] = h5_data['label'].astype(str)
            if 'skeleton_path' in h5_data:
                skeleton = load_pickle_data(h5_data['skeleton_path'])
                h5_data['skeleton'] = skeleton
            data = self.data_packing(h5_data, key, preprocess=True)

            if self.config.model.name == 'vae':
                if 'attributes' in data:
                    del data['attributes']

            return data
        else:
            return super().__getitem__(key)


if __name__ == '__main__':
    from src.utils.train.train_util import load_config
    config = load_config()  
    dataset = MotoricaDataset(config)
    data = dataset[0]
    test=1