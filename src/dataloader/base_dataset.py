from torch.utils.data import Dataset
import numpy as np
import os
import sys
from pathlib import Path
import h5py
from tqdm import tqdm
import copy
import pickle
import lmdb
import random
from .utility import Normalizer, save_pickle_data, load_pickle_data

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.skeleton.preprocessing import motion_preprocessing, motion_postprocessing
from src.skeleton.smpl_fk import SMPLModel
from src.skeleton.forward_kinematics import ForwardKinematics
from src.skeleton.utility import get_motorica_skeleton_names
from src.preprocessing.label_format import (
    label_chunk_record,
    num_description_candidates,
    read_compact_label,
)
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def _strip_mirror_suffix(chunk_id):
    """Mirrored chunks ('<base>_M') share audio/jukebox/beats with the
    original. Use this to rewrite ids when looking up direction-invariant
    artifacts."""
    if chunk_id.endswith('_M'):
        return chunk_id[:-2]
    return chunk_id

class BaseDataset():
    def __init__(self, config):
        self.config = config
        self.root_dir = config.data.dir
        self.data_dir = None
        self._data = {}
        self._data_keys = []
        self.motion_preprocessing = config.data.motion_preprocessing
        self.body_features = config.data.body_features
        self.fk = None
        self.use_neutralized_motion = config.data.use_neutralized_motion
        # self.exp_filter = []
        self.test_key = []
        self.ignore_list = []
        self.max_length = config.data.chunk_time * config.data.motion_fps
        # self.cache = {}
        # self.cache_size = config.data.cache_size

        self.is_att = False
        self.h5_load = False
        self.lmdb_load = False
        self.beat_based = False

        if self.config.data.data_extension == 'h5':
            self.h5_load = True
        elif self.config.data.data_extension == 'lmdb':
            self.lmdb_load = True

        # Lazy npy: keep only the key list in memory and load+preprocess each
        # chunk per __getitem__ call. Defaults on for npy mode unless the
        # config opts back into the eager path.
        self.lazy_load = (
            not self.h5_load and not self.lmdb_load
            and getattr(self.config.data, 'lazy_load', True)
        )
        
        # HDF5 file
        self.h5_file = None
        self.data_index = None
        
        # LMDB environment
        self.lmdb_env = None

        # Data type
        self.data_type = config.data.type
        self.rep = config.data.representation # motorica or smpl

        # if self.config.model.name=='energy' or self.config.model.name=='energy_diffusion' or 'flow' in self.config.model.name or self.config.model.name=='cross_energy_diffusion_joint' or self.config.model.name=='diffusion_joint':
        self.motion_only = False
        # else:
            # self.motion_only = config.model.VAE.motion_only

    def get_keys(self):
        return self._data_keys

    def motion_handler(self, motion_data, neutralized_motion_data, id, preprocess=True):
        if neutralized_motion_data is not None:
            use_neutralized_motion = True
        else:
            use_neutralized_motion = False
        output = {}
        skeleton = None
        if self.data_type == 'tree':
            if 'skeleton' in motion_data:
                skeleton = motion_data['skeleton']
                self.skeleton_dict[id] = skeleton
            if self.fk is None:
                if self.rep == 'motorica':
                    self.fk = ForwardKinematics(skeleton, selected_joints=get_motorica_skeleton_names())
                elif self.rep == 'smpl':
                    smpl_path = './data/smpl/SMPL_NEUTRAL.pkl'
                    if not os.path.exists(smpl_path):
                        raise FileNotFoundError(f"SMPL model file not found at {smpl_path}")
                    self.fk = SMPLModel(num_joints=self.config.data.num_joints, smpl_model_path=smpl_path)
        
            motion = motion_data['motion_data'].astype(np.float32)
            if use_neutralized_motion:
                neutralized_motion = neutralized_motion_data['motion_data'].astype(np.float32)
            else:
                neutralized_motion = None
        else:
            motion = motion_data.astype(np.float32)
            if use_neutralized_motion:
                neutralized_motion = neutralized_motion_data.astype(np.float32)
            else:
                neutralized_motion = None
        
        if preprocess:
            output['motion'], preprocessing = motion_preprocessing(motion,
                                                    data_type=self.data_type,
                                                    method=self.motion_preprocessing,
                                                    dataset=self.rep,
                                                    fk=self.fk,
                                                    skeleton=skeleton)
            output['preprocessing_R'] = preprocessing['rotation']
            output['preprocessing_t'] = preprocessing['translation']
            if use_neutralized_motion:
                output['neutralized_motion'], preprocessing = motion_preprocessing(neutralized_motion,
                                                        data_type=self.data_type,
                                                        method=self.motion_preprocessing,
                                                        dataset=self.rep,
                                                        fk=self.fk,
                                                        skeleton=skeleton)
        else:
            output['motion'] = motion.astype(np.float32)
            if use_neutralized_motion:
                output['neutralized_motion'] = neutralized_motion.astype(np.float32)

        if self.data_type == 'tree':
            input = copy.deepcopy(output['motion'])
            if use_neutralized_motion:
                neutralized_input = copy.deepcopy(output['neutralized_motion'])
            if preprocess:
                input = motion_postprocessing(input, type=self.forward_type, data_type=self.data_type)
                if use_neutralized_motion:
                    neutralized_input = motion_postprocessing(neutralized_input, type=self.forward_type, data_type=self.data_type)
            if 'motion_joint' in motion_data:
                gt_joints = motion_data['motion_joint']
                if use_neutralized_motion:
                    neutralized_gt_joints = neutralized_motion_data['motion_joint']
            elif self.rep == 'motorica':
                gt_joints = self.fk(input, fill_dummy=True, skeleton=skeleton)
                if use_neutralized_motion:
                    neutralized_gt_joints = self.fk(neutralized_input, fill_dummy=True, skeleton=skeleton)
            elif self.rep == 'smpl':
                # with torch.no_grad():
                smpl_result = self.fk(input.reshape(input.shape[0], -1))
                gt_joints = smpl_result.detach().cpu().numpy()
                if use_neutralized_motion:
                    neutralized_smpl_result = self.fk(neutralized_input.reshape(neutralized_input.shape[0], -1))
                    neutralized_gt_joints = neutralized_smpl_result.detach().cpu().numpy()
            else:
                raise ValueError(f"Invalid representation: {self.rep}")
            if isinstance(gt_joints, torch.Tensor):
                gt_joints = gt_joints.detach().cpu().numpy()
            output['motion_joint'] = gt_joints.astype(np.float32)

            if use_neutralized_motion:
                if isinstance(neutralized_gt_joints, torch.Tensor):
                    neutralized_gt_joints = neutralized_gt_joints.detach().cpu().numpy()
                output['neutralized_motion_joint'] = neutralized_gt_joints.astype(np.float32)
        if skeleton is not None:
            output['skeleton'] = skeleton

        return output

    def load_single_data(self, id, h5_packing=False):
        output = {}
        motion_file = os.path.join(self.data_dir, 'sliced_motion', f'{id}_motion.npy')
        if self.use_neutralized_motion:
            neutralized_motion_file = os.path.join(self.data_dir, 'sliced_motion_neutralized', f'{id}_motion.npy')
        else:
            neutralized_motion_file = None
        if self.rep.lower() == 'smpl':
            motion_file = os.path.join(self.data_dir, 'sliced_motion_smpl', f'{id}_motion.npy')
            if self.use_neutralized_motion:
                neutralized_motion_file = os.path.join(self.data_dir, 'sliced_motion_neutralized_smpl', f'{id}_motion.npy')
        # Mirrored chunks share the original audio: strip the _M suffix when
        # resolving direction-invariant artifacts.
        audio_id = _strip_mirror_suffix(id)
        audio_file = os.path.join(self.data_dir, 'sliced_audio_features', f'{audio_id}_audio.npy')

        # Load motion
        motion = np.load(motion_file, allow_pickle=True)[()]
        if neutralized_motion_file is not None:
            neutralized_motion = np.load(neutralized_motion_file, allow_pickle=True)[()]
        else:
            neutralized_motion = None
        if self.beat_based:
            output['current_motion_fps'] = motion['current_fps']
            output['tgt_motion_fps'] = motion['target_fps']
        else:
            output['current_motion_fps'] = self.motion_fps
            output['tgt_motion_fps'] = self.motion_fps
        output['motion'] = motion['motion']
        if neutralized_motion is not None:
            output['neutralized_motion'] = neutralized_motion['motion']
        else:
            output['neutralized_motion'] = None

        if isinstance(motion, np.ndarray):
            motion_length = motion.shape[0]
        elif isinstance(motion, dict):
            motion_length = output['motion']['motion_data'].shape[0]
        else:
            raise ValueError(f'Unknown motion type: {type(motion)}')
        
        # Load audio
        if not self.motion_only or self.config.loss.lambda_disentangle > 0 or h5_packing:
            audio = np.load(audio_file, allow_pickle=True)[()]
            if self.beat_based:
                output['current_audio_fps'] = audio['current_fps']
                output['tgt_audio_fps'] = audio['target_fps']
            else:
                output['current_audio_fps'] = self.audio_fps
                output['tgt_audio_fps'] = self.audio_fps
            output['audio'] = audio['audio'].astype(np.float32)
            if output['audio'].shape[0] != int(motion_length * self.ratio):
                print(f'WARNING: {id} audio and motion have different lengths')
                return None

        
        return output

    def create_key_list(self, key_path):
        ref_path = os.path.join(self.data_dir, 'sliced_motion')
        if self.rep.lower() == 'smpl':
            ref_path = os.path.join(self.data_dir, 'sliced_motion_smpl')
        ref_list = [id for id in os.listdir(ref_path) if id.endswith('.npy')]

        with open(key_path, 'w') as f:
            for ref in ref_list:
                id = ref.split('_motion')[0]
                f.write(f"{id}\n")
        print(f"INFO: Created key list at {key_path}")

    def create_h5_dataset(self, h5_path):
        print(f"INFO: Creating H5 dataset at {h5_path}")
        if self.rep.lower() == 'smpl':
            motion_path = os.path.join(self.data_dir, 'sliced_motion_smpl')
        else:
            motion_path = os.path.join(self.data_dir, 'sliced_motion')
        skeleton_root = os.path.join(self.data_dir, 'skeleton')

        with h5py.File(h5_path, 'w') as f:
            motion_list = [id for id in os.listdir(motion_path) if id.endswith('.npy')]

            for file in tqdm(motion_list):
                id = file.split('_motion')[0]
                id_group = f.create_group(id)

                single_data = self.load_single_data(id, h5_packing=True)
                data = self.data_packing(single_data, id, preprocess=False)

                motion_group = id_group.create_group('motion')
                # if self.rep == 'motorica':
                motion_group.create_dataset('motion_data', data=data['motion'])
                motion_group.create_dataset('scale', data=single_data['motion']['scale'])

                # id_group.create_dataset('motion', data=data['motion'])
                id_group.create_dataset('audio', data=data['audio'])
                if self.beat_based:
                    id_group.create_dataset('current_motion_fps', data=data['current_motion_fps'])
                    id_group.create_dataset('tgt_motion_fps', data=data['tgt_motion_fps'])
                    id_group.create_dataset('current_audio_fps', data=data['current_audio_fps'])
                    id_group.create_dataset('tgt_audio_fps', data=data['tgt_audio_fps'])
                else:
                    id_group.create_dataset('current_motion_fps', data=self.motion_fps)
                    id_group.create_dataset('tgt_motion_fps', data=self.motion_fps)
                    id_group.create_dataset('current_audio_fps', data=self.audio_fps)
                    id_group.create_dataset('tgt_audio_fps', data=self.audio_fps)
                
                if 'skeleton' in data:
                    if not os.path.exists(skeleton_root):
                        os.mkdir(skeleton_root)
                    skeleton_path = os.path.join(skeleton_root, f'{id}_skeleton.pkl')
                    if not os.path.exists(skeleton_path):
                        save_pickle_data(data['skeleton'], skeleton_path)
                    id_group.create_dataset('skeleton_path', data=skeleton_path)
                if 'label' in data:
                    id_group.create_dataset('label', data=data['label'], dtype=h5py.string_dtype(encoding='utf-8'), compression='gzip')
                    id_group.create_dataset('label_index', data=data['label_index'])

    @rank_zero_only
    def create_lmdb_dataset(self, lmdb_path, map_size=1e12, commit_every=512):
        """
        Create LMDB dataset from NPY files similar to create_h5_dataset.

        Args:
            lmdb_path (str): Path where LMDB database will be created
            map_size (int): Maximum size of LMDB database in bytes (default: 1TB)
            commit_every (int): Flush the LMDB transaction every N samples.
                LMDB holds dirty pages in process memory until commit, so a
                single transaction across the whole dataset can need 50–100+
                GB of RAM. Periodic commits cap peak RAM at roughly
                ``commit_every * per_sample_size`` (a few hundred MB).
        """
        print(f"INFO: Creating LMDB dataset at {lmdb_path}")

        # Determine motion path based on representation
        if self.rep.lower() == 'smpl':
            motion_path = os.path.join(self.data_dir, 'sliced_motion_smpl')
        else:
            motion_path = os.path.join(self.data_dir, 'sliced_motion')

        skeleton_root = os.path.join(self.data_dir, 'skeleton')

        # Get list of motion files
        motion_list = [id for id in os.listdir(motion_path) if id.endswith('.npy')]

        # Create LMDB environment
        env = lmdb.open(lmdb_path, map_size=int(map_size))

        txn = env.begin(write=True)
        written = 0
        try:
            # Store metadata once at the head of the first transaction.
            metadata = {
                'total_samples': len(motion_list),
                'representation': self.rep,
                'data_type': self.data_type,
                'beat_based': self.beat_based,
                'motion_fps': getattr(self, 'motion_fps', None),
                'audio_fps': getattr(self, 'audio_fps', None)
            }
            txn.put('__metadata__'.encode(), pickle.dumps(metadata))

            for file in tqdm(motion_list, desc="Converting to LMDB"):
                id = file.split('_motion')[0]

                try:
                    # Load single data
                    single_data = self.load_single_data(id, h5_packing=True)
                    if single_data is None:
                        print(f"WARNING: Skipping {id} - failed to load data")
                        continue

                    # Pack data
                    data = self.data_packing(single_data, id, preprocess=True)

                    # Prepare LMDB entry (store large arrays as float16 to reduce size)
                    lmdb_entry = {
                        'motion': {
                            'motion_data': data['motion'].astype(np.float16),
                            'motion_joint': data['motion_joint'].astype(np.float16),
                            'scale': single_data['motion']['scale'] if 'scale' in single_data['motion'] else 1,
                            'preprocessing_R': data['preprocessing_R'],
                            'preprocessing_t': data['preprocessing_t']
                        },
                    }
                    if 'audio' in data:
                        lmdb_entry['audio'] = data['audio'].astype(np.float16)
                    if self.use_neutralized_motion:
                        lmdb_entry['neutralized_motion']= {
                           'motion_data': data['neutralized_motion'],
                           'motion_joint': data['neutralized_motion_joint'],
                        }
                        lmdb_entry['label_neutral'] = data['label_neutral']
                    # Add FPS information
                    if self.beat_based:
                        lmdb_entry.update({
                            'current_motion_fps': data['current_motion_fps'],
                            'tgt_motion_fps': data['tgt_motion_fps'],
                            'current_audio_fps': data['current_audio_fps'],
                            'tgt_audio_fps': data['tgt_audio_fps']
                        })
                    else:
                        lmdb_entry.update({
                            'current_motion_fps': self.motion_fps,
                            'tgt_motion_fps': self.motion_fps,
                            'current_audio_fps': self.audio_fps,
                            'tgt_audio_fps': self.audio_fps
                        })

                    # Handle skeleton data
                    if 'skeleton' in data:
                        if not os.path.exists(skeleton_root):
                            os.makedirs(skeleton_root)
                        skeleton_path = os.path.join(skeleton_root, f'{id}_skeleton.pkl')
                        if not os.path.exists(skeleton_path):
                            save_pickle_data(data['skeleton'], skeleton_path)
                        lmdb_entry['skeleton_path'] = skeleton_path

                    # Handle label data
                    if 'label' in data:
                        lmdb_entry['label'] = data['label']
                    if 'label_index' in data:
                        lmdb_entry['label_index'] = data['label_index']
                    if 'attributes' in data:
                        lmdb_entry['attributes'] = data['attributes'].astype(np.float16) if isinstance(data['attributes'], np.ndarray) else data['attributes']
                    if 'description' in data:
                        lmdb_entry['description'] = data['description'].astype(np.float16) if isinstance(data['description'], np.ndarray) else data['description']

                    # Serialize and store
                    serialized_data = pickle.dumps(lmdb_entry)
                    txn.put(id.encode('utf-8'), serialized_data)
                    written += 1

                    # Flush dirty pages periodically so the in-memory dirty
                    # set doesn't grow unbounded.
                    if written % commit_every == 0:
                        txn.commit()
                        txn = env.begin(write=True)

                except Exception as e:
                    print(f"ERROR: Failed to process {id}: {str(e)}")
                    continue

            # Final flush for the remainder.
            txn.commit()
            txn = None
            print(f"INFO: Successfully created LMDB dataset with {len(motion_list)} samples")

        except Exception as e:
            print(f"ERROR: Failed to create LMDB dataset: {str(e)}")
            if txn is not None:
                txn.abort()
            raise
        finally:
            env.close()
   
    def create_key_list_from_lmdb(self, lmdb_path, key_path):
        """
        Create key list from existing LMDB dataset.
        
        Args:
            lmdb_path (str): Path to LMDB database
            key_path (str): Path where key list will be saved
        """
        env = lmdb.open(lmdb_path, readonly=True)
        keys = []
        
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    key_str = key.decode('utf-8')
                    if key_str != '__metadata__':  # Skip metadata entry
                        even_id = int(key_str.split('_chunk')[1])
                        # if even_id % 2 != 0:
                        #     continue
                        keys.append(key_str)
            
            with open(key_path, 'w') as f:
                for key in sorted(keys):
                    f.write(f"{key}\n")
            
            print(f"INFO: Created key list at {key_path} with {len(keys)} entries")
            
        finally:
            env.close()
            
    def load_motion_only(self, load_key='motion'):
        output = {}
        motion_path = os.path.join(self.data_dir, 'sliced_motion')
        if self.use_neutralized_motion:
            neutralized_motion_path = os.path.join(self.data_dir, 'sliced_motion_neutralized')
        else:
            neutralized_motion_path = None
        if self.rep.lower() == 'smpl':
            motion_path = os.path.join(self.data_dir, 'sliced_motion_smpl')
            if self.use_neutralized_motion:
                neutralized_motion_path = os.path.join(self.data_dir, 'sliced_motion_neutralized_smpl')
            else:
                neutralized_motion_path = None
        motion_list = [id for id in os.listdir(motion_path) if id.endswith('.npy')]

        for file in motion_list:
            id = file.split('_motion')[0]
            motion = np.load(os.path.join(motion_path, file), allow_pickle=True)[()]['motion']
            if neutralized_motion_path is not None:
                if self.name == 'HumanML3D':
                    neutralized_motion_data = motion.copy()
                else:
                    neutralized_motion_data = np.load(os.path.join(neutralized_motion_path, file), allow_pickle=True)[()]['motion']
            else:
                neutralized_motion_data = None
            motion = self.motion_handler(motion, neutralized_motion_data=neutralized_motion_data, id=id, preprocess=True)[load_key]
            output[id] = motion
        return output
    
    def _load_lmdb_data(self, key):
        """
        Load data from LMDB database. Only SMPL format is supported
        """
        with self.lmdb_env.begin() as txn:
            lmdb_data = pickle.loads(txn.get(key.encode('utf-8')))

        # Upcast float16 arrays back to float32 (stored as fp16 to reduce LMDB size)
        for k, v in lmdb_data.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float16:
                lmdb_data[k] = v.astype(np.float32)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, np.ndarray) and v2.dtype == np.float16:
                        v[k2] = v2.astype(np.float32)

        if 'label' in lmdb_data:
            label = lmdb_data['label']
            if isinstance(label, np.ndarray):
                # Handle bytes dtype
                if label.dtype.kind == 'S':  # bytes string
                    lmdb_data['label'] = np.vectorize(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))(label)
                else:
                    # Convert to object dtype to ensure proper string access
                    lmdb_data['label'] = np.array([[str(item) for item in row] for row in label], dtype=object)
            else:
                lmdb_data['label'] = np.array(lmdb_data['label'], dtype=str)
        
        if self.motion_preprocessing != 'face_forward':
            raise ValueError(f"Only face_forward is supported for Motorica dataset")
        data = self.data_packing(lmdb_data, key, preprocess=False) # Already preprocessed
        
        return data


    def __getitem__(self, key):
        if self.lazy_load:
            single = self.load_single_data(key)
            if single is None:
                raise KeyError(f"No data for key {key!r}")
            return self.data_packing(single, key, preprocess=True)
        if not self.h5_load:
            return self._data[key]
        # For H5 file, we need to override this method at each child class
        
    def __len__(self):
        return len(self._data_keys)
    
    def set_h5_files(self):
        h5_path = os.path.join(self.data_dir, 'h5_dataset.h5')
        self.h5_file = h5py.File(h5_path, 'r')
    
    # def set_lmdb_env(self):
    #     """Initialize LMDB environment for reading"""
    #     lmdb_path = os.path.join(self.data_dir, 'lmdb_dataset')
    #     self.lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
    def set_lmdb_env(self):
        """Initialize LMDB environment with optimizations"""
        lmdb_path = os.path.join(self.data_dir, 'lmdb_dataset')
        self.lmdb_env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,  # Disable: 90GB LMDB barely fits in page cache,
                              # readahead wastes I/O by prefetching pages that get evicted
            meminit=False,
            max_readers=128,
            map_size=1e12,
        )

    def data_packing(self, data, id, preprocess=True):
        if 'neutralized_motion' not in data:
            data['neutralized_motion'] = None
        output = self.motion_handler(data['motion'], data['neutralized_motion'], id=id, preprocess=preprocess)
        if 'audio' in data:
            output['audio'] = data['audio']
        output['current_motion_fps'] = data['current_motion_fps']
        output['tgt_motion_fps'] = data['tgt_motion_fps']
        if not self.motion_only or self.config.loss.lambda_disentangle > 0:
            output['current_audio_fps'] = data['current_audio_fps']
            output['tgt_audio_fps'] = data['tgt_audio_fps']
            
        if 'length' in data:
            output['length'] = data['length']

        return output

    def list_data_keys(self, sample_id=None):
        """Populate ``self._data_keys`` from the sliced motion directory
        without loading any data. Used by lazy npy mode."""
        motion_path = os.path.join(self.data_dir, 'sliced_motion')
        if self.rep.lower() == 'smpl':
            motion_path = os.path.join(self.data_dir, 'sliced_motion_smpl')
        motion_list = [f for f in os.listdir(motion_path) if f.endswith('.npy')]
        keys = []
        for file in motion_list:
            chunk_id = file.split('_motion')[0]
            if sample_id is not None and chunk_id not in sample_id:
                continue
            if chunk_id in self.ignore_list:
                continue
            keys.append(chunk_id)
        self._data_keys = keys
        return keys

    def load_data(self, sample_id=None):
        motion_path = os.path.join(self.data_dir, 'sliced_motion')
        if self.rep.lower() == 'smpl':
            motion_path = os.path.join(self.data_dir, 'sliced_motion_smpl')
        motion_list = [id for id in os.listdir(motion_path) if id.endswith('.npy')]

        counter = 1
        max_length = len(motion_list) * 0.2
        # max_length = 128
        for file in tqdm(motion_list):
            id = file.split('_motion')[0]
            # # if self.config.debug:
            # even_id = int(id.split('_chunk')[1])
            # if even_id % 2 != 0:
            #     continue
            if sample_id is not None and not id in sample_id:
                continue
            if id in self.ignore_list:
                continue
            single_data = self.load_single_data(id)
            if single_data is None:
                continue
            self._data[id] = self.data_packing(single_data, id, preprocess=True)
            
            if self.config.debug and sample_id is None:
                counter += 1
                if counter > max_length:
                    break


class Fusion(Dataset):
    def __init__(self, config, datasets: list[BaseDataset], mode='train', sample_id=None):
        self.name = config.data.name
        self.rep = config.data.representation
        self.mode = mode
        self.config = config
        self.root_dir = config.data.dir
        self.normalize_data = config.data.normalize_data
        self.datasets = {}
        self.skeleton_dict = {}
        self.normalizer = None
        self.normalizer_latent = None
        self.absolute = config.data.absolute
        self.use_neutralized_motion = config.data.use_neutralized_motion
        self.max_length = config.data.chunk_time * config.data.motion_fps
        for dataset in datasets:
            instance = dataset(config, id=sample_id)
            self.datasets[instance.name] = instance
        
        self.h5_load = False
        self.lmdb_load = False

        if self.config.data.data_extension == 'h5':
            self.h5_load = True
        elif self.config.data.data_extension == 'lmdb':
            self.lmdb_load = True

        self.val_split = config.data.val_split
        self.pairs = []
        self.train_pairs = []
        self.val_pairs = []
        self.test_pairs = []

        self.read_dataset_index()

        if self.normalize_data:
            self.normalizer_model_list = ['cross_latent_diffusion']
            self.set_normalizer(mode)

        if not self.rep == 'smpl':
            # Merge skeleton dicts
            for dataset_name, dataset in self.datasets.items():
                self.skeleton_dict.update(dataset.skeleton_dict)

        max_length = len(self.dataset_index) * 0.2
        # max_length = 128
        counter = 1
        dataset_dict_pairs = {}
        for dataset in self.datasets.keys():
            dataset_dict_pairs[dataset] = []
        for key, dataset_name in self.dataset_index.items():
            if sample_id is not None and key in sample_id:
                self.pairs.append(key)
            else:
                if self.select_key_id(key, dataset_name) in self.datasets[dataset_name].test_key:
                    self.test_pairs.append(key)
                else:
                    # self.train_pairs.append(key)
                    dataset_dict_pairs[dataset_name].append(key)

            if self.config.debug and self.config.data.data_extension == 'lmdb':
                counter += 1
                if counter > max_length:
                    break

        # Create validation set
        for dataset_name, pairs in dataset_dict_pairs.items():
            val_size = int(len(pairs) * self.config.data.val_split)
            random.shuffle(pairs)
            self.val_pairs.extend(pairs[:val_size])
            self.train_pairs.extend(pairs[val_size:])

        self.train_pairs = self._dataset_grouped_shuffle(self.train_pairs)
        random.shuffle(self.val_pairs)
        if mode == 'train':
            self.pairs = self.train_pairs
        elif mode == 'val':
            self.pairs = self.val_pairs
        elif mode == 'test':
            self.pairs = self.test_pairs 
        elif mode == 'sample':
            pass # Already loaded
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _dataset_grouped_shuffle(self, pairs):
        """Shuffle within each dataset, then interleave in large chunks.

        This keeps consecutive samples from the same LMDB file together,
        preserving readahead locality while still mixing datasets across
        the epoch. Each chunk contains samples from one dataset only.
        """
        # Group pairs by dataset
        groups = {}
        for key in pairs:
            ds = self.dataset_index[key]
            groups.setdefault(ds, []).append(key)

        # Shuffle within each dataset group
        for ds in groups:
            random.shuffle(groups[ds])

        # Split each group into chunks, then shuffle the chunks
        # Split each group into chunks, then shuffle the chunks
        chunk_size = self.config.data.batch_size * 4  # 4 batches per chunk
        chunks = []
        for ds, keys in groups.items():
            for i in range(0, len(keys), chunk_size):
                chunks.append(keys[i:i + chunk_size])

        random.shuffle(chunks)

        # Flatten back
        result = []
        for chunk in chunks:
            result.extend(chunk)
        return result

    def reshuffle_train(self):
        """Re-shuffle training pairs with dataset grouping. Call each epoch."""
        self.train_pairs = self._dataset_grouped_shuffle(self.train_pairs)
        if hasattr(self, 'pairs') and len(self.pairs) == len(self.train_pairs):
            self.pairs = self.train_pairs

    def switch_mode(self, mode='train'):
        if mode == 'train':
            self.pairs = self.train_pairs
        elif mode == 'val':
            self.pairs = self.val_pairs
        elif mode == 'test':
            self.pairs = self.test_pairs
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_normalizer_path(self, name='smpl'):
        ref_dataset = self.config.data.dataset
        if hasattr(self.config.data, 'trained_dataset'):
            ref_dataset = self.config.data.trained_dataset

        normalizer_name = 'normalizer'
        if name == 'smpl':
            normalizer_name = normalizer_name + '_smpl'
        elif name == 'latent':
            normalizer_name = normalizer_name + '_latent'
        elif name == 'joint':
            normalizer_name = normalizer_name + '_joint'

        if len(ref_dataset) == 1:
            normalizer_name = normalizer_name + '_' + ref_dataset[0].lower()
        elif len(ref_dataset) == 2:
            if 'motorica' in ref_dataset and 'AIST' in ref_dataset:
                normalizer_name = normalizer_name + '_dance_all'
            elif 'motorica'in ref_dataset and 'HumanML3D' in ref_dataset:
                normalizer_name = normalizer_name + '_motorica_humanml3d'
        elif len(ref_dataset) == 3:
            normalizer_name = normalizer_name + '_all'

        normalizer_path = os.path.join(self.root_dir, normalizer_name + '.npy')

        return normalizer_path

    def set_normalizer(self, mode='train'):
        if mode == 'train':
            if self.config.model.name in self.normalizer_model_list:
                normalizer_path = self.get_normalizer_path(name='latent')
                if not os.path.exists(normalizer_path):
                    print(f"Latent normalizer not found at {normalizer_path}. Please run the script to generate the normalizer.")
                    return
                self.normalizer_latent = Normalizer(path=normalizer_path)
                print(f"Latent Normalizer found at {normalizer_path}")

            normalizer_name = 'joint'
            if self.config.data.representation == 'smpl':
                normalizer_name = 'smpl'
            normalizer_path = self.get_normalizer_path(name=normalizer_name)
            if os.path.exists(normalizer_path):
                self.normalizer = Normalizer(path=normalizer_path)
                print(f"Normalizer found at {normalizer_path}")
                return 
            
            print(f"Normalizer not found at {normalizer_path}. Generating normalizer...")
            all_motion = []
            load_key = 'motion'
            if self.absolute:
                load_key = 'motion_joint'
            if self.lmdb_load:
                self.set_lmdb_files()
            for dataset_name, dataset in self.datasets.items():
                if self.h5_load:
                    motion = dataset.load_motion_only(load_key=load_key)
                    all_motion.extend(np.concatenate([motion[key] for key in motion.keys()], axis=0))
                elif self.lmdb_load:
                    motion = []
                    for key in dataset._data_keys:
                        motion.append(dataset._load_lmdb_data(key)[load_key])
                    all_motion.extend(np.stack(motion, axis=0))
                else:
                    all_motion.extend(np.concatenate([dataset._data[key][load_key] for key in dataset._data.keys()], axis=0))

            all_motion = np.stack(all_motion, axis=0)
            if self.absolute:
                all_motion = all_motion.reshape(all_motion.shape[0], -1)
            self.normalizer = Normalizer(data=all_motion)
            self.normalizer.save_normalizer(normalizer_path)
        else:
            if self.config.model.name in self.normalizer_model_list:
                normalizer_path = self.get_normalizer_path(name='latent')
                if not os.path.exists(normalizer_path):
                    print(f"Latent normalizer not found at {normalizer_path}. Please run the script to generate the normalizer.")
                    return
                self.normalizer_latent = Normalizer(path=normalizer_path)
                print(f"Latent Normalizer found at {normalizer_path}")

            if self.config.data.representation == 'smpl':
                normalizer_name = 'smpl'
            normalizer_path = self.get_normalizer_path(name=normalizer_name)
            if os.path.exists(normalizer_path):
                self.normalizer = Normalizer(path=normalizer_path)
                print(f"Normalizer found at {normalizer_path}")
                return 
            else:
                raise FileNotFoundError(f"Normalizer not found at {normalizer_path}")
                
    def read_dataset_index(self):
        self.dataset_index = {}
        for dataset_name, dataset in self.datasets.items():
            keys = dataset.get_keys()
            for key in keys:
                self.dataset_index[key] = dataset_name


    def set_h5_files(self):
        for name, dataset in self.datasets.items():
            dataset.set_h5_files()
    
    def set_lmdb_files(self):
        for name, dataset in self.datasets.items():
            dataset.set_lmdb_env()
    
    @staticmethod
    def select_key_id(key, name):
        if name.lower() == 'aist':
            return key.split('_')[4]
        elif name.lower() == 'motorica':
            return key.split('_chunk')[0]
        elif name.lower() == 'humanml3d':
            return key.split('_chunk')[0]
        else:
            raise ValueError(f"Invalid dataset name: {name}")
            
    def gen_mask(self, length):
        return np.arange(self.max_length) < length

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        key = self.pairs[index]
        dataset_name = self.dataset_index[key]

        # if self.h5_load and key in self.cache:
        #     return self.cache[key]
        
        data = self.datasets[dataset_name][key]
        output = {
            'motion': data['motion'],
            'key': key,
            'current_motion_fps': data['current_motion_fps'],
            'target_motion_fps': data['tgt_motion_fps'],
            'dataset': dataset_name,
        }

        if self.use_neutralized_motion:
            output['neutralized_motion'] = data['neutralized_motion']
            output['neutralized_motion_joint'] = data['neutralized_motion_joint']
            output['attributes_neutral'] = data['attributes_neutral']
            output['description_neutral'] = data['description_neutral']

        if 'label_index' in data:
            output['label_index'] = data['label_index']

        if 'audio' in data:
            output['audio'] = data['audio']
            output['audio_mask'] = True
        else:
            output['audio'] = np.zeros((data['motion'].shape[0], self.config.data.audio_features), dtype=np.float32)
            output['audio_mask'] = False

        if 'current_audio_fps' in data:
            output['current_audio_fps'] = data['current_audio_fps']

        if 'attributes' in data and data['attributes'] is not None:
            output['attributes'] = data['attributes']

        if 'description' in data:
            if data['description'] is not None:
                output['description'] = data['description']
                output['att_mask'] = True
            else:
                # Use the per-dataset label_converter's embedding_dim so this
                # works for CLIP (512) and T5 (768) without a config tweak.
                ds = self.datasets[dataset_name]
                emb_dim = getattr(getattr(ds, 'label_converter', None), 'embedding_dim', 512)
                output['description'] = np.zeros((data['motion'].shape[0], emb_dim), dtype=np.float32)
                output['att_mask'] = False

        if 'motion_joint' in data:
            output['motion_joint'] = data['motion_joint']

        if 'dataset' in data:
            output['dataset'] = data['dataset']

        if 'pos_one_hots' in data:
            output['pos_one_hots'] = data['pos_one_hots']
        if 'sent_len' in data:
            output['sent_len'] = data['sent_len']
        if 'text' in data:
            output['text'] = data['text']

        if 'word_embeddings' in data:
            output['word_embeddings'] = data['word_embeddings']

        if 'length' in data:
            output['length'] = data['length']
            output['mask'] = self.gen_mask(data['length'])
            if data['length'] < self.max_length:
                output['motion'] = np.concatenate([output['motion'], np.zeros((self.max_length - data['length'], output['motion'].shape[1]), dtype=np.float32)], axis=0)
                if 'motion_joint' in data:
                    output['motion_joint'] = np.concatenate([output['motion_joint'], np.zeros((self.max_length - data['length'], output['motion_joint'].shape[1], output['motion_joint'].shape[2]), dtype=np.float32)], axis=0)
                if 'audio' in data:
                    output['audio'] = np.concatenate([output['audio'], np.zeros((self.max_length - data['length'], output['audio'].shape[1]), dtype=np.float32)], axis=0)
        # if self.normalize_data and self.config.model.name not in self.normalizer_model_list:
        #     output['motion'] = self.normalizer.normalize(output['motion'])

        # if self.h5_load and len(self.cache) < self.cache_size:
        #     self.cache[key] = output
            
        return output

