import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from copy import deepcopy
import torch

from src.utils.inference.inference_util import load_config
from src.models.pl_module.cross_energy_edit_dance_joint import CrossEnergyEditDanceJoint
from src.skeleton.preprocessing import motion_postprocessing, motion_preprocessing
from src.skeleton.forward_kinematics import ForwardKinematics
from src.skeleton.smpl_fk import SMPLModel
from src.skeleton.utility import get_motorica_skeleton_names
from src.visualization.skeleton import create_video_from_keypoints
from src.utils.utility import CLIPLabelConverter, T5LabelConverter, read_label_segment_txt
from src.dataloader.utility import Normalizer
from src.preprocessing.utils import process_smpl_data
import re
from src.dataloader.motorica_dataset import MOTORICA_TEST_KEY
from long_range_generation import stitch

class SegDataset():
    def __init__(self, config, args, data_from='test', sample_id=None, device='cpu'):
        self.config = config
        self.args = args
        self.model = config.model.name
        self.LABEL = True
        self.root_dir = './data/'
        self.device = device
        self.normalizer = None
        self.normalizer_latent = None

        self.skeleton_dict = None
        self.set_normalizer()

        self.root_path = os.path.join('./data', args.dataset)
        self.feature_path = list((Path(self.root_path) / 'sliced_audio_features').glob('*.npy'))
        
        def extract_chunk_number(path):
            """Extract the number after 'chunk' from the filename"""
            match = re.search(r'chunk(\d+)', path.stem)
            if match:
                return int(match.group(1))
            return float('inf')  # Put files without chunk number at the end
        
        self.feature_path = sorted(self.feature_path, key=extract_chunk_number)
        # Pick the text encoder declared in the model config so this matches
        # what the checkpoint was trained with. Both converters expose the
        # same encode_text(text)->Tensor API used below, so the call sites
        # in load_data don't need to change.
        label_converter_name = getattr(config.data, 'label_converter', 'clip')
        cache_dir = os.path.join('./data', getattr(config.data, 'dataset', ['motorica'])[0])
        if label_converter_name == 't5':
            text_cache_path = os.path.join(cache_dir, 'label_t5_cache.pkl')
            self.label_converter = T5LabelConverter(text_cache_path=text_cache_path)
        else:
            text_cache_path = os.path.join(cache_dir, 'label_clip_cache.pkl')
            self.label_converter = CLIPLabelConverter(
                embedding_path=Path(os.path.join('./data/motorica/dance_technique_embeddings_clip.pkl')),
                text_cache_path=text_cache_path,
            )
        self.datas = self.load_data(data_from, sample_id)
        self.keys = list(self.datas.keys())

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
    
    @staticmethod
    def select_key_id(key):
        return key.split('_chunk')[0]

    def set_normalizer(self):
        if self.model=='cross_latent_diffusion':
            normalizer_path = self.get_normalizer_path(name='latent')
            if not os.path.exists(normalizer_path):
                print(f"Latent normalizer not found at {normalizer_path}. Please run the script to generate the normalizer.")
                return
            self.normalizer_latent = Normalizer(path=normalizer_path)
            print(f"Latent normalizer found at {normalizer_path}")

        if self.config.data.representation == 'smpl':
            normalizer_name = 'smpl'
        normalizer_path = self.get_normalizer_path(name=normalizer_name)
        if os.path.exists(normalizer_path):
            self.normalizer = Normalizer(path=normalizer_path)
            print(f"Normalizer found at {normalizer_path}")
            return 
        else:
            raise FileNotFoundError(f"Normalizer not found at {normalizer_path}")

    def load_data(self, data_from='test', sample_id=None):
        datas = {}
        if self.config.data.representation == 'smpl':
            fk = SMPLModel()
        else:
            raise ValueError(f"Not supported")

        for feature_path in self.feature_path:
            id = feature_path.stem.split('_audio')[0]
            if sample_id is not None and not id==sample_id:
                continue
            elif data_from == 'test':
                if not self.select_key_id(id) in MOTORICA_TEST_KEY:
                    continue
            else:
                if self.select_key_id(id) in MOTORICA_TEST_KEY:
                    continue

            feature = np.load(feature_path, allow_pickle=True)[()]

            data = {
                'key': id,
                'audio': feature['audio'], # (B, T, 1024)
                'audio_mask': True,
                'att_mask': True,
                'current_audio_fps': feature['current_fps'],
                'tgt_audio_fps': feature['target_fps']
            }

            if self.LABEL:
                label_file = os.path.join(self.root_path, 'sliced_labels', f'{id}_label.txt')
                if os.path.exists(label_file):
                    label_dict = read_label_segment_txt(label_file)
                    data['label'] = label_dict['genre'] + ": " + label_dict['label']
                    data['text'] = label_dict['description']
                    data['length'] = label_dict['end'] - label_dict['start']
                    data['attributes'] = self.label_converter.encode_text(data['label']).detach().cpu().numpy().astype(np.float32)
                    data['description'] = self.label_converter.encode_text(label_dict['description']).detach().cpu().numpy().astype(np.float32)
                else:
                    print(f'WARNING: {id} does not have label')
                    return None
            
            motion_path = os.path.join(self.root_path, 'sliced_motion_smpl', f'{id}_motion.npy')
            motion = np.load(motion_path, allow_pickle=True)[()]
            if 'smpl_trans' in motion:
                motion = process_smpl_data(self.root_path/'sliced_motion_smpl', id, fk, data=motion)
            data['current_motion_fps'] = motion['current_fps']
            data['tgt_motion_fps'] = motion['target_fps']
            data['motion'], _ = motion_preprocessing(motion['motion']['motion_data'],
                                                            data_type=self.config.data.type,
                                                            method=self.config.data.motion_preprocessing,
                                                            dataset=self.config.data.representation,
                                                            fk=fk,
                                                            skeleton=None)
            data['motion_positions'] = fk(data['motion'])
            for k, v in data.items():
                if k == 'label':
                    continue
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v)
            datas[id] = data
        return datas

    def __getitem__(self, index):
        output = {}
        id = self.keys[index]
        data = self.datas[id]
        output['key'] = id
        length = data['length']
        output['audio'] = data['audio'].unsqueeze(0).to(self.device)
        output['attributes'] = data['attributes'].unsqueeze(0).repeat(1, length, 1).to(self.device)
        output['description'] = data['description'].unsqueeze(0).repeat(1, length, 1).to(self.device)
        if 'label_index' in data:
            output['label_index'] = data['label_index'].unsqueeze(0).to(self.device)
        if 'motion' in data:
            output['gt_motion'] = data['motion'].unsqueeze(0).to(self.device)
            output['gt_motion_joint'] = data['motion_positions'].unsqueeze(0).to(self.device)
        output['audio_mask'] = torch.tensor([True] * output['audio'].shape[0], dtype=torch.bool).to(self.device)
        output['att_mask'] = torch.tensor([True] * output['attributes'].shape[0], dtype=torch.bool).to(self.device)
        output['text'] = data['text']
        return output

    def __len__(self):
        return len(self.datas)

    def get_dataset_info(self):
        return {
            'normalizer': self.normalizer,
            'normalizer_latent': self.normalizer_latent,
            'skeleton_dict': self.skeleton_dict
        }




def chunked_forward(model, data, max_frames, device):
    """
    Slice long segments into overlapping chunks and generate sequentially with
    prefix injection for smooth transitions (dynamic CFG masking + latent blending).
    """
    stride = max_frames // 2  # 75 frames for 150-frame windows
    fk = SMPLModel()

    audio = data['audio']  # (1, T, audio_dim)
    T = audio.shape[1]

    # Compute chunk start indices with stride overlap
    starts = list(range(0, T - max_frames + 1, stride))
    # Ensure last chunk covers the end of the sequence
    if not starts or starts[-1] + max_frames < T:
        starts.append(T - max_frames)

    # Attributes and description are single CLIP embeddings repeated per frame
    att_embed = data['attributes'][0, 0:1]  # (1, embed_dim)
    desc_embed = data['description'][0, 0:1]  # (1, embed_dim)

    # Generate chunks sequentially, passing prefix from previous chunk
    motions = []
    prev_normalized = None
    for s in starts:
        chunk_input = {
            'audio': audio[:, s:s + max_frames, :].to(device),
            'attributes': att_embed.unsqueeze(0).repeat(1, max_frames, 1).to(device),
            'description': desc_embed.unsqueeze(0).repeat(1, max_frames, 1).to(device),
            'audio_mask': torch.tensor([True], dtype=torch.bool).to(device),
            'att_mask': torch.tensor([True], dtype=torch.bool).to(device),
        }
        if 'text' in data:
            chunk_input['text'] = data['text']

        prefix_kwargs = {}
        if prev_normalized is not None:
            prefix_kwargs = {
                'prefix_motion': prev_normalized[:, -stride:, :],
                'prefix_frames': stride,
                'blend_frames': 20,
            }

        result = model.forward(chunk_input, **prefix_kwargs)
        prev_normalized = result.get('normalized_motion', None)
        motions.append(result['motion'].detach().cpu().numpy().squeeze(0))

    motion = np.stack(motions)  # (num_chunks, max_frames, 147)

    # Stitch overlapping chunks (alignment is light since prefix already ensures continuity)
    stitched_motion = stitch(motion, fk, interpolate=True)  # (T_stitched, 147)

    # Run FK on stitched result
    motion_fk = fk(stitched_motion)
    if isinstance(motion_fk, torch.Tensor):
        motion_fk = motion_fk.detach()
    else:
        motion_fk = torch.from_numpy(motion_fk).float()

    # Return with batch dimension to match model.forward() output format
    motion_tensor = torch.from_numpy(stitched_motion).float().unsqueeze(0)
    motion_fk = motion_fk.unsqueeze(0)

    return {
        'motion': motion_tensor,
        'motion_fk': motion_fk,
    }


def main(data_from='test',  data_name='AIST', id=None):
    # 1. Setup config and model
    config, args = load_config()
    link_type = data_name.lower()
    if config.data.representation == 'motorica':
        link_type = 'motorica'
    elif config.data.representation == 'smpl':
        link_type = 'smpl'

    exp_name = config.name

    save_path = os.path.join('./results', exp_name, args.dataset, 'motions')
    os.makedirs(save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = SegDataset(config, args, device=device)
    dataset_info = dataset.get_dataset_info()

    if config.model.name == 'vae':
        model = EditDance(config, dataset_info, mode='inference')
    elif config.model.name == 'cross_latent_diffusion':
        model = CrossEnergyEditDance(config, dataset_info, mode='inference')
    elif 'joint' in config.model.name:
        model = CrossEnergyEditDanceJoint(config, dataset_info, mode='inference')
    else:
        raise ValueError(f"Model {config.model.name} not supported")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    max_frames = config.data.motion_fps * config.data.chunk_time
    for data in tqdm(dataset):
        if data['audio'].shape[1] > max_frames:
            motion_recon = chunked_forward(model, data, max_frames, device)
        else:
            motion_recon = model.forward(data)
        motion_joint_recon = motion_recon['motion_fk'].detach().cpu().numpy()
        motion_recon = motion_recon['motion'].detach().cpu().numpy()
        motion_joint_gt = data['gt_motion_joint'].detach().cpu().numpy()
        motion_gt = data['gt_motion'].detach().cpu().numpy()
        key = data['key']
        output = {'motion_joint_gt': motion_joint_gt.squeeze(0), 
                    'motion_gt': motion_gt.squeeze(0),
                    'motion_joint_recon': motion_joint_recon.squeeze(0), 
                    'motion_recon': motion_recon.squeeze(0), 
                    'sample_id': key, 
                    'data_name': data_name,
                    'text': data['text'],
                    'data_type': link_type}
        np.save(os.path.join(save_path, 'recon_' + key + '.npy'), output)

if __name__ == "__main__":
    id = None
    # id = ['kthjazz_gCH_sFM_cAll_d02_mCH_ch01_whitemanpaulandhisorchestraloisiana_006_chunk138', 'kthjazz_gJZ_sFM_cAll_d02_mJZ_ch01_bennygoodmansugarfootstomp_003_chunk101', 'kthjazz_gCH_sFM_cAll_d02_mCH_ch01_whitemanpaulandhisorchestraloisiana_006_chunk29']
    main('test',
         data_name='motorica',
         id=id)