import os
from pathlib import Path
import numpy as np
from copy import deepcopy
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from scipy import interpolate
import pytorch3d.transforms as t3d
import re
from tqdm import tqdm
from src.preprocessing.utils import process_smpl_data
from src.skeleton.smpl_fk import SMPLModel
from src.skeleton.preprocessing import motion_preprocessing
from src.models.pl_module.cross_energy_edit_dance_joint import CrossEnergyEditDanceJoint
from src.visualization.skeleton import create_video_from_keypoints
from src.utils.utility import CLIPLabelConverter
from src.dataloader.utility import Normalizer


def load_config():
    parser = ArgumentParser()
    parser.add_argument("--folder", 
                        type=str,
                        required=True, 
                        help="folder to load the saved model",
                        default=None)
    parser.add_argument("--model", 
                        type=str,
                        required=True,
                        help="vae | diffusion ",
                        default="vae")
    parser.add_argument("--model_version", 
                        type=str,
                        required=False,
                        help="model version",
                        default="last")
    parser.add_argument("--data_folder",
                        type=str,
                        required=True,
                        help="path to the data folder",
                        default='./data/test_data')
    parser.add_argument("--sample_id",
                        type=str,
                        required=False,
                        help="sample id to process one sample by one",
                        default=None)
    args = parser.parse_args()
    args.cfg = os.path.join(args.folder, "config.yaml")
    config = OmegaConf.load(args.cfg)
    
    path = os.path.join(args.folder, 'checkpoints')
    model_path = os.path.join(path, f"{args.model_version}.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found")
    if args.model == "vae":
        config.model.VAE.pretrained_vae = model_path
    elif 'joint' in args.model:
        config.model.pretrained_energy = model_path
    elif 'diffusion' in args.model:
        config.model.Diffusion.pretrained_diffusion = model_path

    config.data.trained_dataset = deepcopy(config.data.dataset)
    # config.data.dataset = [args.dataset]
    
    # config.data.data_extension = "lmdb"
    config.trainer.devices = 1

    return config, args

class InferenceDataset():
    def __init__(self, config, args, GT_MOTION, batch_size=128, device='cpu'):
        self.config = config
        self.args = args
        self.GT_MOTION = GT_MOTION
        self.model = config.model.name
        self.LABEL = True
        self.root_dir = './data/'
        if self.model == 'vae':
            self.LABEL = False
        self.device = device
        self.normalizer = None
        self.normalizer_latent = None

        self.skeleton_dict = None
        self.set_normalizer()
        self.batch_size = batch_size

        self.root_path = Path(args.data_folder)
        self.feature_path = list((Path(args.data_folder) / 'sliced_audio_features').glob('*.npy'))
        
        def extract_chunk_number(path):
            """Extract the number after 'chunk' from the filename"""
            match = re.search(r'chunk(\d+)', path.stem)
            if match:
                return int(match.group(1))
            return float('inf')  # Put files without chunk number at the end
        
        self.feature_path = sorted(self.feature_path, key=extract_chunk_number)
        self.label_converter = CLIPLabelConverter(embedding_path=Path(os.path.join('./data/motorica/dance_technique_embeddings_clip.pkl')))
        self.datas = self.load_data()
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

    def load_data(self):
        datas = {}
        if self.config.data.representation == 'smpl':
            fk = SMPLModel()
        else:
            raise ValueError(f"Not supported")

        counter = 0
        for feature_path in self.feature_path:
            id = feature_path.stem.split('_audio')[0]
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
                label_file = os.path.join(self.root_path, 'sliced_labels', f'{id}_label.npy')
                if os.path.exists(label_file):
                    data['label'] = np.load(label_file, allow_pickle=True)[()]['data']
                    data['label_index'] = np.load(label_file, allow_pickle=True)[()]['label_index']
                    # if self.config.data.use_neutralized_motion:
                    #     data['label_neutral'] = data['label'].copy() # HumanML3D data doesn't have neutralized motion
                else:
                    print(f'WARNING: {id} does not have label')
                    return None
                data['attributes'], data['description'] = self.label_converter.class_label_to_embedding(data['label'])
            
            if self.GT_MOTION:
                motion_path = self.root_path / 'sliced_motion_smpl' / f"{id}_motion.npy"
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
        output = {
            'audio': [],
            'attributes': [],
            'description': [],
            'label_index': [],
            'key': None
        }
        gt_motion = []
        for j in range(min(self.batch_size, len(self.keys) - index*self.batch_size)):
            id = self.keys[index*self.batch_size + j]
            data = self.datas[id]
            output['audio'].append(data['audio'].unsqueeze(0))
            output['attributes'].append(data['attributes'].unsqueeze(0))
            output['description'].append(data['description'].unsqueeze(0))
            if 'label_index' in data:
                output['label_index'].append(data['label_index'].unsqueeze(0))
            if 'motion' in data:
                gt_motion.append(data['motion'].unsqueeze(0))
        output['audio'] = torch.cat(output['audio'], dim=0).to(self.device)
        output['attributes'] = torch.cat(output['attributes'], dim=0).to(self.device)
        output['description'] = torch.cat(output['description'], dim=0).to(self.device)
        if 'label_index' in data:
            output['label_index'] = torch.cat(output['label_index'], dim=0).to(self.device)
        output['audio_mask'] = torch.tensor([True] * output['audio'].shape[0]).to(self.device)
        output['att_mask'] = torch.tensor([True] * output['attributes'].shape[0]).to(self.device)
        if len(gt_motion) > 0:
            output['motion'] = torch.cat(gt_motion, dim=0).to(self.device)
        return output
    
    @staticmethod
    def slice_data(data, fps, window_time=5):
        """
        Slice the data into chunks of 5 seconds
        """
        stride=2.5 # Don't change this value. 
        window_length = int(window_time * fps)
        total_length = data.shape[0]
        stride_length = int(stride * fps)
        num_bins = (total_length - window_length) // stride_length + 1

        output = []
        for i in range(num_bins):
            start_idx = i * stride_length
            end_idx = start_idx + window_length
            output.append(data[start_idx:end_idx])
        return np.stack(output)


    def __len__(self):
        return len(self.datas)

    def get_dataset_info(self):
        return {
            'normalizer': self.normalizer,
            'normalizer_latent': self.normalizer_latent,
            'skeleton_dict': self.skeleton_dict
        }

def extract_y_rotation(R):
    """
    Extract y-axis rotation from rotation matrix R
    R: (3, 3) rotation matrix
    """
    sin_y = R[0, 2]
    cos_y = torch.sqrt(R[0, 0]**2 + R[0, 1]**2)
    y_angle = torch.atan2(sin_y, cos_y)
    cos_new = torch.cos(y_angle)
    sin_new = torch.sin(y_angle)
    R_y_only = torch.tensor([[cos_new, 0, sin_new],
                          [0, 1, 0],
                          [-sin_new, 0, cos_new]], dtype=R.dtype, device=R.device)
    return R_y_only

def get_front_vector(motion, idx):
    motion = motion.squeeze()
    if isinstance(motion, torch.Tensor):
        motion = motion.detach().cpu().numpy()
    l_hip, r_hip, l_shoulder, r_shoulder = idx['l_hip'], idx['r_hip'], idx['l_shoulder'], idx['r_shoulder']
    x_axis = motion[l_hip, :] - motion[r_hip, :] 
    x_axis = x_axis / np.linalg.norm(x_axis, axis=-1, keepdims=True)
    z_axis = (motion[l_shoulder, :] + motion[r_shoulder, :]) / 2 - (motion[l_hip, :] + motion[r_hip, :]) / 2
    z_axis = z_axis - np.sum(z_axis * x_axis, axis=-1, keepdims=True) * x_axis
    z_axis = z_axis / np.linalg.norm(z_axis, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    # y_2d_vector = y_axis[1:]
    y_2d_vector = np.concatenate([y_axis[0:1], y_axis[2:]], axis=0)
    y_2d_vector = y_2d_vector / np.linalg.norm(y_2d_vector, axis=-1, keepdims=True)
    return y_2d_vector

def find_aligned_rotation(first_motion_joints, ref_motion_joints):
    idx = {'root': 0, 'l_hip': 1, 'r_hip':2, 'l_shoulder':13, 'r_shoulder':14} # for SMPL
    first_y_2d = get_front_vector(first_motion_joints, idx)
    ref_y_2d = get_front_vector(ref_motion_joints, idx)
    # Signed angle via atan2(cross, dot) — preserves rotation direction
    cross = first_y_2d[0] * ref_y_2d[1] - first_y_2d[1] * ref_y_2d[0]
    dot = np.sum(first_y_2d * ref_y_2d)
    angle = np.arctan2(cross, dot)
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    R = torch.from_numpy(R.astype(np.float32))
    return R, angle

def align_motion(motion, ref_motion, fk, RELATIVE=False):
    """
    motion: (T, 24, 3) or (T, 147)
    ref_motion: (24, 3) or (147)
    """
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    if isinstance(ref_motion, np.ndarray):
        ref_motion = torch.from_numpy(ref_motion)

    if RELATIVE:
        first_motion_joints = fk(motion[0:1])
        ref_motion_joints = fk(ref_motion.unsqueeze(0))
        aligned_R, prev_angle = find_aligned_rotation(first_motion_joints, ref_motion_joints)

        # Rotate translation path around the pivot (first frame), then shift to reference
        trans_centered = motion[:, :3] - motion[0:1, :3]  # (T, 3)
        trans_rotated = (aligned_R.T @ trans_centered.unsqueeze(-1)).squeeze(-1)  # (T, 3)
        trans = trans_rotated + ref_motion[:3].unsqueeze(0)

        R = t3d.rotation_6d_to_matrix(motion[:, 3:9])
        R = aligned_R.T @ R
        result = torch.cat([trans, t3d.matrix_to_rotation_6d(R), motion[:, 9:]], dim=-1)
    else:
        pass #TODO: Implement absolute motion alignment

    return result

def quaternion_slerp(q1, q2, t):
    """
    Spherical linear interpolation between quaternions q1 and q2
    q1, q2: (N, 4) quaternions
    t: scalar interpolation parameter [0, 1]
    """
    EPS = 1e-7
    
    # Ensure quaternions are normalized (with epsilon to prevent division by zero)
    q1_norm = torch.norm(q1, dim=-1, keepdim=True)
    q2_norm = torch.norm(q2, dim=-1, keepdim=True)
    q1 = q1 / torch.clamp(q1_norm, min=EPS)
    q2 = q2 / torch.clamp(q2_norm, min=EPS)
    
    # Compute dot product
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)  # (N, 1)
    
    # If dot product is negative, negate one quaternion to take shorter path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    
    # Clamp dot to valid range to prevent numerical issues
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation (element-wise check)
    DOT_THRESHOLD = 0.9995
    close_mask = dot > DOT_THRESHOLD
    
    # Calculate angle between quaternions
    theta_0 = torch.acos(dot)  # (N, 1)
    sin_theta_0 = torch.sin(theta_0)
    
    # Handle case where sin_theta_0 is very small (close to zero)
    sin_theta_0 = torch.clamp(sin_theta_0, min=EPS)
    
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    # Use linear interpolation for very close quaternions
    result = torch.where(close_mask, 
                        q1 + t * (q2 - q1),
                        s0 * q1 + s1 * q2)
    
    # Normalize result (with epsilon to prevent division by zero)
    result_norm = torch.norm(result, dim=-1, keepdim=True)
    result = result / torch.clamp(result_norm, min=EPS)
    
    return result

def interpolate_motion(start_motion, end_motion, num_frames=4, RELATIVE=False):
    """
    Interpolation between start_motion and end_motion with num_frames
    start_motion: (24, 3) or (147)
    end_motion: (24, 3) or (147)
    """
    # Convert to tensors if needed
    if isinstance(start_motion, np.ndarray):
        start_motion = torch.from_numpy(start_motion).float()
    if isinstance(end_motion, np.ndarray):
        end_motion = torch.from_numpy(end_motion).float()
    if RELATIVE:
        start_trans = start_motion[:3]
        end_trans = end_motion[:3]
        
        # Linear interpolation for translation
        t_values = torch.linspace(0, 1, num_frames).unsqueeze(1)  # (num_frames, 1)
        interpolated_trans = start_trans + t_values * (end_trans - start_trans)  # (num_frames, 3)
        start_rot = t3d.rotation_6d_to_matrix(start_motion[3:].reshape(-1, 6))
        end_rot = t3d.rotation_6d_to_matrix(end_motion[3:].reshape(-1, 6))
        
        # SLERP interpolation for rotation matrices
        # Convert rotation matrices to quaternions for SLERP
        start_quat = t3d.matrix_to_quaternion(start_rot)  # (N, 4)
        end_quat = t3d.matrix_to_quaternion(end_rot)      # (N, 4)
        
        # Create interpolation weights
        t_values = torch.linspace(0, 1, num_frames)  # (num_frames,)

        # Perform SLERP for each frame
        interpolated_quats = []
        for i in range(num_frames):
            t = t_values[i]
            interpolated_quat = quaternion_slerp(start_quat, end_quat, t)
            interpolated_quats.append(interpolated_quat)
        
        interpolated_quats = torch.stack(interpolated_quats, dim=0)  # (num_frames, N, 4)
        
        # Convert back to rotation matrices and then to 6D representation
        interpolated_rot_matrices = t3d.quaternion_to_matrix(interpolated_quats)  # (num_frames, N, 3, 3)
        interpolated_rot_6d = t3d.matrix_to_rotation_6d(interpolated_rot_matrices)  # (num_frames, N, 6)
        
        # Flatten the rotation 6D to match the original format
        interpolated_rot_6d = interpolated_rot_6d.reshape(num_frames, -1)  # (num_frames, N*6)
        
        # Combine translation and rotation
        interpolated_motion = torch.cat([interpolated_trans, interpolated_rot_6d], dim=-1)
        
    else:
        # For absolute motion, just use linear interpolation for now
        start_motion_tensor = torch.from_numpy(start_motion) if isinstance(start_motion, np.ndarray) else start_motion
        end_motion_tensor = torch.from_numpy(end_motion) if isinstance(end_motion, np.ndarray) else end_motion
        
        t_values = torch.linspace(0, 1, num_frames).unsqueeze(1)  # (num_frames, 1)
        interpolated_motion = start_motion_tensor + t_values * (end_motion_tensor - start_motion_tensor)

    return interpolated_motion


def stitch(motion_data, fk, interpolate=True):
    """
    Assume that the data is overlapped by stride 2.5 seconds (half of the total length, this is hardcoded in the slice_data function above and './src/preprocessing/compute_jukebox_feature.py')
    motion_data: (B, T, 24, 3) or (B, T, 147)
    """
    RELATIVE = False
    B = motion_data.shape[0]
    T = motion_data.shape[1]
    ctr = int(T / 2)
    stitched_motion = np.array(motion_data[0][:ctr])
    if motion_data.shape[-1] == 147:
        RELATIVE = True
        fk = SMPLModel()

    for i in range(1, B):
        last_motion = stitched_motion[-1]
        if i==B-1:
            ctr = T # Stitch all the data if it is the last batch
        new_motion = align_motion(motion_data[i][:ctr], last_motion, fk, RELATIVE)
        if interpolate:
            last_motion = stitched_motion[-3] 
            first_motion = new_motion[2]
            interpolated_motion = interpolate_motion(last_motion, first_motion, num_frames=4, RELATIVE=RELATIVE)
            if isinstance(interpolated_motion, torch.Tensor):
                interpolated_motion = interpolated_motion.detach().cpu().numpy()
            stitched_motion = np.concatenate([stitched_motion[:-2], interpolated_motion, new_motion[2:]])
        else:
            stitched_motion = np.concatenate([stitched_motion, new_motion])

    return stitched_motion


def main(config, args, GT_MOTION, BATCH_SIZE=-1):
    EXP_TYPE = args.model
    # Check data folder
    path_data = Path(args.data_folder)
    path_audio = str(path_data) + '/audio.wav'
    video_path = path_data

    if config.data.representation == 'motorica':
        raise ValueError(f"Motorica is not supported for long range generation")
    elif config.data.representation == 'smpl':
        link_type = 'smpl'
        fk = SMPLModel()

    if EXP_TYPE=='diffusion':
        pass #TODO: Implement attribute generation    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = InferenceDataset(config, args, GT_MOTION, BATCH_SIZE, device=device)
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

    # Deterministic Sampling
    samples = torch.randn(
        (1, config.data.motion_fps * config.data.chunk_time, config.data.body_features * config.data.num_joints+3),
        device=device,
        dtype=torch.float
    )
    motion_results = []
    gt_motions = []
    if BATCH_SIZE == -1:
        prev_normalized = None
        stride = config.data.motion_fps * config.data.chunk_time // 2  # overlap frames (75)
        for id in tqdm(dataset.datas.keys()):
            data = dataset.datas[id]

            model_input = {
                'audio': data['audio'].unsqueeze(0).to(device),
                'attributes': data['attributes'].unsqueeze(0).to(device),
                'noise': samples.clone().to(device),
                'description': data['description'].unsqueeze(0).to(device),
                'label_index': data['label_index'].unsqueeze(0).to(device),
                'audio_mask': torch.tensor([True], dtype=torch.bool, device=device),
                'att_mask': torch.tensor([True], dtype=torch.bool, device=device),
            }
            # Pass previous chunk's overlap as prefix for smooth transitions
            prefix_kwargs = {}
            if prev_normalized is not None:
                prefix_kwargs = {
                    'prefix_motion': prev_normalized[:, -stride:, :],
                    'prefix_frames': stride,
                    'blend_frames': 20,
                    'foot_optim': True,
                }
            result = model.forward(model_input, **prefix_kwargs)
            motion_recon = result['motion']
            prev_normalized = result.get('normalized_motion', None)
            motion_results.append(motion_recon)
            if 'motion' in data:
                gt_motions.append(data['motion'])
    else:        
        total_len = len(dataset.datas)
        batch_iterate = total_len // BATCH_SIZE + 1
        keys = list(dataset.datas.keys())
        for i in range(batch_iterate):
            model_input = dataset[i]
            model_input['audio'] = torch.cat([model_input['audio']] * 2, dim=1)
            model_input['description'] = torch.cat([model_input['description']] * 2, dim=1)
            model_input['label_index'] = torch.cat([model_input['label_index']] * 2, dim=1)
            model_input['attributes'] = torch.cat([model_input['attributes']] * 2, dim=1)
            motion_recon = model.forward(model_input)['motion']
            motion_results.append(motion_recon)
            if GT_MOTION:
                gt_motions.append(dataset[i]['motion'])
    
    motion_results = torch.cat(motion_results, dim=0).detach().cpu().numpy()
    motion_results = stitch(motion_results, fk, True)
    motion_results_joint = fk(motion_results).detach().cpu().numpy()
    if GT_MOTION:
        gt_motions = torch.stack(gt_motions, dim=0).detach().cpu().numpy()
        if gt_motions.ndim == 4:
            gt_motions = gt_motions.squeeze()
        motion_gt = stitch(gt_motions, fk, True)
        motion_gt_joint = fk(motion_gt).detach().cpu().numpy()
    else:
        motion_gt = None
        motion_gt_joint=None

    output = {
        'recon_motion': motion_results,
        'recon_motion_joint': motion_results_joint,
    }
    if GT_MOTION:
        output['gt_motion'] = motion_gt
        output['gt_motion_joint'] = motion_gt_joint
    np.save(os.path.join(str(path_data), 'result.npy'), output)
        
    create_video_from_keypoints(keypoints=motion_results_joint,
                                output_path=os.path.join(video_path, "result.mp4"),
                                audio_path=path_audio,
                                link_type=link_type,
                                gt=motion_gt_joint,
                                gt_link_type=link_type,
                                fps=config.data.motion_fps,
                                flipped=True)

if __name__ == "__main__":
    config, args = load_config()
    GT_MOTION = False
    BATCH_SIZE=-1

    # Check data folder
    path_data = Path(args.data_folder)
    path_audio = path_data / 'wav'
    path_audio_features = path_data / 'audio_features'
    path_motion = path_data / 'sliced_motion_smpl'

    if not path_motion.exists():
        GT_MOTION = False

    main(config, args, GT_MOTION, BATCH_SIZE=BATCH_SIZE)
