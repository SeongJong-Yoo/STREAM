
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, get_worker_info, DistributedSampler
from copy import deepcopy
import random

# import MotoricaDataset
from .base_dataset import BaseDataset, Fusion
from .aist_dataset import AISTDataset
from .motorica_dataset import MotoricaDataset
from .humanml3d_dataset import Humanml3D

DATASET_DICT = {
    'aist': AISTDataset,
    'motorica': MotoricaDataset,
    'humanml3d': Humanml3D,
}

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    fusion = worker_info.dataset
    if fusion.h5_load:
        fusion.set_h5_files()
    elif fusion.lmdb_load:
        fusion.set_lmdb_files()

class DataModule(pl.LightningDataModule):
    def __init__(self, datasets: list[BaseDataset], config, mode='train', id=None):
        super().__init__()
        self.config = config
        self.name = config.data.name

        # Parameters for DataLoader
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.mode = mode
        
        if config.trainer.accelerator == 'gpu':
            self.pin_memory = True
        else:
            self.pin_memory = False

        self.datasets = datasets

        # Initialize as None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.normalizer = None
        self.normalizer_latent = None
        self.skeleton_dict = None

        # Multi-GPU
        self.multi_gpu = False
        if config.trainer.devices > 1:
            self.multi_gpu = True

    def get_dataset_info(self):
        return {
            'normalizer': self.normalizer,
            'normalizer_latent': self.normalizer_latent,
            'skeleton_dict': self.skeleton_dict
        }
    
    def setup(self, stage=None, id=None):
        if id is not None:
            print(f"setup called with id={id}")
            self.sample_data = Fusion(self.config, self.datasets, self.mode, id)
            self.normalizer = self.sample_data.normalizer
            self.normalizer_latent = self.sample_data.normalizer_latent
            self.skeleton_dict = self.sample_data.skeleton_dict
            return

        print(f"setup called with stage={stage}")
        if stage == 'fit' or stage == 'train':# or stage == 'validate':
            self.mode = 'train'
            if self.train_data is None:
                self.train_data = Fusion(self.config, self.datasets, self.mode)
                self.normalizer = self.train_data.normalizer
                self.normalizer_latent = self.train_data.normalizer_latent
                self.skeleton_dict = self.train_data.skeleton_dict
                if self.val_data is None:
                    self.val_data = deepcopy(self.train_data)
                    self.val_data.switch_mode('val')

            print(f'INFO: Training on {len(self.train_data)} Chunks')
            print(f'INFO: Validation on {len(self.val_data)} Chunks')

        elif stage == 'validate':
            self.mode = 'val'
            if self.val_data is None:
                if self.train_data is None:
                    self.val_data = Fusion(self.config, self.datasets, self.mode)
                else:
                    self.val_data = deepcopy(self.train_data)
                    self.val_data.switch_mode('val')

            print(f'INFO: Validation on {len(self.val_data)} Chunks')
        
        elif stage == 'test':
            self.mode = 'test'
            if self.test_data is None:
                if self.train_data is None:
                    self.test_data = Fusion(self.config, self.datasets, self.mode)
                else:
                    self.test_data = deepcopy(self.train_data)
                    self.test_data.switch_mode('test')

            print(f'INFO: Testing on {len(self.test_data)} Chunks')
            self.normalizer = self.test_data.normalizer
            self.normalizer_latent = self.test_data.normalizer_latent
            self.skeleton_dict = self.test_data.skeleton_dict
        
        
    def on_train_epoch_start(self):
        """Re-shuffle with dataset grouping at the start of each epoch."""
        if self.train_data is not None and hasattr(self.train_data, 'reshuffle_train'):
            self.train_data.reshuffle_train()

    def _mp_context(self):
        # Spawn so DataLoader workers can initialize their own CUDA context;
        # forked workers cannot use CUDA after the parent has touched it
        # (Lightning moves the model to GPU before iterating the loader).
        # Only applied when num_workers > 0; setting it for 0 raises.
        return "spawn" if self.num_workers > 0 else None

    def train_dataloader(self, shuffle=True):
        # Disable DataLoader shuffle — we handle it via dataset-grouped shuffle
        # in Fusion._dataset_grouped_shuffle() to preserve LMDB read locality
        use_shuffle = shuffle and not self.train_data.lmdb_load
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=use_shuffle,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            prefetch_factor=4,
            persistent_workers=True if self.num_workers > 0 else False,
            multiprocessing_context=self._mp_context(),
        )

    def val_dataloader(self, shuffle=False):
        # if self.multi_gpu:
        #     sampler = DistributedSampler(self.val_data, shuffle=shuffle)
        # else:
        sampler = None
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            # sampler=sampler,
            worker_init_fn=worker_init_fn,
            prefetch_factor=4,
            persistent_workers=True if self.num_workers > 0 else False,
            multiprocessing_context=self._mp_context(),
        )

    def test_dataloader(self, shuffle=False):
        # if self.multi_gpu:
        #     sampler = DistributedSampler(self.test_data, shuffle=shuffle)
        # else:
        sampler = None
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=self._mp_context(),
        )

    def sample_dataloader(self):
        return DataLoader(
            self.sample_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=self._mp_context(),
        )

def get_datasets(config, mode='train', id=None):
    # dataset_name = config.data.name
    dataset_list = []
    for dataset in config.data.dataset:
        dataset_list.append(DATASET_DICT[dataset.lower()])
    data_module = DataModule(dataset_list, config, mode, id)
    
    return data_module