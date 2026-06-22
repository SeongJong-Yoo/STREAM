import os
import numpy as np
import torch
from omegaconf import OmegaConf
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


from src.utils.train.train_util import *
from src.models.pl_module.cross_energy_edit_dance_joint import CrossEnergyEditDanceJoint
from src.utils.train.callbacks import ProgressLogger

from src.losses.utils import metric_monitor

from src.dataloader.dataset_loader import get_datasets


def validate_multi_gpu_setup(config):
    """
    Validate that multi-GPU setup is correct before training starts.
    Raises AssertionError if multi-GPU setup is invalid.
    """
    configured_devices = config.trainer.devices
    available_gpus = torch.cuda.device_count()
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    print("=" * 80)
    print("MULTI-GPU SETUP VALIDATION")
    print("=" * 80)
    print(f"Configured devices in config: {configured_devices}")
    print(f"Available GPUs (torch.cuda.device_count()): {available_gpus}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices if cuda_visible_devices else 'Not set (all GPUs visible)'}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! Cannot use GPU training. "
            "Check your CUDA installation and GPU drivers."
        )
    
    # Check if multi-GPU is configured
    if configured_devices <= 1:
        raise AssertionError(
            f"Multi-GPU training requires config.trainer.devices > 1, "
            f"but got devices={configured_devices}. "
            f"Please set devices to the number of GPUs you want to use (e.g., devices: 2)."
        )
    
    # Check if enough GPUs are available
    if available_gpus < configured_devices:
        raise AssertionError(
            f"Insufficient GPUs available! "
            f"Config requires {configured_devices} GPUs, but only {available_gpus} GPU(s) detected. "
            f"Check your SLURM allocation (--gres=gpu:...) or CUDA_VISIBLE_DEVICES setting."
        )
    
    # Log GPU details
    print("\nGPU Details:")
    for i in range(available_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    
    print(f"\n✓ Multi-GPU setup validated: {configured_devices} GPU(s) will be used")
    print("=" * 80)
    return True


def create_trainer(config, loggers):
    # Validate multi-GPU setup before creating trainer
    if config.trainer.devices > 1:
        validate_multi_gpu_setup(config)
    
    # Check resume
    if config.train.resume:
        resume_path = Path(config.train.resume).parents[1]
        if os.path.exists(os.path.join(resume_path, 'wandb')):
            wandb_list = sorted(os.listdir(os.path.join(resume_path, "wandb")),
                                reverse=True)
            for item in wandb_list:
                if "run-" in item:
                    config.logger.wandb.id = item.split("-")[-1]
                    print(f"Wandb id: {config.logger.wandb.id}")
        else:
            raise FileNotFoundError(f"Wandb file not found at {resume_path}")

    rank = int(os.environ.get('RANK', 0))
    print(f"Rank: {rank}")
    pl.seed_everything(config.seed + rank)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ProgressLogger(metric_monitor=metric_monitor),
        ModelCheckpoint(
            verbose=True,
            dirpath=os.path.join(config.output_dir, 'checkpoints'),
            filename="{epoch}",
            monitor="total/val",
            mode="min",
            every_n_epochs=config.logger.checkpoint_epoch,
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=False,
        ),
    ]
    # if not config.train.resume:
    #     callbacks.append(pl.callbacks.RichProgressBar()) # Has some issues with resuming

    if config.trainer.devices > 1:
        ddp_strategy = "ddp_find_unused_parameters_true"
        print(f"\n✓ DDP Strategy: ddp_find_unused_parameters_true (using {config.trainer.devices} GPUs)")
    else:
        ddp_strategy = "auto"
        print(f"\n✓ Single GPU training (devices={config.trainer.devices})")

    trainer = pl.Trainer(
        logger=loggers, 
        callbacks=callbacks,
        strategy=ddp_strategy,
        default_root_dir=config.output_dir,
        log_every_n_steps=config.logger.val_every_steps,
        check_val_every_n_epoch=config.logger.val_every_steps,
        **config.trainer
    )
    
    # Verify trainer was created with correct device count
    if config.trainer.devices > 1:
        actual_devices = trainer.num_devices if hasattr(trainer, 'num_devices') else None
        if actual_devices is not None and actual_devices != config.trainer.devices:
            print(f"WARNING: Trainer reports {actual_devices} devices, but config specified {config.trainer.devices}")
        else:
            print(f"✓ Trainer initialized with {config.trainer.devices} device(s)")
    
    return trainer

def train(config, loggers):
    logger = logging.getLogger()

    # Create trainer
    trainer = create_trainer(config, loggers)
    logger.info("Trainer Created")
    print("Trainer Created")

    # # Create dataset
    dataset = get_datasets(config, 'train')
    dataset.setup('train')

    dataset_info = dataset.get_dataset_info()
    
    # Create model
    # model = EnergyEditDance(config, dataset_info)
    if 'joint' in config.model.name:
        model = CrossEnergyEditDanceJoint(config, dataset_info)
    logger.info("Model {} initialized".format(config.model.name))
    print("Model {} initialized".format(config.model.name))

    # Train
    if config.continue_train:
        trainer.fit(model, dataset, ckpt_path=config.train.resume)
        print("Training Resumed")
        logger.info("Training resumed from {}".format(config.train.resume))
    else:
        print("Training Started")
        trainer.fit(model, dataset)
        print("Training Completed")

    # Test
    # test_dataset = get_datasets(config, mode='test')
    print("Testing Started")
    dataset.setup('test')
    trainer.test(model, dataset)

def main():
    config = load_config()  
    loggers = set_logger(config)
    
    train(config, loggers)


if __name__ == "__main__":
    print(f"[PID {os.getpid()}] Rank: {os.environ.get('SLURM_PROCID')}, "
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"Num GPUs: {torch.cuda.device_count()}")
    main()