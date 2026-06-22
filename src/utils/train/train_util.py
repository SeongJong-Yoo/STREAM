from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from argparse import ArgumentParser
from omegaconf import OmegaConf
import logging
import os
from pathlib import Path
import time
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import glob

def load_config():
    parser = ArgumentParser()
    parser.add_argument("--cfg", 
                        type=str,
                        required=False, 
                        default="config/config.yaml")
    parser.add_argument("--folder", 
                        type=str,
                        required=False, 
                        help="folder to load the saved model",
                        default=None)

    args = parser.parse_args()

    if args.folder is not None:
        args.cfg = os.path.join(args.folder, "config.yaml")

    config = OmegaConf.load(args.cfg)
    if args.folder is not None:
        config.continue_train = True
        path = os.path.join(args.folder, 'checkpoints')

        last_epoch = 0
        for file in glob.glob(os.path.join(path, '*.ckpt')):
            if not file.endswith("last.ckpt"):
                continue
            config.train.resume = file
            last_epoch = file.split("=")[-1].split(".")[0]
        config.output_dir = "./outputs/"
        config.name = config.name + "_resume_" + str(last_epoch)

    models = ['cross_latent_diffusion']
    if config.model.name in models:
        if not config.model.VAE.pretrained_vae.endswith('ckpt'):
            path_vae = config.model.VAE.pretrained_vae
            vae_config = os.path.join(path_vae, 'config.yaml')
            vae_config = OmegaConf.load(vae_config)
            config.model.VAE = vae_config.model.VAE
            if config.model.latent_dim != vae_config.model.latent_dim:
                raise ValueError(f"Latent dim mismatch: {config.model.latent_dim} != {vae_config.model.latent_dim}")
            
            ckpt_path = os.path.join(path_vae, 'checkpoints')
            for file in os.listdir(ckpt_path):
                if file.endswith("ckpt"):
                    config.model.VAE.pretrained_vae = os.path.join(ckpt_path, file)
                    break

    # if config.data.beat_based:
    #     if config.data.dir.split('/')[-1] == 'AIST':
    #         config.data.dir = config.data.dir.replace('AIST', 'AIST_beats')

    # Create output directory
    days = time.strftime("%Y-%m-%d")
    times = time.strftime("%H-%M-%S")
    save_dir_config(config, days, times)
    config.evaluation = False

    return config


def set_logger(config):
    python_logger = python_logger_config(config)

    if python_logger is None:
        python_logger = logging.getLogger()
        python_logger.setLevel(logging.CRITICAL)
    else:
        python_logger.info(OmegaConf.to_yaml(config))
    
    loggers = []

    if config.logger.wandb.project is not None:
        wb_logger = WandbLogger(
            project=config.logger.wandb.project,
            entity=config.logger.wandb.entity,
            offline=config.logger.wandb.offline,
            id=config.logger.wandb.id,
            save_dir=config.output_dir,
            version="",
            name=config.name,
            anonymous=False,
            log_model=False,
            config=OmegaConf.to_container(config, resolve=True)
        )
        loggers.append(wb_logger)

    if config.logger.tensorboard:
        tb_logger = TensorBoardLogger(
            save_dir=config.output_dir,
            sub_dir='tensorboard',
            name="",
            version="",
        )
        loggers.append(tb_logger)

    print("Setting up logger complete")
    return loggers
        
@rank_zero_only
def python_logger_config(config):
    head = '%(asctime)-15s %(message)s'
    log_file = os.path.join(config.output_dir, '.log')
    logging.basicConfig(filename=log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter(head))
    file_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(file_handler)
    return logger


@rank_zero_only
def save_dir_config(config, days, times):
    root_dir = config.output_dir
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    names = config.name
    output_dir = os.path.join(root_dir, days, times + '_' + names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output_dir)
    config_path = os.path.join(output_dir, 'config.yaml')
    OmegaConf.save(config, config_path)
    print(f"Configuration setup complete, model will be saved in {output_dir}")
    