from argparse import ArgumentParser
from copy import deepcopy
from omegaconf import OmegaConf
import os
from pathlib import Path
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import glob
from src.utils.utility import resample

def load_config():
    parser = ArgumentParser()
    parser.add_argument("--folder", 
                        type=str,
                        required=False, 
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
    parser.add_argument("--audio_path", 
                        type=str,
                        required=False,
                        help="path to the audio file",
                        default=None)
    parser.add_argument("--dataset",
                        type=str,
                        required=False,
                        help="dataset name",
                        default=None)
    parser.add_argument("--from_test_label",
                        action='store_true', 
                        default=False,
                        required=False,
                        help="use test label for evaluation")

    args = parser.parse_args()
    args.cfg = os.path.join(args.folder, "config.yaml")
    config = OmegaConf.load(args.cfg)
    
    path = os.path.join(args.folder, 'checkpoints')
    model_path = os.path.join(path, f"{args.model_version}.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found")
    if args.model == "vae":
        config.model.VAE.pretrained_vae = model_path
    elif args.model == "diffusion" or args.model == "energy_diffusion" or args.model == "cross_energy_diffusion" or args.model == "cross_latent_diffusion":
        config.model.Diffusion.pretrained_diffusion = model_path
    elif args.model == "energy_flow" or args.model == "cross_energy_flow":
        config.model.Diffusion.pretrained_flow = model_path
    elif args.model == "energy" or args.model == "cross_energy_diffusion_joint" or args.model == "cross_energy_flow_joint" or args.model == "energy_flow_joint" or args.model == "flow_joint" or args.model == "diffusion_joint":
        config.model.pretrained_energy = model_path

    if args.dataset is not None:
        config.data.trained_dataset = deepcopy(config.data.dataset)
        config.data.dataset = [args.dataset]
    
    # config.data.data_extension = "lmdb"
    config.trainer.devices = 1

    return config, args

def adjust_original_fps(motion, curr_fps, tgt_fps, tgt_len, beat_based):
    if beat_based:
        result = resample(motion, tgt_fps, curr_fps, tgt_len)
    else:
        result = motion
    return result