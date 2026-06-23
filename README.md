# Structural-Temporal Rhythmic Energy-based Attention for dance Motion generation (STREAM)

<p align="center">
  <img src="media/Short_demo.gif" width="720" alt="STREAM — Dance Design Studio demo">
  <br>
  <em>Dance Design Studio — interactive editing demo.</em>
</p>


This repository contains code for Structure-Temporal Rhythmic Energy-based Attention for Dance Motion Generation. The code is provided without any warranty. If using this code, please cite our work as shown below. For more information and results please visit our [project website](https://sj-yoo.info/stream/) or check our [paper](https://arxiv.org/abs/2606.22726)


    @inproceedings{yoo2026stream,
    title     = {Text Dictates, Music Decorates: Energy-based Attention for Editable Dance Motion Generation},
    author    = {Yoo, Seong Jong and Peng, Siyuan and Gu, Felix and Aloimonos, Stratis and Fermüller, Cornelia},
    journal = {European Conference on Computer Vision, ECCV},
    year      = {2026},
    }

# News
- [2026/06/21] first release scripts, dataset, weights, and demo

# Installation

STREAM is developed and tested with **Python 3.9** and **PyTorch 2.2.2 (CUDA 11.8)**.
The steps below recreate the verified `edance` environment. Run them in order — PyTorch,
PyTorch3D, and the git-based packages must be installed *before* `requirements.txt`.

### 1. Create the conda environment

```bash
conda create -n edance python=3.9 -y
conda activate edance
```

### 2. Install PyTorch (CUDA 11.8 build)

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install PyTorch3D

Use the prebuilt wheel matching `py39 / cu118 / pyt222`:

```bash
pip install --no-index --no-cache-dir pytorch3d==0.7.8 \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt222/download.html
```

> If no matching wheel is available for your platform, build from source instead:
> `pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"`
> (requires a CUDA toolkit matching your PyTorch build). See the
> [official install guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

### 4. Install the core dependencies

```bash
pip install -r requirements.txt
```

### 5. Install git-based packages

```bash
# OpenAI CLIP (text/motion embeddings)
pip install git+https://github.com/openai/CLIP.git

# jukemirlib (Jukebox audio features)
pip install git+https://github.com/rodrigo-castellon/jukemirlib.git
```

### 6. (Optional) Demo / Dance Design Studio GUI

Only needed to run the interactive app — see the [Demo](#demo-dance-design-studio) section.

```bash
pip install -r requirements-demo.txt
```

# File Structure

```text
STREAM/
├── train.py                   # Training entry point
├── eds_experiment.py          # Editable dance-synthesis experiments
├── long_range_generation.py   # Long-range / full-song generation + stitching
├── evaluate_and_save_seg.py   # Metric evaluation on text-based segmentation
├── requirements.txt           # Core dependencies
├── requirements-demo.txt      # GUI-only dependencies
├── config/                    # Hydra training configs
│   ├── stream.yaml            #   STREAM on Motorica++
│   └── stream_hm.yaml         #   STREAM on HumanML3D
├── src/                       # Model, dataloaders, losses, metrics, skeleton/FK, visualization
│   └── metric/checkpoints/    # Feature-extractor weights for evaluation (gitignored)
├── app/                       # Dance Design Studio — interactive PyQt5 GUI (see Demo)
├── demo/                      # Ready-to-run demo samples used by the GUI
│   ├── long_samples/ex1..ex7  # Long generation examples (audio + labels + motion)
│   └── techniques/            # Short dance technique examples
├── media/                     # README assets (demo video + thumbnail)
├── data/                      # Datasets + stats  ── downloaded separately (see below)
└── outputs/                   # Trained models + logs ── downloaded separately (see below)
```

- **Note:** `data/` and `outputs/` are **not** tracked in git (only `.gitkeep` placeholders are committed; see `.gitignore`). Locally they hold ~73 GB and ~5 GB respectively, so you populate them by downloading from the links in the [Dataset](#dataset) section and the released weights.

## `data/`

Data root (`data.dir: "./data"` in the configs). Each sample is sliced into fixed-length **chunks**.

```text
data/
├── smpl/
│   └── SMPL_NEUTRAL.pkl          # SMPL neutral body model — required for FK / mesh rendering
├── motorica/                     # Motorica++ training set (SMPL representation)
│   ├── sliced_audio/             #   per-chunk audio clips                (*.wav)
│   ├── sliced_audio_features/    #   precomputed Jukebox audio features   (*_audio.npy)
│   ├── sliced_beats/             #   per-chunk beat annotations           (*.npy)
│   ├── sliced_labels/            #   per-chunk text labels                (*_label.json)
│   ├── sliced_motion/            #   motion in Motorica format            (optional)
│   ├── sliced_motion_smpl/       #   per-chunk SMPL motion                (*_motion.npy)
│   ├── gt_features/              #   ground-truth features for FID / metric evaluation
│   │   ├── kinetic_features/     #     kinetic features                  (*.npy)
│   │   └── manual_features_new/  #     manual / geometric features        (*.npy)
│   ├── Mean.npy, Std.npy         #   normalization statistics
│   ├── motion_stats.npz          #   motion statistics
│   ├── label_clip_cache.pkl      #   cached CLIP text embeddings
│   └── label_t5_cache.pkl        #   cached T5 text embeddings
├── motorica_seg/                 # text-based segmentation set (metric evaluation)
│   ├── sliced_audio/             #   (*.wav)
│   ├── sliced_audio_features/    #   (*_audio.npy)
│   ├── sliced_labels/            #   per-chunk text labels                (*_label.txt)
│   ├── sliced_motion_smpl/       #   per-chunk SMPL motion                (*_motion.npy)
│   └── VQ_tokens/                #   VQ motion tokens                     (*.txt)
└── HumanML3D/                 # text-based segmentation set (metric evaluation)
```

## `outputs/`

Training output root (`output_dir: "./outputs/"`). Each run directory is named after `config.name`
and contains the resolved config plus the checkpoint that the GUI and generation scripts load.

```text
outputs/
├── STREAM/                       # STREAM trained on Motorica++
│   ├── config.yaml               #   resolved Hydra config (loaded at inference)
│   ├── checkpoints/
│   │   └── last.ckpt             #   PyTorch-Lightning checkpoint
│   ├── tensorboard/              #   TensorBoard event logs
│   └── wandb/                    #   Weights & Biases run logs
├── STREAM_HM/                    # STREAM trained on HumanML3D (same layout)
│   └── ...
```

In the GUI, **Load Model** points at one of these run directories (it reads `config.yaml` +
`checkpoints/last.ckpt`).

# Dataset
## Motorica++
- Motorica++ is based on [Motorica dataset](https://github.com/simonalexanderson/MotoricaDanceDataset/). All copyright of main (motion and audio) dataset belongs to original Motorica Dataset.
- Motorica++ is frame-level text annotation, you can find at [here (raw data)](https://drive.google.com/file/d/1bNfNdJeZlC2hRRD3R-YFWxR-hg-HNefB/view?usp=sharing) 
- If you want preprocessed train/evaluate ready data then please check [here](https://drive.google.com/file/d/1lWTXXB-a1AIUpCOuPjYXoZGfgOGtsjb0/view?usp=sharing)
- For metric evaluation, we use text-based segmentation. You can find them at [here](https://drive.google.com/file/d/1xlyxcyqizXGc9o3Q6n9lXRk_cc5kBrM6/view?usp=drive_link)
- If you want to process data from scratch please run './src/preprocessing/data_prep.py'
    - Before running, please properly set the raw dataset path at 'line 240'

## HumanML3D
- We use SMPL format of HumanML3D dataset
    - You can download them from [here](https://drive.google.com/file/d/1_5x7KuSTuU1YMxSKEY4ay3dNt8qSFlY5/view?usp=drive_link) or get from original [HumanML3D data](https://github.com/EricGuo5513/HumanML3D) then optimize them with SMPL format
- If you want to process data from scratch please run './src/preprocessing/humanml3d_data_prep.py'

# Evaluate 
## Pre-trained model weights
- We provide two different version of pre-trained STREAM models
    - STREAM: Trained solely on Motorica++ dataset
    - STREAM_HM: Trained on Motorica++ and HumanML3D
- To download weights, please run './script/download_weights.sh'

## Compute EDS metric
- Our EDS metric computes the ratio between text-motion and BAS. 
- We use [TMR](https://github.com/Mathux/TMR) to evaluate text-motion alignment. We finetuned TMR models and you can download model weights at [here](https://drive.google.com/file/d/1EI4yy9eoLxiW1gT0CBRJWUhAoas5nGcZ/view?usp=drive_link)

1. Clone TMR repository
2. Download finetuned model weights and locate them at 'TMR/outputs/finetune_motorica'
3. Run `python eds_experiment.py --folder ./outputs/STREAM --model cross_energy_diffusion_joint --model_version last --dataset motorica_seg --tmr_run_dir {TMR/outputs/finetune_motorica} --tmr_dir {DIR of TMR}`
4. Then the results will be saved at './results/STREAM/eds/eds_results.json'

## Compute other metrics
- In order to compute other metrics, please run the followings:
```
python evaluate_and_save_seg.py --folder ./outputs/STREAM --model cross_energy_diffusion_joint --model_version last --dataset motorica_seg
python ./src/metric/evaluation_metrics.py --exp_name STREAM --dataset motorica_seg
```

- The results will be saved at './results/STREAM/motorica_seg'


# Demo (Dance Design Studio)
## Install

```bash
conda activate edance
pip install -r requirements-demo.txt
```

- Download demo data from [here](https://drive.google.com/file/d/15ichJ6_znWBGjmhe85wqVwv3XTCeOUE8/view?usp=drive_link). 'demo' folder contains demo datasets, including long-range generation and short dance technique examples. 
- The AI suggestion feature uses the Google Gemini API. Provide your key in-app via the
- **"Gemini API Key"** button (`app/frontend/main_window.py`); no environment variable is required.

## Run

From the project root (`STREAM/`):

```bash
conda activate edance
python -m app.main
```

1. Click 'Load Data' -> Select './demo/long_samples/ex1'
2. Click 'Load Model' -> Select './outputs/STREAM' (or STREAM_HM)
3. (Optional) Put Gemini API Key if you want to use AI-Suggestion 
4. Click 'Generate'

## How to edit
- You can find editing panel at right section. 
- When you click one of label at timeline panel, it is lightly highlighted and 'Time Range', 'Genre', 'Label', and 'Description' are accordingly changed. 
- You can edit label by choosing different 'Genre', 'Label', and writing 'Description' (or using AI-Suggestion by 'Suggest'). After editing, when you click 'Apply' then it will re-generate corresponding part. 
- When you change 'Time Range', then you can split label into two different ones
- When you click multiple labels with 'shift' key, and click 'Apply', then you can 'merge' the labels into single label


# Train from scratch
- Prepare dataset [Dataset](#dataset) 
- Run `python train.py --cfg ./config/stream.yaml`

# Acknowledgement
We would like to thank to all the open source repositories. Our code and work are able to build on top of their works. Especially, [Motorica](https://github.com/simonalexanderson/MotoricaDanceDataset/tree/main), [EDGE](https://github.com/stanford-tml/edge), [MDM](https://github.com/guytevet/motion-diffusion-model), [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), [Energy-Based Cross-Attention](https://github.com/EnergyAttention/Energy-Based-CrossAttention), [Modern Hopfield Networks](https://github.com/ml-jku/hopfield-layers)

# License
STREAM code and the Motorica++ annotations are released under **CC BY-NC 4.0** (non-commercial; academic use permitted) — see [LICENSE](LICENSE). The underlying Motorica **motion/audio is *not* covered** and remains under the [original Motorica Dataset license](https://github.com/simonalexanderson/MotoricaDanceDataset/blob/main/LICENSE.txt) (music audio has separate copyright).

# Bug Report
Please raise an issue on Github for issues related to this code. If you have any questions related about the code feel free to send an email to here (yoosj@umd.edu).
