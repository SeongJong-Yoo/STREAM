# Dance Design Studio

Interactive tool for designing and editing dance motion with SMPL mesh visualization.

## Install

```bash
conda activate edance
pip install PyQt5 pyqtgraph PyOpenGL
```

## Run

From the project root (`STREAM/`):

```bash
conda activate edance
python -m app.main
```

## Project Folder Structure

To load a project, prepare a folder with:

```
your_project/
├── audio.wav              # music file
├── label/                 # per-segment .txt label files
│   ├── segment_0.txt
│   ├── segment_1.txt
│   └── ...
└── model/
    ├── config.yaml
    └── checkpoints/
        └── last.ckpt
```

Each label `.txt` file follows this format:

```
genre: charleston
label: cross step
start: 999
end: 1044
fps: 30
description: The person steps out with one foot and crosses the other behind it.
```

Audio features (`sliced_audio_features/`) are auto-computed on first load if missing.
