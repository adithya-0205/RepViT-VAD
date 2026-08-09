# RepViT-VAD

RepViT-VAD is a video anomaly detection project that combines a RepViT-M1.0 backbone, a temporal convolutional network (TCN), and a classification head for clip-based anomaly classification. The repository includes training, continual-learning, quantization-aware training (QAT), and ONNX export workflows.

## What this repository does

- Loads video clips as image-frame folders instead of raw videos.
- Builds clips of 16 frames, resizes them to 224x224, and feeds them through a RepViT-based model.
- Supports continual learning with replay buffers, EWC, and knowledge distillation.
- Can export the trained model to ONNX for RISC-V / accelerator deployment.

## Model overview

The main model is implemented in [models/vad_model.py](models/vad_model.py) and uses:

- RepViT backbone from [models/repvit_backbone.py](models/repvit_backbone.py)
- TCN from [models/tcn.py](models/tcn.py)
- A final classifier head that outputs class logits

The dataset loader in [video_dataset.py](video_dataset.py) expects frame folders organized by class and video clip.

## Project structure

```text
RepViT-VAD/
├── continual_train.py       # Continual-learning training loop
├── train.py                 # Training entry point for the VAD pipeline
├── qat_train.py             # QAT / INT8 quantization workflow
├── export_riscv_onnx.py     # Export to ONNX for RISC-V deployment
├── extract_frames.py        # Helper to extract frames from videos
├── video_dataset.py         # Dataset loader for frame-based clips
├── models/                  # Backbone, TCN, and VAD model code
├── replay_buffer/           # Saved replay clips for continual learning
├── dataset/                 # Training/validation data root
└── requirements.txt
```

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

This project also expects a working PyTorch installation.

## Dataset layout

The training code looks for data under the default folder structure:

```text
dataset/
├── train/
│   ├── normal/
│   │   └── video_001/
│   │       ├── frame_0001.jpg
│   │       ├── frame_0002.jpg
│   │       └── ...
│   └── fighting/
│       └── video_002/
│           ├── frame_0001.jpg
│           └── ...
└── val/
    ├── normal/
    └── fighting/
```

Each video folder is treated as a sample clip. The loader selects 16 frames by default and resizes them to 224x224.

## Extract frames from raw videos

If you have raw videos, you can use the helper script:

```bash
python extract_frames.py
```

It writes frame images into folders under the extracted_frames directory.

## Training

### Standard training

```bash
python train.py --new-class fighting
```

This script trains the VAD model and saves checkpoints such as:

- best_model.pth
- classes.json
- replay_buffer/

### Continual-learning training

The repository also contains a continual-learning workflow with EWC, knowledge distillation, and replay:

```bash
python continual_train.py
```

## Quantization-aware training

To run QAT and save an INT8 checkpoint:

```bash
python qat_train.py
```

Output:

- repvit_m1_0_tcn_int8.pth

## ONNX export

To export the model for ONNX / RISC-V deployment:

```bash
python export_riscv_onnx.py --output repvit_m1_0_tcn_riscv.onnx
```

You can also create a static-shape export with:

```bash
python export_riscv_onnx.py --output repvit_m1_0_tcn_riscv.onnx --static
```

## Notes

- The default training configuration uses a clip length of 16 frames.
- The model is currently built around a small set of classes such as normal and fighting, but the code is structured so additional classes can be added later.
- The repository includes checkpoint and replay-buffer artifacts that are generated during training runs.

## Citation

If you use this code in your work, please cite the original RepViT paper:

```bibtex
@inproceedings{wang2024repvit,
  title={RepViT: Revisiting Mobile CNN from ViT Perspective},
  author={Wang, Ao and Chen, Hui and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
