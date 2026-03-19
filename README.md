# Pitch Detection Framework

End-to-end tooling for baseball pitch event detection using multimodal features:
- Pose features from YOLOv8 pose estimation
- RGB video features from ResNet50 embeddings
- A temporal model for frame-level pitch probability and segment extraction

The main package is in `pitch_detection/` and includes preprocessing, training, inference, and evaluation utilities, plus a Typer CLI.

## Repository Layout

- `pitch_detection/`: core library and CLI
- `pose_features/`, `video_features/`: extracted features used for training/inference
- `features/`: label JSON files and legacy feature artifacts
- `checkpoints/`: saved model checkpoints
- `examples/`: helper scripts
- `notebooks/`: experimentation notebooks

## Hardware Requirements

### Minimum (CPU)
- Linux or macOS
- Python 3.10+
- 16 GB RAM

### Recommended (GPU)
- NVIDIA GPU with CUDA support
- 8+ GB VRAM
- 32+ GB RAM

## Python Dependencies

Install the packages used by the current codebase:

```bash
python -m pip install --upgrade pip
python -m pip install \
	torch torchvision \
	numpy scipy \
	opencv-python pillow \
	ultralytics \
	typer tqdm
```

Notes:
- `ultralytics` is required for pose extraction.
- PyTorch install commands vary by CUDA version. If needed, use the selector at pytorch.org to install a CUDA-matched build.

## Quick Start

### 1) Verify environment

```bash
python -m pitch_detection.cli version
```

### 2) Preprocess one or more videos

```bash
python -m pitch_detection.cli preprocess path/to/game1.mp4 path/to/game2.mp4 \
	--output-dir . \
	--yolo-model yolov8l-pose.pt \
	--batch-size 64 \
	--pose-conf 0.3 \
	--sample-rate 1 \
	--device auto
```

This creates:
- `pose_features/<video_name>_pose_features.json`
- `pose_features/<video_name>_pose_features_summary.json`
- `video_features/<video_name>_video_features.npy`
- `video_features/<video_name>_video_metadata.json`
- `preprocessing_summary.json`

### 3) Train a model

```bash
python -m pitch_detection.cli train \
	--pose pose_features/game1_pose_features.json \
	--video video_features/game1_video_features.npy \
	--labels features/game1_labels.json \
	--epochs 50 \
	--batch-size 32 \
	--lr 0.001 \
	--checkpoint-dir checkpoints \
	--device auto
```

Output:
- best checkpoint in `checkpoints/` with filename pattern `best_model_iou_*.pth`

### 4) Predict pitch segments

```bash
python -m pitch_detection.cli predict \
	--pose pose_features/game1_pose_features.json \
	--video video_features/game1_video_features.npy \
	--checkpoint-dir checkpoints \
	--threshold 0.5 \
	--fps 30 \
	--output predictions.json
```

### 5) Evaluate against labels

```bash
python -m pitch_detection.cli evaluate \
	--pose pose_features/game1_pose_features.json \
	--video video_features/game1_video_features.npy \
	--labels features/game1_labels.json \
	--checkpoint-dir checkpoints \
	--threshold 0.3 \
	--fps 30 \
	--output evaluation.json
```

## CLI Reference

All commands:

```bash
python -m pitch_detection.cli --help
```

Key commands:
- `train`: train a model from pose/video/label files
- `predict`: run inference and output detected pitch segments
- `evaluate`: compute precision/recall/F1/IoU against labeled segments
- `preprocess`: extract pose + video features from raw videos
- `find-checkpoint`: locate best checkpoint by IoU in a directory
- `info`: print checkpoint metadata and final training metrics
- `config`: show or export current config
- `version`: print framework and runtime version info

## Expected Label Format

Training labels are parsed from JSON files that contain `videoLabels` entries with `ranges` and `timelinelabels`, where pitch ranges are selected when the label includes `Pitch`.

If your labels use a different schema, update parsing logic in `pitch_detection/dataset.py`.

## Configuration

Default settings live in `pitch_detection/config.py`, including:
- model dimensions
- training hyperparameters
- loss weighting
- inference thresholding behavior
- evaluation IoU threshold

You can inspect active config from CLI:

```bash
python -m pitch_detection.cli config
python -m pitch_detection.cli config --export config_snapshot.json
```

## Troubleshooting

- No checkpoint found:
	- confirm `checkpoints/` contains files matching `best_model_iou_*.pth`
- Preprocessing fails with YOLO import error:
	- install `ultralytics`
- CUDA not used:
	- verify `python -m pitch_detection.cli version` and PyTorch CUDA install
- Shape or file mismatch errors:
	- ensure pose/video/label files correspond to the same source video and frame timeline

## Development Notes

- Entry point for module execution: `python -m pitch_detection.cli`
- Package version is defined in `pitch_detection/__init__.py`
- Current `setup.py` is empty; this repository is currently run directly from source.