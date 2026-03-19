clear
# Pitch Detection Framework

A modular framework for detecting pitch events in baseball videos using pose and video features.

## Structure

```
pitch_detection/
├── __init__.py          # Package initialization and exports
├── model.py             # Neural network architecture (MultiModalBMN)
├── dataset.py           # Dataset and dataloader utilities
├── loss.py              # Loss functions (IoU, Dice, Precision, Tversky, etc.)
├── metrics.py           # Evaluation metrics and utilities
├── trainer.py           # Training utilities
├── inference.py         # Inference and model loading utilities
└── preprocess.py        # Video preprocessing (YOLO pose + RGB features)
```

## Usage

### Preprocessing Videos

The preprocessing module extracts both pose features (using YOLOv8-Pose) and RGB video features (using ResNet50) from input videos.

#### Command Line Usage

```bash
# Preprocess a single video
python -m pitch_detection.cli preprocess video.mp4 --output-dir ./data

# Preprocess multiple videos
python -m pitch_detection.cli preprocess video1.mp4 video2.mp4 --output-dir ./data

# With custom settings
python -m pitch_detection.cli preprocess video.mp4 \
    --output-dir ./data \
    --yolo-model yolov8l-pose.pt \
    --batch-size 128 \
    --pose-conf 0.3 \
    --sample-rate 1 \
    --device cuda
```

#### Python API Usage

```python
from pitch_detection import VideoPreprocessor

# Initialize preprocessor
preprocessor = VideoPreprocessor(
    yolo_model_path='yolov8l-pose.pt',
    device='cuda',
    feature_dim=400,
    batch_size=64
)

# Preprocess a single video
summary = preprocessor.preprocess_video(
    video_path='path/to/video.mp4',
    output_base_dir='./data',
    pose_conf_threshold=0.3,
    video_sample_rate=1
)

# Preprocess multiple videos
summaries = preprocessor.preprocess_videos(
    video_paths=['video1.mp4', 'video2.mp4'],
    output_base_dir='./data'
)
```

#### Individual Extractors

You can also use the pose and video extractors separately:

```python
from pitch_detection import YOLOPoseExtractor, VideoFeatureExtractor

# Extract only pose features
pose_extractor = YOLOPoseExtractor(model_path='yolov8l-pose.pt', device='cuda')
pose_data, metadata = pose_extractor.extract_from_video(
    video_path='video.mp4',
    output_dir='pose_features',
    conf_threshold=0.3
)

# Extract only video features  
video_extractor = VideoFeatureExtractor(device='cuda', feature_dim=400, batch_size=64)
features, metadata = video_extractor.extract_from_video(
    video_path='video.mp4',
    output_dir='video_features',
    sample_rate=1
)
```

### Training

```python
from pitch_detection import train_pitch_detector

model, history, checkpoint_path = train_pitch_detector(
    pose_files=['path/to/pose1.json'],
    video_files=['path/to/video1.npy'],
    label_files=['path/to/labels1.json'],
    num_epochs=50,
    batch_size=32,
    lr=0.001,
    checkpoint_dir='./checkpoints'
)
```

### Inference

```python
from pitch_detection import load_best_model, predict_pitch_timestamps, find_best_checkpoint

# Find and load best model
checkpoint_path = find_best_checkpoint('./checkpoints')
model, info = load_best_model(checkpoint_path)

# Make predictions
segments, predictions = predict_pitch_timestamps(
    model,
    pose_file='path/to/pose.json',
    video_file='path/to/video.npy',
    fps=30,
    threshold=0.5
)
```

### Evaluation

```python
from pitch_detection import evaluate_model_performance

results = evaluate_model_performance(
    model,
    pose_file='path/to/pose.json',
    video_file='path/to/video.npy',
    label_file='path/to/labels.json',
    fps=30
)

print(f"Precision: {results['metrics']['precision']:.3f}")
print(f"Recall: {results['metrics']['recall']:.3f}")
print(f"Mean IoU: {results['metrics']['mean_iou']:.3f}")
```

## Components

### Preprocessing (`preprocess.py`)
- `VideoPreprocessor`: Combined pose + video feature extraction
- `YOLOPoseExtractor`: Extract pose keypoints using YOLOv8-Pose
- `VideoFeatureExtractor`: Extract RGB features using ResNet50

### Model (`model.py`)
- `MultiModalBMN`: Multi-Modal Boundary Matching Network with pose and video encoders

### Dataset (`dataset.py`)
- `GPUSlidingWindowDataset`: Sliding window dataset for training
- `create_dataloader`: Optimized dataloader creation

### Loss Functions (`loss.py`)
- `segment_aware_loss`: Main combined loss function with precision focus
- `calculate_iou_loss`: Intersection over Union loss
- `calculate_dice_loss`: Dice coefficient loss
- `calculate_precision_loss`: Direct precision optimization
- `calculate_false_positive_penalty`: Explicit FP penalization
- `calculate_tversky_loss`: Weighted FP/FN loss

### Metrics (`metrics.py`)
- `calculate_segment_iou`: IoU for segments
- `extract_segments_from_predictions`: Convert frame predictions to segments
- `evaluate_pitch_detection`: Comprehensive evaluation metrics
- `analyze_dataset_distribution`: Dataset statistics

### Training (`trainer.py`)
- `train_pitch_detector`: Full training pipeline with checkpointing

### Inference (`inference.py`)
- `load_best_model`: Load model from checkpoint
- `find_best_checkpoint`: Find best checkpoint by IoU
- `predict_pitch_timestamps`: Make predictions on new data
- `evaluate_model_performance`: Evaluate on labeled data

## Features

- **Modular Design**: Easy to modify individual components
- **GPU Optimized**: Efficient training and inference
- **Precision Focused**: Multiple loss terms to minimize false positives
- **IoU Optimized**: Primary metric for segment quality
- **Checkpointing**: Automatic saving of best models
- **Comprehensive Metrics**: Precision, Recall, F1, IoU tracking
