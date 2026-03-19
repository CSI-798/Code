"""
Pitch Detection Framework
A modular framework for detecting pitch events in baseball videos using pose and video features.
"""

from .model import MultiModalBMN
from .dataset import GPUSlidingWindowDataset, create_dataloader
from .loss import (
    segment_aware_loss,
    calculate_iou_loss,
    calculate_dice_loss,
    calculate_precision_loss,
    calculate_false_positive_penalty,
    calculate_tversky_loss
)
from .metrics import (
    calculate_segment_iou,
    extract_segments_from_predictions,
    evaluate_pitch_detection,
    analyze_dataset_distribution
)
from .trainer import train_pitch_detector
from .inference import (
    predict_pitch_timestamps,
    load_best_model,
    find_best_checkpoint,
    evaluate_model_performance
)
from .preprocess import (
    YOLOPoseExtractor,
    VideoFeatureExtractor,
    VideoPreprocessor
)
from .config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    LOSS_CONFIG,
    DATASET_CONFIG,
    INFERENCE_CONFIG,
    EVALUATION_CONFIG,
    DEVICE,
    get_all_configs,
    print_config
)

# CLI is available via: python -m pitch_detection.cli
from . import cli

__version__ = "0.1.0"

__all__ = [
    # Model
    "MultiModalBMN",
    
    # Dataset
    "GPUSlidingWindowDataset",
    "create_dataloader",
    
    # Loss functions
    "segment_aware_loss",
    "calculate_iou_loss",
    "calculate_dice_loss",
    "calculate_precision_loss",
    "calculate_false_positive_penalty",
    "calculate_tversky_loss",
    
    # Metrics
    "calculate_segment_iou",
    "extract_segments_from_predictions",
    "evaluate_pitch_detection",
    "analyze_dataset_distribution",
    
    # Training
    "train_pitch_detector",
    
    # Inference
    "predict_pitch_timestamps",
    "load_best_model",
    "find_best_checkpoint",
    "evaluate_model_performance",
    
    # Preprocessing
    "YOLOPoseExtractor",
    "VideoFeatureExtractor",
    "VideoPreprocessor",
    
    # Configuration
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "LOSS_CONFIG",
    "DATASET_CONFIG",
    "INFERENCE_CONFIG",
    "EVALUATION_CONFIG",
    "DEVICE",
    "get_all_configs",
    "print_config",
]
