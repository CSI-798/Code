"""
Configuration file for pitch detection framework.
Modify these parameters to customize training and inference.
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_CONFIG = {
    'pose_dim': 68,           # Dimension of pose features
    'video_dim': 400,         # Dimension of video features
    'temporal_scale': 100,    # Temporal scale for BMN
    'hidden_dim': 256,        # Hidden dimension size
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    'num_epochs': 50,         # Number of training epochs
    'batch_size': 32,         # Batch size
    'learning_rate': 0.001,   # Learning rate
    'weight_decay': 0.01,     # Weight decay for AdamW
    'lr_step_size': 3,        # LR scheduler step size
    'lr_gamma': 0.7,          # LR decay factor
    'gradient_clip': 1.0,     # Gradient clipping max norm
    'save_best': True,        # Save best model checkpoint
    'checkpoint_dir': './checkpoints',
}

# ============================================================================
# LOSS FUNCTION CONFIGURATION
# ============================================================================
LOSS_CONFIG = {
    'use_precision_focus': True,    # Enable precision-focused loss terms
    'fp_penalty_weight': 2.5,        # False positive penalty weight (higher = stricter)
    
    # Loss component weights (when precision_focus=True)
    'tversky_weight': 0.30,          # Tversky loss (IoU with FP emphasis)
    'precision_weight': 0.20,        # Direct precision optimization
    'fp_penalty_weight_loss': 0.20,  # Explicit FP penalization
    'focal_weight': 0.15,            # Focal loss for class imbalance
    'segment_weight': 0.10,          # Boundary detection
    'iou_weight': 0.05,              # Basic IoU
    
    # Tversky parameters
    'tversky_alpha': 0.7,            # FP weight (higher = penalize FP more)
    'tversky_beta': 0.3,             # FN weight
    
    # Focal loss parameters
    'focal_alpha': 0.75,             # Positive class weight
    'focal_gamma': 2.0,              # Focusing parameter
    
    # Boundary loss
    'boundary_pos_weight': 10.0,     # Positive weight for boundary detection
}

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    'window_size': 100,       # Sliding window size
    'stride': 50,             # Sliding window stride
    'fps': 30,                # Frames per second
    'num_workers': 2,         # DataLoader workers
    'pin_memory': True,       # Pin memory for GPU
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
INFERENCE_CONFIG = {
    'threshold': 0.5,                 # Detection threshold
    'min_pitch_duration': 0.3,        # Minimum pitch duration (seconds)
    'smoothing_window': 5,            # Smoothing window size
    'use_hysteresis': True,           # Use hysteresis thresholding
    'hysteresis_factor': 0.7,         # Low threshold = threshold * factor
    'adaptive_threshold': True,        # Use adaptive thresholding
    'min_confidence_ratio': 0.8,      # Min avg confidence relative to threshold
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVALUATION_CONFIG = {
    'iou_threshold': 0.3,     # IoU threshold for matching segments
    'eval_batch_size': 1,     # Batch size for evaluation
}

# ============================================================================
# FILE PATHS (Update these with your actual paths)
# ============================================================================
DATA_PATHS = {
    'pose_files': [
        'pose_features/tennessee_alabama_sr_gm1_pose_features.json',
        # Add more files...
    ],
    'video_files': [
        'video_features/tennessee_alabama_sr_gm1_video_features.npy',
        # Add more files...
    ],
    'label_files': [
        'features/tennessee_alabama_sr_gm1_20251208_211428_features.json',
        # Add more files...
    ],
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config():
    """Get model configuration dictionary"""
    return MODEL_CONFIG.copy()

def get_training_config():
    """Get training configuration dictionary"""
    return TRAINING_CONFIG.copy()

def get_loss_config():
    """Get loss configuration dictionary"""
    return LOSS_CONFIG.copy()

def get_dataset_config():
    """Get dataset configuration dictionary"""
    return DATASET_CONFIG.copy()

def get_inference_config():
    """Get inference configuration dictionary"""
    return INFERENCE_CONFIG.copy()

def get_evaluation_config():
    """Get evaluation configuration dictionary"""
    return EVALUATION_CONFIG.copy()

def get_all_configs():
    """Get all configurations as a single dictionary"""
    return {
        'model': get_model_config(),
        'training': get_training_config(),
        'loss': get_loss_config(),
        'dataset': get_dataset_config(),
        'inference': get_inference_config(),
        'evaluation': get_evaluation_config(),
        'device': str(DEVICE),
    }

def print_config():
    """Print all configuration settings"""
    print("=" * 70)
    print("PITCH DETECTION FRAMEWORK CONFIGURATION")
    print("=" * 70)
    
    print("\n📦 MODEL CONFIG:")
    for key, value in MODEL_CONFIG.items():
        print(f"   {key:20s} = {value}")
    
    print("\n🚀 TRAINING CONFIG:")
    for key, value in TRAINING_CONFIG.items():
        print(f"   {key:20s} = {value}")
    
    print("\n📉 LOSS CONFIG:")
    for key, value in LOSS_CONFIG.items():
        print(f"   {key:25s} = {value}")
    
    print("\n💾 DATASET CONFIG:")
    for key, value in DATASET_CONFIG.items():
        print(f"   {key:20s} = {value}")
    
    print("\n🔮 INFERENCE CONFIG:")
    for key, value in INFERENCE_CONFIG.items():
        print(f"   {key:25s} = {value}")
    
    print("\n📊 EVALUATION CONFIG:")
    for key, value in EVALUATION_CONFIG.items():
        print(f"   {key:20s} = {value}")
    
    print(f"\n💻 DEVICE: {DEVICE}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
