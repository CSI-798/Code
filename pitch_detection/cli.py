"""
Command-line interface for the pitch detection framework.
Uses Typer for a clean, user-friendly CLI experience.

Usage:
    python -m pitch_detection.cli train --help
    python -m pitch_detection.cli predict --help
    python -m pitch_detection.cli evaluate --help
    
    OR (when run directly from pitch_detection directory):
    python cli.py train --help
"""

import sys
from pathlib import Path

# Handle direct execution - add parent directory to path
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "pitch_detection"

import typer
from typing import List, Optional
import json
import torch

app = typer.Typer(
    name="pitch-detection",
    help="Pitch Detection Framework - Train and evaluate pitch detection models",
    add_completion=False
)


@app.command()
def train(
    pose_files: List[Path] = typer.Option(
        ..., 
        "--pose", "-p",
        help="Path to pose feature JSON files (can specify multiple)",
        exists=True
    ),
    video_files: List[Path] = typer.Option(
        ...,
        "--video", "-v",
        help="Path to video feature NPY files (can specify multiple)",
        exists=True
    ),
    label_files: List[Path] = typer.Option(
        ...,
        "--labels", "-l",
        help="Path to label JSON files (can specify multiple)",
        exists=True
    ),
    num_epochs: int = typer.Option(
        50,
        "--epochs", "-e",
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Batch size for training"
    ),
    learning_rate: float = typer.Option(
        0.001,
        "--lr",
        help="Learning rate"
    ),
    checkpoint_dir: Path = typer.Option(
        "./checkpoints",
        "--checkpoint-dir", "-c",
        help="Directory to save checkpoints"
    ),
    hidden_dim: int = typer.Option(
        256,
        "--hidden-dim",
        help="Hidden dimension for model"
    ),
    device: str = typer.Option(
        "auto",
        "--device", "-d",
        help="Device to use (auto, cuda, cpu)"
    ),
    augment: bool = typer.Option(
        False,
        "--augment/--no-augment",
        help="Enable on-the-fly training data augmentation"
    ),
    augment_factor: int = typer.Option(
        3,
        "--augment-factor",
        help="Dataset multiplier when augmentation is enabled (e.g., 3 = 3x windows)"
    ),
    temporal_shift: int = typer.Option(
        6,
        "--temporal-shift",
        help="Max frame shift for temporal jitter augmentation"
    ),
    pose_noise_std: float = typer.Option(
        0.01,
        "--pose-noise-std",
        help="Gaussian noise std applied to pose features during augmentation"
    ),
    video_noise_std: float = typer.Option(
        0.005,
        "--video-noise-std",
        help="Gaussian noise std applied to video features during augmentation"
    ),
    feature_dropout_prob: float = typer.Option(
        0.1,
        "--feature-dropout-prob",
        help="Probability of dropping full timesteps in feature sequences during augmentation"
    ),
    val_ratio: float = typer.Option(
        0.33,
        "--val-ratio",
        help="Fraction of input files used for validation checkpoint selection"
    ),
    hard_negative_ratio: float = typer.Option(
        3.0,
        "--hard-negative-ratio",
        help="Hard negative mining ratio (negatives per positive frame)"
    ),
):
    """
    Train a pitch detection model.
    
    Example:
        python -m pitch_detection.cli train \\
            --pose pose_features/game1.json \\
            --video video_features/game1.npy \\
            --labels features/game1_labels.json \\
            --epochs 50 --batch-size 32
    """
    from .trainer import train_pitch_detector
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    typer.echo(f"🚀 Starting training on {device}")
    typer.echo(f"📊 Training configuration:")
    typer.echo(f"   Pose files: {len(pose_files)}")
    typer.echo(f"   Video files: {len(video_files)}")
    typer.echo(f"   Label files: {len(label_files)}")
    typer.echo(f"   Epochs: {num_epochs}")
    typer.echo(f"   Batch size: {batch_size}")
    typer.echo(f"   Learning rate: {learning_rate}")
    typer.echo(f"   Checkpoint dir: {checkpoint_dir}")
    typer.echo(f"   Augmentation: {augment}")
    if augment:
        typer.echo(f"   Augment factor: {augment_factor}")
        typer.echo(f"   Temporal shift: {temporal_shift}")
        typer.echo(f"   Pose noise std: {pose_noise_std}")
        typer.echo(f"   Video noise std: {video_noise_std}")
        typer.echo(f"   Feature dropout prob: {feature_dropout_prob}")
    typer.echo(f"   Validation ratio: {val_ratio}")
    typer.echo(f"   Hard negative ratio: {hard_negative_ratio}")
    typer.echo("")
    
    # Convert paths to strings
    pose_files_str = [str(p) for p in pose_files]
    video_files_str = [str(p) for p in video_files]
    label_files_str = [str(p) for p in label_files]
    
    try:
        model, history, checkpoint_path = train_pitch_detector(
            pose_files=pose_files_str,
            video_files=video_files_str,
            label_files=label_files_str,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=learning_rate,
            save_best=True,
            checkpoint_dir=str(checkpoint_dir),
            device=device,
            use_augmentation=augment,
            augment_factor=augment_factor,
            temporal_shift=temporal_shift,
            pose_noise_std=pose_noise_std,
            video_noise_std=video_noise_std,
            feature_dropout_prob=feature_dropout_prob,
            val_ratio=val_ratio,
            hard_negative_ratio=hard_negative_ratio,
        )
        
        typer.echo("")
        typer.echo("✅ Training completed successfully!")
        typer.echo(f"📊 Final metrics:")
        typer.echo(f"   Mean IoU: {history['mean_iou'][-1]:.4f}")
        typer.echo(f"   Precision: {history['precision'][-1]:.4f}")
        typer.echo(f"   Recall: {history['recall'][-1]:.4f}")
        typer.echo(f"   F1 Score: {history['f1'][-1]:.4f}")
        
        if checkpoint_path:
            typer.echo(f"💾 Best model saved to: {checkpoint_path}")
        
    except Exception as e:
        typer.echo(f"❌ Training failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def predict(
    pose_file: Path = typer.Option(
        ...,
        "--pose", "-p",
        help="Path to pose feature JSON file",
        exists=True
    ),
    video_file: Path = typer.Option(
        ...,
        "--video", "-v",
        help="Path to video feature NPY file",
        exists=True
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint", "-c",
        help="Path to model checkpoint (or auto-find best)",
        exists=True
    ),
    checkpoint_dir: Path = typer.Option(
        "./checkpoints",
        "--checkpoint-dir",
        help="Directory to search for best checkpoint if --checkpoint not provided"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save predictions JSON file"
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold", "-t",
        help="Optional threshold override (default: use calibrated checkpoint setting)"
    ),
    fps: int = typer.Option(
        30,
        "--fps",
        help="Frames per second"
    ),
    device: str = typer.Option(
        "auto",
        "--device", "-d",
        help="Device to use (auto, cuda, cpu)"
    ),
):
    """
    Make predictions on a video using a trained model.
    
    Example:
        python -m pitch_detection.cli predict \\
            --pose pose_features/game1.json \\
            --video video_features/game1.npy \\
            --output predictions.json
    """
    from .inference import load_best_model, find_best_checkpoint, predict_pitch_timestamps
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find checkpoint if not provided
    if checkpoint is None:
        typer.echo(f"🔍 Searching for best checkpoint in {checkpoint_dir}")
        checkpoint = find_best_checkpoint(str(checkpoint_dir))
        if checkpoint is None:
            typer.echo("❌ No checkpoint found. Please train a model first or specify --checkpoint", err=True)
            raise typer.Exit(code=1)
    
    typer.echo(f"📂 Loading model from: {checkpoint}")
    
    try:
        # Load model
        model, info = load_best_model(str(checkpoint), device=device)
        
        typer.echo(f"🔮 Making predictions on {pose_file.name}")
        
        # Make predictions
        segments, predictions = predict_pitch_timestamps(
            model=model,
            pose_file=str(pose_file),
            video_file=str(video_file),
            fps=fps,
            threshold=threshold,
            device=device,
            inference_config=info.get('inference_config'),
        )
        
        typer.echo("")
        typer.echo(f"✅ Predictions completed!")
        typer.echo(f"📊 Found {len(segments)} pitch segments:")
        
        for i, seg in enumerate(segments, 1):
            typer.echo(
                f"   Pitch {i}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s "
                f"({seg['duration']:.2f}s, confidence: {seg['confidence']:.3f})"
            )
        
        # Save predictions if output path provided
        if output:
            output_data = {
                'model_checkpoint': str(checkpoint),
                'model_info': {
                    'best_iou': info['best_iou'],
                    'epoch': info['epoch']
                },
                'input_files': {
                    'pose_file': str(pose_file),
                    'video_file': str(video_file)
                },
                'parameters': {
                    'threshold': threshold,
                    'inference_config': info.get('inference_config'),
                    'fps': fps
                },
                'segments': segments,
                'num_segments': len(segments)
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            typer.echo(f"💾 Predictions saved to: {output}")
        
    except Exception as e:
        typer.echo(f"❌ Prediction failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    pose_file: Path = typer.Option(
        ...,
        "--pose", "-p",
        help="Path to pose feature JSON file",
        exists=True
    ),
    video_file: Path = typer.Option(
        ...,
        "--video", "-v",
        help="Path to video feature NPY file",
        exists=True
    ),
    label_file: Path = typer.Option(
        ...,
        "--labels", "-l",
        help="Path to ground truth label JSON file",
        exists=True
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint", "-c",
        help="Path to model checkpoint (or auto-find best)",
        exists=True
    ),
    checkpoint_dir: Path = typer.Option(
        "./checkpoints",
        "--checkpoint-dir",
        help="Directory to search for best checkpoint if --checkpoint not provided"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save evaluation results JSON file"
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold", "-t",
        help="Optional threshold override (default: use calibrated checkpoint setting)"
    ),
    fps: int = typer.Option(
        30,
        "--fps",
        help="Frames per second"
    ),
    device: str = typer.Option(
        "auto",
        "--device", "-d",
        help="Device to use (auto, cuda, cpu)"
    ),
):
    """
    Evaluate a trained model on labeled data.
    
    Example:
        python -m pitch_detection.cli evaluate \\
            --pose pose_features/game1.json \\
            --video video_features/game1.npy \\
            --labels features/game1_labels.json \\
            --output evaluation.json
    """
    from .inference import load_best_model, find_best_checkpoint, evaluate_model_performance
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find checkpoint if not provided
    if checkpoint is None:
        typer.echo(f"🔍 Searching for best checkpoint in {checkpoint_dir}")
        checkpoint = find_best_checkpoint(str(checkpoint_dir))
        if checkpoint is None:
            typer.echo("❌ No checkpoint found. Please train a model first or specify --checkpoint", err=True)
            raise typer.Exit(code=1)
    
    typer.echo(f"📂 Loading model from: {checkpoint}")
    
    try:
        # Load model
        model, info = load_best_model(str(checkpoint), device=device)
        
        typer.echo(f"📊 Evaluating on {pose_file.name}")
        
        # Evaluate
        results = evaluate_model_performance(
            model=model,
            pose_file=str(pose_file),
            video_file=str(video_file),
            label_file=str(label_file),
            fps=fps,
            threshold=threshold,
            device=device,
            inference_config=info.get('inference_config'),
        )
        
        typer.echo("")
        typer.echo("✅ Evaluation completed!")
        typer.echo(f"📊 Performance Metrics:")
        typer.echo(f"   Precision:  {results['metrics']['precision']:.3f}")
        typer.echo(f"   Recall:     {results['metrics']['recall']:.3f}")
        typer.echo(f"   F1 Score:   {results['metrics']['f1']:.3f}")
        typer.echo(f"   Mean IoU:   {results['metrics']['mean_iou']:.3f}")
        typer.echo(f"   Matched:    {results['metrics']['matched_segments']}/{results['metrics']['total_true']}")
        
        if results['timing_errors']:
            avg_start_error = sum(e['start_error'] for e in results['timing_errors']) / len(results['timing_errors'])
            avg_end_error = sum(e['end_error'] for e in results['timing_errors']) / len(results['timing_errors'])
            typer.echo(f"\n⏱️  Timing Accuracy:")
            typer.echo(f"   Avg Start Error: {avg_start_error:.2f}s")
            typer.echo(f"   Avg End Error:   {avg_end_error:.2f}s")
        
        # Save results if output path provided
        if output:
            output_data = {
                'model_checkpoint': str(checkpoint),
                'model_info': {
                    'best_iou': info['best_iou'],
                    'epoch': info['epoch']
                },
                'input_files': {
                    'pose_file': str(pose_file),
                    'video_file': str(video_file),
                    'label_file': str(label_file)
                },
                'parameters': {
                    'threshold': threshold,
                    'fps': fps
                },
                'metrics': results['metrics'],
                'timing_errors': results['timing_errors'],
                'num_predicted_segments': len(results['segments']),
                'predicted_segments': results['segments']
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            typer.echo(f"💾 Results saved to: {output}")
        
    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def find_checkpoint(
    checkpoint_dir: Path = typer.Option(
        "./checkpoints",
        "--checkpoint-dir", "-c",
        help="Directory to search for checkpoints"
    ),
):
    """
    Find the best checkpoint in a directory.
    
    Example:
        python -m pitch_detection.cli find-checkpoint --checkpoint-dir ./checkpoints
    """
    from .inference import find_best_checkpoint
    
    typer.echo(f"🔍 Searching for best checkpoint in {checkpoint_dir}")
    
    try:
        best_checkpoint = find_best_checkpoint(str(checkpoint_dir))
        
        if best_checkpoint:
            typer.echo(f"✅ Found best checkpoint: {best_checkpoint}")
        else:
            typer.echo("❌ No checkpoints found in directory", err=True)
            raise typer.Exit(code=1)
            
    except Exception as e:
        typer.echo(f"❌ Search failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def config(
    show: bool = typer.Option(
        True,
        "--show/--no-show",
        help="Show current configuration"
    ),
    export: Optional[Path] = typer.Option(
        None,
        "--export", "-e",
        help="Export configuration to JSON file"
    ),
):
    """
    View or export configuration settings.
    
    Example:
        python -m pitch_detection.cli config
        python -m pitch_detection.cli config --export config.json
    """
    from .config import print_config, get_all_configs
    
    if show:
        print_config()
    
    if export:
        try:
            configs = get_all_configs()
            with open(export, 'w') as f:
                json.dump(configs, f, indent=2)
            typer.echo(f"\n💾 Configuration exported to: {export}")
        except Exception as e:
            typer.echo(f"❌ Export failed: {e}", err=True)
            raise typer.Exit(code=1)


@app.command()
def info(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint", "-c",
        help="Path to model checkpoint",
        exists=True
    ),
):
    """
    Display information about a checkpoint.
    
    Example:
        python -m pitch_detection.cli info --checkpoint checkpoints/best_model.pth
    """
    typer.echo(f"📂 Loading checkpoint info from: {checkpoint}")
    
    try:
        checkpoint_data = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        
        typer.echo("")
        typer.echo("📊 Checkpoint Information:")
        typer.echo("=" * 60)
        typer.echo(f"Epoch:              {checkpoint_data['epoch']}")
        typer.echo(f"Best IoU:           {checkpoint_data['best_iou']:.4f}")
        
        if 'hyperparameters' in checkpoint_data:
            typer.echo("\n🔧 Hyperparameters:")
            for key, value in checkpoint_data['hyperparameters'].items():
                typer.echo(f"  {key:20s} = {value}")
        
        if 'history' in checkpoint_data:
            history = checkpoint_data['history']
            typer.echo("\n📈 Final Metrics:")
            if history['mean_iou']:
                typer.echo(f"  Mean IoU:     {history['mean_iou'][-1]:.4f}")
            if history['precision']:
                typer.echo(f"  Precision:    {history['precision'][-1]:.4f}")
            if history['recall']:
                typer.echo(f"  Recall:       {history['recall'][-1]:.4f}")
            if history['f1']:
                typer.echo(f"  F1 Score:     {history['f1'][-1]:.4f}")
        
        if 'dataset_stats' in checkpoint_data:
            stats = checkpoint_data['dataset_stats']
            typer.echo("\n💾 Dataset Statistics:")
            typer.echo(f"  Total frames:     {stats.get('total_frames', 'N/A'):,}")
            typer.echo(f"  Pitch frames:     {stats.get('pitch_frames', 'N/A'):,}")
            typer.echo(f"  Pitch ratio:      {stats.get('pitch_ratio', 0):.1%}")
        
        typer.echo("=" * 60)
        
    except Exception as e:
        typer.echo(f"❌ Failed to load checkpoint info: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def version():
    """
    Display version information.
    """
    try:
        from . import __version__
    except (ImportError, ValueError):
        # Fallback when running directly
        import pitch_detection
        __version__ = pitch_detection.__version__
    
    typer.echo(f"Pitch Detection Framework v{__version__}")
    typer.echo(f"PyTorch version: {torch.__version__}")
    typer.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"CUDA version: {torch.version.cuda}")
        typer.echo(f"GPU: {torch.cuda.get_device_name(0)}")


@app.command()
def preprocess(
    videos: List[Path] = typer.Argument(
        ...,
        help="Path(s) to video file(s) to preprocess",
        exists=True
    ),
    output_dir: Path = typer.Option(
        ".",
        "--output-dir", "-o",
        help="Base output directory for preprocessed features"
    ),
    yolo_model: Path = typer.Option(
        "yolov8l-pose.pt",
        "--yolo-model", "-y",
        help="Path to YOLO pose model weights"
    ),
    feature_dim: int = typer.Option(
        400,
        "--feature-dim", "-f",
        help="Dimension of motion features from image subtraction"
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size", "-b",
        help="Batch size for both pose and motion feature extraction"
    ),
    pose_conf: float = typer.Option(
        0.3,
        "--pose-conf", "-c",
        help="Confidence threshold for pose detection"
    ),
    sample_rate: int = typer.Option(
        1,
        "--sample-rate", "-s",
        help="Sample every N frames for video features (1 = every frame)"
    ),
    device: str = typer.Option(
        "auto",
        "--device", "-d",
        help="Device to use (auto, cuda, cpu)"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output"
    ),
):
    """
    Preprocess videos to extract pose and video features.
    
    This command extracts:
    1. Pose features using YOLOv8-Pose (saved as JSON)
    2. Motion features using frame differencing/image subtraction (saved as NPY)
    
    Example:
        python -m pitch_detection.cli preprocess video1.mp4 video2.mp4 \\
            --output-dir ./data \\
            --batch-size 128 \\
            --device cuda
    """
    from .preprocess import VideoPreprocessor
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not quiet:
        typer.echo(f"🎬 Video Preprocessing Pipeline")
        typer.echo(f"{'='*80}")
        typer.echo(f"Videos to process: {len(videos)}")
        typer.echo(f"Output directory: {output_dir}")
        typer.echo(f"Device: {device}")
        typer.echo(f"YOLO model: {yolo_model}")
        typer.echo(f"Feature dimension: {feature_dim}")
        typer.echo(f"Batch size (pose + motion): {batch_size}")
        typer.echo(f"Pose confidence: {pose_conf}")
        typer.echo(f"Sample rate: {sample_rate}")
        typer.echo(f"{'='*80}\n")
    
    try:
        # Initialize preprocessor
        preprocessor = VideoPreprocessor(
            yolo_model_path=str(yolo_model),
            device=device,
            feature_dim=feature_dim,
            batch_size=batch_size
        )
        
        # Process videos
        summaries = preprocessor.preprocess_videos(
            video_paths=[str(v) for v in videos],
            output_base_dir=str(output_dir),
            pose_conf_threshold=pose_conf,
            video_sample_rate=sample_rate,
            verbose=not quiet
        )
        
        if not quiet:
            typer.echo(f"\n{'='*80}")
            typer.echo("✅ All videos preprocessed successfully!")
            typer.echo(f"{'='*80}\n")
            
            typer.echo("📊 Summary:")
            for i, summary in enumerate(summaries, 1):
                typer.echo(f"\n{i}. {summary['video_name']}")
                typer.echo(f"   Pose features: {summary['pose_features']['output_path']}")
                typer.echo(f"   Video features: {summary['video_features']['output_path']}")
                typer.echo(f"   Frames processed: {summary['pose_features']['total_frames']:,}")
                typer.echo(f"   Frames with poses: {summary['pose_features']['frames_with_poses']:,}")
        
        # Save overall summary
        summary_path = Path(output_dir) / 'preprocessing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summaries, f, indent=2)
        
        if not quiet:
            typer.echo(f"\n💾 Summary saved to: {summary_path}")
            
    except FileNotFoundError as e:
        typer.echo(f"❌ File not found: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Preprocessing failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("generate-clips")
def generate_clips(
    predictions: Path = typer.Option(
        ...,
        "--predictions", "-p",
        help="Path to predictions JSON file from the predict command",
        exists=True,
    ),
    pose_file: Path = typer.Option(
        ...,
        "--pose", "-s",
        help="Path to pose features JSON file",
        exists=True,
    ),
    video_file: Path = typer.Option(
        ...,
        "--video", "-v",
        help="Path to original source video file",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        "./pitch_clips_overlay",
        "--output-dir", "-o",
        help="Directory where overlay clips will be written",
    ),
    padding_seconds: float = typer.Option(
        2.0,
        "--padding-seconds",
        help="Seconds of context before and after each predicted pitch segment",
    ),
    max_clips: Optional[int] = typer.Option(
        None,
        "--max-clips",
        help="Maximum number of clips to generate",
    ),
    pose_confidence_threshold: float = typer.Option(
        0.25,
        "--pose-confidence-threshold",
        help="Minimum pose joint confidence for drawing skeleton edges/points",
    ),
    overlay: bool = typer.Option(
        True,
        "--overlay/--no-overlay",
        help="Enable or disable pose/pitch overlays in exported clips.",
    ),
):
    """
    Generate pitch clips from prediction segments with optional overlays.

    Example:
        python -m pitch_detection.cli generate-clips \
            --predictions ./alabama_virginiatech_gm1_predictions.json \
            --pose ./pose_features/alabama_virginiatech_gm1_pose_features.json \
            --video ../Data/training/alabama_virginiatech_gm1.mp4 \
            --output-dir ./pitch_clips_overlay
    """
    from .clips import generate_overlay_clips

    typer.echo("🎬 Generating pitch clips")
    typer.echo(f"   Predictions: {predictions}")
    typer.echo(f"   Pose file: {pose_file}")
    typer.echo(f"   Video file: {video_file}")
    typer.echo(f"   Output dir: {output_dir}")
    typer.echo(f"   Overlay: {'on' if overlay else 'off'}")
    if max_clips is not None:
        typer.echo(f"   Max clips: {max_clips}")

    try:
        output_files = generate_overlay_clips(
            predictions_file=str(predictions),
            pose_file=str(pose_file),
            video_file=str(video_file),
            output_dir=str(output_dir),
            padding_seconds=padding_seconds,
            max_clips=max_clips,
            confidence_threshold=pose_confidence_threshold,
            overlay=overlay,
        )

        typer.echo("")
        typer.echo(f"✅ Generated {len(output_files)} clip(s)")
        for idx, clip_path in enumerate(output_files, 1):
            typer.echo(f"   {idx}. {clip_path}")
    except Exception as e:
        typer.echo(f"❌ Clip generation failed: {e}", err=True)
        raise typer.Exit(code=1)


def main():
    """Entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()
