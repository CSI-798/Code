"""
Training utilities for pitch detection model.
"""

import os
import datetime
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler, Subset

from .model import MultiModalBMN
from .dataset import GPUSlidingWindowDataset, create_dataloader
from .loss import segment_aware_loss, calculate_iou_loss
from .metrics import (
    extract_segments_from_predictions,
    extract_segments_strict,
    evaluate_pitch_detection,
    analyze_dataset_distribution
)
from .inference import calibrate_inference_config


def train_pitch_detector(
    pose_files,
    video_files,
    label_files,
    num_epochs=10,
    batch_size=32,
    window_size=100,
    stride=50,
    lr=0.001,
    save_best=True,
    checkpoint_dir="./checkpoints",
    device='cuda',
    use_augmentation=False,
    augment_factor=1,
    temporal_shift=0,
    pose_noise_std=0.0,
    video_noise_std=0.0,
    feature_dropout_prob=0.0,
    val_ratio=0.33,
    hard_negative_ratio=3.0,
    iou_thresholds=None,
    balance_windows=True,
    positive_window_target=0.5,
):
    """
    Train the pitch detection model with IoU-optimized loss function and best model saving.
    
    Args:
        pose_files: List of paths to pose feature JSON files
        video_files: List of paths to video feature .npy files
        label_files: List of paths to label JSON files
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        save_best: Whether to save the best model checkpoint
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        model: Trained model
        history: Training history dictionary
        best_model_path: Path to the best saved model checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    if save_best:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"📁 Checkpoint directory: {checkpoint_dir}")

    if not iou_thresholds:
        iou_thresholds = [0.3]
    iou_thresholds = [float(t) for t in iou_thresholds]
    
    num_files = min(len(pose_files), len(video_files), len(label_files))
    if num_files <= 0:
        raise ValueError("No training files provided")

    pose_files = pose_files[:num_files]
    video_files = video_files[:num_files]
    label_files = label_files[:num_files]

    all_pose_files = pose_files
    all_video_files = video_files
    all_label_files = label_files

    # Build base dataset for window-level split across all games
    base_dataset = GPUSlidingWindowDataset(
        all_pose_files,
        all_video_files,
        all_label_files,
        window_size=window_size,
        stride=stride,
        augment=False,
    )

    total_base_windows = len(base_dataset)
    has_validation = total_base_windows >= 2 and val_ratio > 0
    val_count = int(round(total_base_windows * float(val_ratio))) if has_validation else 0
    if has_validation:
        val_count = max(1, min(total_base_windows - 1, val_count))

    indices = np.arange(total_base_windows)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    if val_count > 0:
        val_base_indices = indices[:val_count].tolist()
        train_base_indices = indices[val_count:].tolist()
    else:
        val_base_indices = []
        train_base_indices = indices.tolist()

    print(
        f"📚 Window split across all games: train={len(train_base_indices)} "
        f"| val={len(val_base_indices)} (val_ratio={val_ratio:.2f})"
    )

    # Create train dataset (with optional augmentation)
    train_dataset = GPUSlidingWindowDataset(
        all_pose_files,
        all_video_files,
        all_label_files,
        window_size=window_size,
        stride=stride,
        augment=use_augmentation,
        augment_factor=augment_factor,
        temporal_shift=temporal_shift,
        pose_noise_std=pose_noise_std,
        video_noise_std=video_noise_std,
        feature_dropout_prob=feature_dropout_prob,
    )

    train_subset_indices = []
    base_window_count = len(train_dataset.windows)
    replica_count = train_dataset.augment_factor if train_dataset.augment else 1
    for replica in range(replica_count):
        offset = replica * base_window_count if train_dataset.augment else 0
        train_subset_indices.extend([offset + idx for idx in train_base_indices])

    train_data = Subset(train_dataset, train_subset_indices)

    train_sampler = None
    if balance_windows:
        window_flags = train_dataset.get_window_positive_flags()
        if window_flags and train_base_indices:
            selected_flags = [window_flags[idx] for idx in train_base_indices]
            positive_count = sum(1 for flag in selected_flags if flag)
            negative_count = len(selected_flags) - positive_count
            target = min(0.95, max(0.05, float(positive_window_target)))

            if positive_count > 0 and negative_count > 0:
                pos_weight_window = target / positive_count
                neg_weight_window = (1.0 - target) / negative_count
                base_weights = [pos_weight_window if flag else neg_weight_window for flag in selected_flags]

                if train_dataset.augment and train_dataset.augment_factor > 1:
                    sample_weights = base_weights * train_dataset.augment_factor
                else:
                    sample_weights = base_weights

                train_sampler = WeightedRandomSampler(
                    weights=torch.DoubleTensor(sample_weights),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                print(
                    f"🎚️ Balanced window sampling enabled: target positive ratio={target:.2f}, "
                    f"base positive windows={positive_count}/{len(selected_flags)}"
                )
            else:
                print("⚠️ Could not enable balanced sampling (missing positive or negative windows)")

    dataloader = create_dataloader(
        train_data,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )

    val_dataset = None
    val_dataloader = None
    if len(val_base_indices) > 0:
        val_dataset = GPUSlidingWindowDataset(
            all_pose_files,
            all_video_files,
            all_label_files,
            window_size=window_size,
            stride=stride,
            augment=False,
        )
        val_data = Subset(val_dataset, val_base_indices)
        val_dataloader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)

    if use_augmentation:
        print("🧪 Data augmentation enabled")
        print(f"   Augment factor: {augment_factor}")
        print(f"   Temporal shift: {temporal_shift}")
        print(f"   Pose noise std: {pose_noise_std}")
        print(f"   Video noise std: {video_noise_std}")
        print(f"   Feature dropout: {feature_dropout_prob}")
    
    # Analyze dataset distribution before training
    dataset_stats = analyze_dataset_distribution(Subset(base_dataset, train_base_indices))
    
    # Enhanced class weight calculation for better IoU performance
    pitch_ratio = dataset_stats['pitch_ratio']
    if pitch_ratio > 0:
        # More aggressive weighting for severe imbalance
        if pitch_ratio < 0.05:  # <5% pitch frames
            pos_weight = min(50.0, (1 - pitch_ratio) / pitch_ratio)  # Cap very high weights
            print(f"   SEVERE imbalance: {pitch_ratio:.1%} pitch frames")
        elif pitch_ratio < 0.15:  # <15% pitch frames  
            pos_weight = min(20.0, (1 - pitch_ratio) / pitch_ratio)
            print(f"   HIGH imbalance: {pitch_ratio:.1%} pitch frames")
        else:
            pos_weight = (1 - pitch_ratio) / pitch_ratio
            print(f"   Moderate imbalance: {pitch_ratio:.1%} pitch frames")
        
        print(f"   Using enhanced positive class weight: {pos_weight:.2f}")
        print(f"   Expected to improve IoU by better minority class learning")
    else:
        pos_weight = 1.0
        print("   No pitch frames found in dataset!")
        return None, None, None
    
    # Initialize fresh model with correct dimensions
    torch.cuda.empty_cache()  # Clear GPU memory
    
    # Create new model instance
    model = MultiModalBMN(hidden_dim=256).to(device)
    print(f"   Model initialized with hidden_dim=256")
    
    # Use different optimizers and loss functions for better learning
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)  # More aggressive LR decay
    
    # Training history with IoU tracking
    history = {
        'loss': [], 'iou_loss': [], 'precision': [], 'recall': [], 'f1': [], 'mean_iou': [],
        'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_mean_iou': []
    }
    
    print(f"🚀 Starting IoU-optimized training with {num_epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Window size: {window_size}")
    print(f"   Stride: {stride}")
    print(f"   Learning rate: {lr}")
    print(f"   Total train windows: {len(train_data)}")
    print(f"   Focus: Maximizing mean IoU performance")
    print(f"   Eval IoU thresholds: {', '.join(f'{t:.2f}' for t in iou_thresholds)}")

    eval_threshold = 0.20
    eval_min_duration = max(6, int(0.25 * train_dataset.fps))
    eval_max_duration = max(20, int(2.5 * train_dataset.fps))
    eval_merge_gap = max(0, int(0.05 * train_dataset.fps))
    print(
        f"   Eval segment decoding: thr={eval_threshold:.2f}, min={eval_min_duration}f, "
        f"max={eval_max_duration}f, gap={eval_merge_gap}f"
    )
    
    # Training loop with IoU-focused loss
    model.train()
    best_iou = 0.0
    best_score = -1.0
    best_model_path = None
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_iou_loss = 0
        all_predictions = []
        all_true_segments = []
        calibration_prediction_sequences = []
        calibration_true_segments = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to GPU
            pose_features = batch['pose_features'].to(device)
            video_features = batch['video_features'].to(device)
            frame_labels = batch['frame_labels'].to(device)
            start_labels = batch['start_boundary_labels'].to(device)
            end_labels = batch['end_boundary_labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(pose_features, video_features)
            
            # Use logits directly from model output for loss calculation
            frame_logits = outputs['frame_logits']
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']
            
            # Use IoU-optimized loss function
            total_loss = segment_aware_loss(
                frame_logits, frame_labels,
                start_logits, start_labels,
                end_logits, end_labels,
                pos_weight=pos_weight,
                hard_negative_ratio=hard_negative_ratio,
            )
            
            # Calculate pure IoU loss for tracking
            batch_iou_loss = 0
            for i in range(frame_logits.size(0)):
                iou_loss = calculate_iou_loss(torch.sigmoid(frame_logits[i]), frame_labels[i])
                batch_iou_loss += iou_loss
            batch_iou_loss /= frame_logits.size(0)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_iou_loss += batch_iou_loss.item()
            
            # Collect predictions for IoU evaluation
            if len(all_predictions) < 100:  # More samples for better evaluation
                with torch.no_grad():
                    batch_preds = torch.sigmoid(frame_logits).cpu().numpy()
                    batch_labels = frame_labels.cpu().numpy()

                    if len(calibration_prediction_sequences) < 150:
                        for i in range(min(len(batch_preds), 6)):
                            calibration_prediction_sequences.append(batch_preds[i].copy())
                            calibration_true_segments.append(
                                extract_segments_from_predictions(batch_labels[i], threshold=0.5, min_duration=1)
                            )

                    # Stable, strict thresholding for training metrics
                    for i in range(min(5, len(batch_preds))):  # More samples for better stats
                        pred_segments = extract_segments_strict(
                            batch_preds[i],
                            threshold=eval_threshold,
                            min_duration=eval_min_duration,
                            max_duration=eval_max_duration,
                            merge_gap=eval_merge_gap,
                        )
                        true_segments = extract_segments_from_predictions(
                            batch_labels[i], threshold=0.5, min_duration=1
                        )

                        all_predictions.append(pred_segments)
                        all_true_segments.append(true_segments)
            
            # Show prediction statistics with IoU focus
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    pred_probs = torch.sigmoid(frame_logits)
                    mean_pred = pred_probs.mean().item()
                    max_pred = pred_probs.max().item()
                    min_pred = pred_probs.min().item()
                    progress_bar.set_postfix({
                        'Loss': f"{total_loss.item():.4f}",
                        'IoU_Loss': f"{batch_iou_loss.item():.4f}",
                        'Pred': f"{mean_pred:.3f}({min_pred:.3f}-{max_pred:.3f})"
                    })
        
        # Calculate epoch metrics with focus on IoU
        avg_loss = epoch_loss / len(dataloader)
        avg_iou_loss = epoch_iou_loss / len(dataloader)
        
        # Evaluate with IoU-optimized criteria
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_iou = 0
        valid_samples = 0
        
        for pred_segs, true_segs in zip(all_predictions, all_true_segments):
            threshold_metrics = [
                evaluate_pitch_detection(pred_segs, true_segs, iou_threshold=t)
                for t in iou_thresholds
            ]
            metrics = {
                'precision': float(np.mean([m['precision'] for m in threshold_metrics])),
                'recall': float(np.mean([m['recall'] for m in threshold_metrics])),
                'f1': float(np.mean([m['f1'] for m in threshold_metrics])),
                'mean_iou': float(np.mean([m['mean_iou'] for m in threshold_metrics])),
            }
            if len(pred_segs) > 0 or len(true_segs) > 0:
                total_precision += metrics['precision']
                total_recall += metrics['recall']
                total_f1 += metrics['f1']
                total_iou += metrics['mean_iou']
                valid_samples += 1
        
        if valid_samples > 0:
            avg_precision = total_precision / valid_samples
            avg_recall = total_recall / valid_samples
            avg_f1 = total_f1 / valid_samples
            avg_mean_iou = total_iou / valid_samples
        else:
            avg_precision = avg_recall = avg_f1 = avg_mean_iou = 0.0
        
        # Store metrics with IoU tracking
        history['loss'].append(avg_loss)
        history['iou_loss'].append(avg_iou_loss)
        history['precision'].append(avg_precision)
        history['recall'].append(avg_recall)
        history['f1'].append(avg_f1)
        history['mean_iou'].append(avg_mean_iou)
        
        # Validation pass (used for checkpoint selection when available)
        val_loss = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        val_mean_iou = 0.0
        val_calibration_prediction_sequences = []
        val_calibration_true_segments = []

        if val_dataloader is not None:
            model.eval()
            val_predictions = []
            val_true_segments = []

            with torch.no_grad():
                for val_batch in val_dataloader:
                    pose_features = val_batch['pose_features'].to(device)
                    video_features = val_batch['video_features'].to(device)
                    frame_labels = val_batch['frame_labels'].to(device)
                    start_labels = val_batch['start_boundary_labels'].to(device)
                    end_labels = val_batch['end_boundary_labels'].to(device)

                    outputs = model(pose_features, video_features)
                    frame_logits = outputs['frame_logits']
                    start_logits = outputs['start_logits']
                    end_logits = outputs['end_logits']

                    batch_val_loss = segment_aware_loss(
                        frame_logits, frame_labels,
                        start_logits, start_labels,
                        end_logits, end_labels,
                        pos_weight=pos_weight,
                        hard_negative_ratio=hard_negative_ratio,
                    )
                    val_loss += batch_val_loss.item()

                    batch_preds = torch.sigmoid(frame_logits).cpu().numpy()
                    batch_labels = frame_labels.cpu().numpy()

                    for i in range(min(len(batch_preds), 6)):
                        pred_segments = extract_segments_strict(
                            batch_preds[i],
                            threshold=eval_threshold,
                            min_duration=eval_min_duration,
                            max_duration=eval_max_duration,
                            merge_gap=eval_merge_gap,
                        )
                        true_segments = extract_segments_from_predictions(
                            batch_labels[i], threshold=0.5, min_duration=1
                        )
                        val_predictions.append(pred_segments)
                        val_true_segments.append(true_segments)

                        if len(val_calibration_prediction_sequences) < 200:
                            val_calibration_prediction_sequences.append(batch_preds[i].copy())
                            val_calibration_true_segments.append(true_segments)

            if len(val_dataloader) > 0:
                val_loss /= len(val_dataloader)

            valid_val_samples = 0
            for pred_segs, true_segs in zip(val_predictions, val_true_segments):
                threshold_metrics = [
                    evaluate_pitch_detection(pred_segs, true_segs, iou_threshold=t)
                    for t in iou_thresholds
                ]
                metrics = {
                    'precision': float(np.mean([m['precision'] for m in threshold_metrics])),
                    'recall': float(np.mean([m['recall'] for m in threshold_metrics])),
                    'f1': float(np.mean([m['f1'] for m in threshold_metrics])),
                    'mean_iou': float(np.mean([m['mean_iou'] for m in threshold_metrics])),
                }
                if len(pred_segs) > 0 or len(true_segs) > 0:
                    val_precision += metrics['precision']
                    val_recall += metrics['recall']
                    val_f1 += metrics['f1']
                    val_mean_iou += metrics['mean_iou']
                    valid_val_samples += 1

            if valid_val_samples > 0:
                val_precision /= valid_val_samples
                val_recall /= valid_val_samples
                val_f1 /= valid_val_samples
                val_mean_iou /= valid_val_samples

            model.train()

        history['val_loss'].append(val_loss)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_mean_iou'].append(val_mean_iou)

        selection_iou = val_mean_iou if val_dataloader is not None else avg_mean_iou
        selection_precision = val_precision if val_dataloader is not None else avg_precision
        selection_recall = val_recall if val_dataloader is not None else avg_recall
        selection_score = 0.60 * selection_iou + 0.35 * selection_precision + 0.05 * selection_recall
        if selection_precision < 0.03:
            selection_score *= 0.5

        # Track best checkpoint using validation score when available
        if selection_score > best_score:
            best_score = selection_score
            best_iou = selection_iou
            print(
                f"   🎯 New best checkpoint score: {best_score:.4f} "
                f"(IoU={selection_iou:.4f}, Precision={selection_precision:.4f}, Recall={selection_recall:.4f})"
            )

            calibrated_config, calibration_metrics = calibrate_inference_config(
                val_calibration_prediction_sequences if len(val_calibration_prediction_sequences) > 0 else calibration_prediction_sequences,
                val_calibration_true_segments if len(val_calibration_true_segments) > 0 else calibration_true_segments,
                fps=train_dataset.fps,
                iou_thresholds=iou_thresholds,
            )
            print(
                "   🔧 Calibrated inference config: "
                f"thr={calibrated_config['threshold']:.3f}, "
                f"low_ratio={calibrated_config['low_threshold_ratio']:.2f}, "
                f"min={calibrated_config['min_duration_frames']}f, "
                f"max={calibrated_config['max_duration_frames']}f, "
                f"gap={calibrated_config['merge_gap_frames']}f"
            )
            print(
                f"   📐 Calibration score={calibration_metrics['score']:.4f} "
                f"(IoU={calibration_metrics['mean_iou']:.4f}, "
                f"Precision={calibration_metrics['precision']:.4f}, "
                f"Recall={calibration_metrics.get('recall', 0.0):.4f})"
            )
            
            # Save the best model checkpoint
            if save_best:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = os.path.join(checkpoint_dir, f"best_model_iou_{best_iou:.4f}_epoch_{epoch+1}_{timestamp}.pth")
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_iou': best_iou,
                    'history': history,
                    'hyperparameters': {
                        'num_epochs': num_epochs,
                        'batch_size': batch_size,
                        'window_size': window_size,
                        'stride': stride,
                        'lr': lr,
                        'hidden_dim': 256,
                        'use_augmentation': use_augmentation,
                        'augment_factor': augment_factor,
                        'temporal_shift': temporal_shift,
                        'pose_noise_std': pose_noise_std,
                        'video_noise_std': video_noise_std,
                        'feature_dropout_prob': feature_dropout_prob,
                        'val_ratio': val_ratio,
                        'hard_negative_ratio': hard_negative_ratio,
                        'iou_thresholds': iou_thresholds,
                        'balance_windows': balance_windows,
                        'positive_window_target': positive_window_target,
                    },
                    'dataset_stats': dataset_stats,
                    'training_files': {
                        'pose_files': pose_files,
                        'video_files': video_files,
                        'label_files': label_files
                    },
                    'inference_config': calibrated_config,
                }
                
                torch.save(checkpoint, best_model_path)
                print(f"   💾 Saved best model: {os.path.basename(best_model_path)}")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch summary with IoU emphasis
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Total Loss: {avg_loss:.4f} | IoU Loss: {avg_iou_loss:.4f} | LR: {current_lr:.6f}")
        print(f"  Precision: {avg_precision:.3f} | Recall: {avg_recall:.3f} | F1: {avg_f1:.3f}")
        print(f"  🎯 Mean IoU: {avg_mean_iou:.4f} | Best IoU: {best_iou:.4f}")
        if val_dataloader is not None:
            print(
                f"  ✅ Val Loss: {val_loss:.4f} | Val IoU: {val_mean_iou:.4f} | "
                f"Val Precision: {val_precision:.3f} | Val Recall: {val_recall:.3f} | Val F1: {val_f1:.3f}"
            )
        
        # Early stopping based on IoU plateau
        if epoch > 3 and len(history['mean_iou']) > 3:
            recent_ious = history['mean_iou'][-3:]
            if all(abs(recent_ious[i] - recent_ious[i-1]) < 0.001 for i in range(1, len(recent_ious))):
                print(f"   IoU plateaued - consider stopping or adjusting hyperparameters")
    
    print("✅ IoU-optimized training completed!")
    print(f"🏆 Final best IoU: {best_iou:.4f}")
    
    if save_best and best_model_path:
        print(f"💾 Best model saved to: {best_model_path}")
        print(f"   You can load this model later using: load_best_model('{best_model_path}')")
    
    return model, history, best_model_path
