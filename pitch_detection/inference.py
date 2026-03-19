"""
Inference utilities for pitch detection model.
"""

import os
import glob
import json
import torch
import numpy as np
from scipy.ndimage import uniform_filter1d

from .model import MultiModalBMN
from .metrics import calculate_segment_iou, evaluate_pitch_detection, extract_segments_from_predictions


def _body_center_and_scale(frame_xy: np.ndarray):
    hips = frame_xy[[11, 12]]
    shoulders = frame_xy[[5, 6]]

    hip_valid = np.all(hips > 0, axis=1)
    shoulder_valid = np.all(shoulders > 0, axis=1)

    if np.any(hip_valid):
        center = hips[hip_valid].mean(axis=0)
    elif np.any(shoulder_valid):
        center = shoulders[shoulder_valid].mean(axis=0)
    else:
        valid = frame_xy[np.all(frame_xy > 0, axis=1)]
        if len(valid) == 0:
            return np.array([0.5, 0.5], dtype=np.float32), 1.0
        center = valid.mean(axis=0)

    scale_candidates = []
    if np.all(shoulders > 0):
        scale_candidates.append(float(np.linalg.norm(shoulders[0] - shoulders[1])))
    if np.all(hips > 0):
        scale_candidates.append(float(np.linalg.norm(hips[0] - hips[1])))

    valid = frame_xy[np.all(frame_xy > 0, axis=1)]
    if len(valid) >= 2:
        bbox_size = float(np.linalg.norm(valid.max(axis=0) - valid.min(axis=0)))
        if bbox_size > 0:
            scale_candidates.append(0.5 * bbox_size)

    scale = np.median(scale_candidates) if len(scale_candidates) > 0 else 1.0
    scale = float(max(scale, 1e-3))

    return center.astype(np.float32), scale


def _build_pose_representation(poses_xy_flat: np.ndarray):
    if poses_xy_flat.ndim != 2 or poses_xy_flat.shape[1] != 34:
        return np.zeros((len(poses_xy_flat), 68), dtype=np.float32)

    num_frames = poses_xy_flat.shape[0]
    rel_coords = np.zeros((num_frames, 34), dtype=np.float32)

    for frame_idx in range(num_frames):
        frame_xy = poses_xy_flat[frame_idx].reshape(17, 2).astype(np.float32)
        center, scale = _body_center_and_scale(frame_xy)

        centered = np.zeros_like(frame_xy, dtype=np.float32)
        valid = np.all(frame_xy > 0, axis=1)
        if np.any(valid):
            centered[valid] = (frame_xy[valid] - center) / scale
            centered = np.clip(centered, -3.0, 3.0)

        rel_coords[frame_idx] = centered.reshape(-1)

    velocities = np.zeros_like(rel_coords, dtype=np.float32)
    if num_frames > 1:
        velocities[1:] = rel_coords[1:] - rel_coords[:-1]
        velocities = np.clip(velocities, -1.5, 1.5)

    return np.concatenate([rel_coords, velocities], axis=1).astype(np.float32)


def _default_inference_config(fps=30):
    return {
        'threshold': 0.2,
        'low_threshold_ratio': 0.7,
        'start_threshold': 0.12,
        'end_threshold': 0.12,
        'min_duration_frames': max(8, int(0.30 * fps)),
        'max_duration_frames': max(20, int(2.20 * fps)),
        'merge_gap_frames': max(1, int(0.08 * fps)),
        'min_avg_confidence_ratio': 0.9,
        'smoothing_size': 5,
    }


def _normalize_inference_config(inference_config, fps=30, threshold_override=None):
    cfg = _default_inference_config(fps)
    if isinstance(inference_config, dict):
        cfg.update(inference_config)

    if threshold_override is not None:
        cfg['threshold'] = float(threshold_override)

    cfg['threshold'] = float(min(0.95, max(0.01, cfg['threshold'])))
    cfg['low_threshold_ratio'] = float(min(0.99, max(0.1, cfg['low_threshold_ratio'])))
    cfg['start_threshold'] = float(min(0.99, max(0.01, cfg['start_threshold'])))
    cfg['end_threshold'] = float(min(0.99, max(0.01, cfg['end_threshold'])))
    cfg['min_duration_frames'] = int(max(1, cfg['min_duration_frames']))
    cfg['max_duration_frames'] = int(max(cfg['min_duration_frames'], cfg['max_duration_frames']))
    cfg['merge_gap_frames'] = int(max(0, cfg['merge_gap_frames']))
    cfg['min_avg_confidence_ratio'] = float(min(1.0, max(0.0, cfg['min_avg_confidence_ratio'])))
    cfg['smoothing_size'] = int(max(1, cfg['smoothing_size']))
    return cfg


def _decode_pitch_segments(smoothed_predictions, fps, cfg, start_scores=None, end_scores=None):
    pitch_segments = []
    in_pitch = False
    start_frame = 0

    high_threshold = cfg['threshold']
    low_threshold = high_threshold * cfg['low_threshold_ratio']

    for frame_idx, confidence in enumerate(smoothed_predictions):
        if confidence > high_threshold and not in_pitch:
            in_pitch = True
            start_frame = frame_idx
        elif confidence <= low_threshold and in_pitch:
            in_pitch = False
            end_frame = frame_idx
            duration_frames = end_frame - start_frame

            if cfg['min_duration_frames'] <= duration_frames <= cfg['max_duration_frames']:
                avg_conf = float(np.mean(smoothed_predictions[start_frame:end_frame]))
                end_idx = max(start_frame, min(len(smoothed_predictions) - 1, end_frame - 1))
                start_conf = float(start_scores[start_frame]) if start_scores is not None else high_threshold
                end_conf = float(end_scores[end_idx]) if end_scores is not None else high_threshold

                if (
                    avg_conf >= high_threshold * cfg['min_avg_confidence_ratio']
                    and start_conf >= cfg['start_threshold']
                    and end_conf >= cfg['end_threshold']
                ):
                    pitch_segments.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'start_time': start_frame / fps,
                        'end_time': end_frame / fps,
                        'duration': (end_frame - start_frame) / fps,
                        'confidence': avg_conf,
                    })

    if in_pitch:
        end_frame = len(smoothed_predictions)
        duration_frames = end_frame - start_frame
        if cfg['min_duration_frames'] <= duration_frames <= cfg['max_duration_frames']:
            avg_conf = float(np.mean(smoothed_predictions[start_frame:end_frame]))
            end_idx = max(start_frame, min(len(smoothed_predictions) - 1, end_frame - 1))
            start_conf = float(start_scores[start_frame]) if start_scores is not None else high_threshold
            end_conf = float(end_scores[end_idx]) if end_scores is not None else high_threshold
            if (
                avg_conf >= high_threshold * cfg['min_avg_confidence_ratio']
                and start_conf >= cfg['start_threshold']
                and end_conf >= cfg['end_threshold']
            ):
                pitch_segments.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_frame / fps,
                    'end_time': end_frame / fps,
                    'duration': (end_frame - start_frame) / fps,
                    'confidence': avg_conf,
                })

    if cfg['merge_gap_frames'] > 0 and len(pitch_segments) > 1:
        merged_segments = [pitch_segments[0]]
        for seg in pitch_segments[1:]:
            prev = merged_segments[-1]
            gap = seg['start_frame'] - prev['end_frame']
            merged_duration = seg['end_frame'] - prev['start_frame']

            if gap <= cfg['merge_gap_frames'] and merged_duration <= cfg['max_duration_frames']:
                prev['end_frame'] = seg['end_frame']
                prev['end_time'] = seg['end_time']
                prev['duration'] = prev['end_time'] - prev['start_time']
                prev['confidence'] = max(prev['confidence'], seg['confidence'])
            else:
                merged_segments.append(seg)
        pitch_segments = merged_segments

    return pitch_segments


def calibrate_inference_config(prediction_sequences, true_segments_sequences, fps=30, iou_thresholds=None):
    """Grid-search inference config using IoU with precision guardrail."""
    if not prediction_sequences or not true_segments_sequences:
        return _default_inference_config(fps), {'score': 0.0, 'precision': 0.0, 'mean_iou': 0.0}

    if not iou_thresholds:
        iou_thresholds = [0.3]
    iou_thresholds = [float(t) for t in iou_thresholds]

    threshold_grid = [0.1, 0.14, 0.18, 0.22, 0.28, 0.35]
    low_ratio_grid = [0.65, 0.7, 0.75]
    min_duration_grid = [max(6, int(0.25 * fps)), max(8, int(0.30 * fps)), max(10, int(0.35 * fps))]
    max_duration_grid = [max(18, int(1.8 * fps)), max(22, int(2.2 * fps)), max(26, int(2.6 * fps))]
    gap_grid = [max(0, int(0.03 * fps)), max(1, int(0.08 * fps))]

    best_cfg = _default_inference_config(fps)
    best_metrics = {'score': -1.0, 'precision': 0.0, 'mean_iou': 0.0, 'recall': 0.0}
    fallback_cfg = _default_inference_config(fps)
    fallback_metrics = {'score': -1.0, 'precision': 0.0, 'mean_iou': 0.0, 'recall': 0.0}

    valid_pairs = [
        (preds, true_segments)
        for preds, true_segments in zip(prediction_sequences, true_segments_sequences)
        if len(true_segments) > 0
    ]
    if not valid_pairs:
        return _default_inference_config(fps), {'score': 0.0, 'precision': 0.0, 'mean_iou': 0.0, 'recall': 0.0}

    for threshold in threshold_grid:
        for low_ratio in low_ratio_grid:
            for min_duration in min_duration_grid:
                for max_duration in max_duration_grid:
                    if max_duration < min_duration:
                        continue
                    for merge_gap in gap_grid:
                        cfg = _normalize_inference_config(
                            {
                                'threshold': threshold,
                                'low_threshold_ratio': low_ratio,
                                'start_threshold': max(0.06, threshold * 0.55),
                                'end_threshold': max(0.06, threshold * 0.55),
                                'min_duration_frames': min_duration,
                                'max_duration_frames': max_duration,
                                'merge_gap_frames': merge_gap,
                            },
                            fps=fps,
                        )

                        metrics_list = []
                        for preds, true_segments in valid_pairs:
                            smoothed = uniform_filter1d(preds, size=cfg['smoothing_size']) if len(preds) > 3 else preds
                            pred_segments = _decode_pitch_segments(smoothed, fps=fps, cfg=cfg)
                            pred_pairs = [(seg['start_frame'], seg['end_frame']) for seg in pred_segments]
                            threshold_metrics = [
                                evaluate_pitch_detection(pred_pairs, true_segments, iou_threshold=t)
                                for t in iou_thresholds
                            ]
                            metrics_list.append({
                                'mean_iou': float(np.mean([m['mean_iou'] for m in threshold_metrics])),
                                'precision': float(np.mean([m['precision'] for m in threshold_metrics])),
                                'recall': float(np.mean([m['recall'] for m in threshold_metrics])),
                            })

                        mean_iou = float(np.mean([m['mean_iou'] for m in metrics_list]))
                        precision = float(np.mean([m['precision'] for m in metrics_list]))
                        recall = float(np.mean([m['recall'] for m in metrics_list]))
                        raw_score = 0.60 * mean_iou + 0.30 * precision + 0.10 * recall

                        if raw_score > fallback_metrics['score']:
                            fallback_metrics = {'score': raw_score, 'precision': precision, 'mean_iou': mean_iou, 'recall': recall}
                            fallback_cfg = cfg

                        # Precision floor to avoid choosing degenerate very-high-threshold configs
                        if precision < 0.02:
                            continue

                        score = raw_score

                        if score > best_metrics['score']:
                            best_metrics = {'score': score, 'precision': precision, 'mean_iou': mean_iou, 'recall': recall}
                            best_cfg = cfg

    if best_metrics['score'] < 0:
        if fallback_metrics['score'] >= 0:
            return fallback_cfg, fallback_metrics
        return _default_inference_config(fps), {'score': 0.0, 'precision': 0.0, 'mean_iou': 0.0, 'recall': 0.0}

    return best_cfg, best_metrics


def load_best_model(checkpoint_path, device='cuda'):
    """
    Load the best saved model from checkpoint
    
    Args:
        checkpoint_path: Path to the saved checkpoint file
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model in evaluation mode
        checkpoint_info: Dictionary with training information
    """
    print(f"📂 Loading model checkpoint from: {checkpoint_path}")
    
    # Load checkpoint with weights_only=False for compatibility with numpy objects
    # This is safe since we trust our own checkpoint files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with same architecture
    hidden_dim = checkpoint['hyperparameters'].get('hidden_dim', 256)
    model = MultiModalBMN(hidden_dim=hidden_dim).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract useful information
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'best_iou': checkpoint['best_iou'],
        'hyperparameters': checkpoint['hyperparameters'],
        'inference_config': checkpoint.get('inference_config'),
        'dataset_stats': checkpoint.get('dataset_stats', {}),
        'training_files': checkpoint.get('training_files', {}),
        'final_metrics': {
            'final_loss': checkpoint['history']['loss'][-1] if checkpoint['history']['loss'] else None,
            'final_precision': checkpoint['history']['precision'][-1] if checkpoint['history']['precision'] else None,
            'final_recall': checkpoint['history']['recall'][-1] if checkpoint['history']['recall'] else None,
            'final_f1': checkpoint['history']['f1'][-1] if checkpoint['history']['f1'] else None,
            'final_iou': checkpoint['history']['mean_iou'][-1] if checkpoint['history']['mean_iou'] else None
        }
    }
    
    print(f"✅ Model loaded successfully!")
    print(f"   📊 Best IoU achieved: {checkpoint_info['best_iou']:.4f}")
    print(f"   📈 Trained for {checkpoint_info['epoch']} epochs")
    print(f"   🎯 Final metrics:")
    print(f"      IoU: {checkpoint_info['final_metrics']['final_iou']:.4f}")
    print(f"      F1: {checkpoint_info['final_metrics']['final_f1']:.4f}")
    print(f"      Precision: {checkpoint_info['final_metrics']['final_precision']:.4f}")
    print(f"      Recall: {checkpoint_info['final_metrics']['final_recall']:.4f}")
    
    return model, checkpoint_info


def find_best_checkpoint(checkpoint_dir="./checkpoints"):
    """
    Find the checkpoint with the highest IoU in the checkpoint directory
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
    
    Returns:
        best_checkpoint_path: Path to the best checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "best_model_iou_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return None
    
    # Extract IoU values from filenames and find the best one
    best_iou = 0.0
    best_checkpoint = None
    
    for checkpoint_file in checkpoint_files:
        try:
            # Extract IoU from filename: best_model_iou_0.1234_epoch_...
            filename = os.path.basename(checkpoint_file)
            iou_start = filename.find("iou_") + 4
            iou_end = filename.find("_epoch_")
            iou_value = float(filename[iou_start:iou_end])
            
            if iou_value > best_iou:
                best_iou = iou_value
                best_checkpoint = checkpoint_file
                
        except (ValueError, IndexError):
            print(f"⚠️ Could not parse IoU from filename: {filename}")
            continue
    
    if best_checkpoint:
        print(f"🏆 Found best checkpoint: IoU = {best_iou:.4f}")
        print(f"📁 Path: {best_checkpoint}")
    else:
        print(f"❌ Could not determine best checkpoint")
    
    return best_checkpoint


def predict_pitch_timestamps(model, pose_file, video_file, window_size=100, stride=50,
                            fps=30, threshold=None, device='cuda', inference_config=None):
    """
    Predict pitch start and end timestamps from pose and video features
    
    Args:
        model: Trained pitch detection model
        pose_file: Path to pose feature JSON file
        video_file: Path to video feature .npy file
        window_size: Size of sliding window
        stride: Stride for sliding window
        fps: Frames per second of the video
        threshold: Optional threshold override
        device: Device to run inference on
        inference_config: Optional calibrated inference settings dictionary
    
    Returns:
        pitch_segments: List of detected pitch segments with timing info
        predictions: Frame-level prediction scores
    """
    # Load pose data with proper error handling
    with open(pose_file, 'r') as f:
        pose_json = json.load(f)

    pose_meta = pose_json.get('metadata', {})
    frame_width = float(pose_meta.get('width', 1920) or 1920)
    frame_height = float(pose_meta.get('height', 1080) or 1080)
    frame_width = max(1.0, frame_width)
    frame_height = max(1.0, frame_height)
    
    # Process poses frame by frame to handle missing poses
    poses_list = []
    for frame in pose_json['poses']:
        if frame['poses'] and len(frame['poses']) > 0:
            poses = frame['poses']
            primary_pose_index = frame.get('primary_pose_index', None)

            if isinstance(primary_pose_index, int) and 0 <= primary_pose_index < len(poses):
                selected_pose = poses[primary_pose_index]
            else:
                selected_pose = max(poses, key=lambda pose: pose.get('pose_score', 0.0))

            keypoints = selected_pose['keypoints']
            if len(keypoints) == 34:  # Ensure we have x,y for 17 joints
                kp = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
                kp[:, 0] = np.clip(kp[:, 0] / frame_width, 0.0, 1.0)
                kp[:, 1] = np.clip(kp[:, 1] / frame_height, 0.0, 1.0)
                poses_list.append(kp.reshape(-1))
            else:
                poses_list.append(np.zeros(34))  # Fallback for malformed data
        else:
            poses_list.append(np.zeros(34))  # No pose detected
    
    # Convert to numpy array safely and build body-centric+velocity representation
    poses = np.array(poses_list, dtype=np.float32)
    extended_poses = _build_pose_representation(poses)
    
    video_features = np.load(video_file).astype(np.float32)
    if video_features.ndim != 2:
        video_features = np.reshape(video_features, (video_features.shape[0], -1))

    vf_mean = np.mean(video_features, axis=0, keepdims=True)
    vf_std = np.std(video_features, axis=0, keepdims=True)
    video_features = (video_features - vf_mean) / (vf_std + 1e-6)
    video_features = np.clip(video_features, -5.0, 5.0)
    
    # Sync lengths
    min_len = min(len(extended_poses), len(video_features))
    extended_poses = extended_poses[:min_len]
    video_features = video_features[:min_len]
    
    print(f"Processing {min_len} frames for prediction...")
    
    # Create sliding windows for prediction
    model.eval()
    predictions = np.zeros(min_len)
    start_predictions = np.zeros(min_len)
    end_predictions = np.zeros(min_len)
    window_counts = np.zeros(min_len)  # Track overlapping windows
    
    with torch.no_grad():
        for start in range(0, max(1, min_len - window_size + 1), stride):
            end = min(start + window_size, min_len)
            actual_window_size = end - start
            
            # Skip if window is too small
            if actual_window_size < window_size // 2:
                continue
                
            # Pad window if needed
            pose_window = extended_poses[start:end]
            video_window = video_features[start:end]
            
            if actual_window_size < window_size:
                pad_size = window_size - actual_window_size
                pose_window = np.pad(pose_window, ((0, pad_size), (0, 0)), mode='constant')
                video_window = np.pad(video_window, ((0, pad_size), (0, 0)), mode='constant')
            
            pose_tensor = torch.FloatTensor(pose_window).unsqueeze(0).to(device)
            video_tensor = torch.FloatTensor(video_window).unsqueeze(0).to(device)
            
            outputs = model(pose_tensor, video_tensor)
            frame_preds = outputs['frame_predictions'].cpu().numpy()[0]
            start_preds = outputs['start_predictions'].cpu().numpy()[0]
            end_preds = outputs['end_predictions'].cpu().numpy()[0]
            
            # Only use predictions for actual frames (not padded ones)
            valid_preds = frame_preds[:actual_window_size]
            valid_start = start_preds[:actual_window_size]
            valid_end = end_preds[:actual_window_size]
            predictions[start:end] += valid_preds
            start_predictions[start:end] += valid_start
            end_predictions[start:end] += valid_end
            window_counts[start:end] += 1
    
    # Average overlapping predictions
    mask = window_counts > 0
    predictions[mask] = predictions[mask] / window_counts[mask]
    start_predictions[mask] = start_predictions[mask] / window_counts[mask]
    end_predictions[mask] = end_predictions[mask] / window_counts[mask]
    
    cfg = _normalize_inference_config(inference_config, fps=fps, threshold_override=threshold)

    # Apply smoothing to reduce noise
    if len(predictions) > cfg['smoothing_size']:
        smoothed_predictions = uniform_filter1d(predictions, size=cfg['smoothing_size'])
    else:
        smoothed_predictions = predictions

    boundary_smoothing = max(3, cfg['smoothing_size'] // 2)
    if len(start_predictions) > boundary_smoothing:
        smoothed_start = uniform_filter1d(start_predictions, size=boundary_smoothing)
        smoothed_end = uniform_filter1d(end_predictions, size=boundary_smoothing)
    else:
        smoothed_start = start_predictions
        smoothed_end = end_predictions
    
    # Debug: Print prediction statistics
    print(f"Raw prediction stats: min={np.min(predictions):.6f}, max={np.max(predictions):.6f}, mean={np.mean(predictions):.6f}")
    print(f"Smoothed prediction stats: min={np.min(smoothed_predictions):.6f}, max={np.max(smoothed_predictions):.6f}, mean={np.mean(smoothed_predictions):.6f}")
    
    # Check if predictions are meaningful
    pred_max = np.max(smoothed_predictions)
    pred_mean = np.mean(smoothed_predictions)
    pred_std = np.std(smoothed_predictions)
    
    print(f"Prediction analysis: max={pred_max:.6f}, mean={pred_mean:.6f}, std={pred_std:.6f}")
    
    # If all predictions are essentially zero or very low, return empty results
    if pred_max < 0.001 or pred_std < 0.001:
        print(f"⚠️ Model predictions are too low or uniform (max={pred_max:.6f}, std={pred_std:.6f})")
        print("This suggests the model hasn't learned to distinguish pitch vs non-pitch frames.")
        print("Consider:")
        print("  1. Training for more epochs")
        print("  2. Adjusting learning rate or loss function")
        print("  3. Checking if training data has proper labels")
        return [], predictions
    
    print(
        f"Using inference config: threshold={cfg['threshold']:.3f}, "
        f"low_ratio={cfg['low_threshold_ratio']:.2f}, "
        f"start_thr={cfg['start_threshold']:.2f}, end_thr={cfg['end_threshold']:.2f}, "
        f"min_dur={cfg['min_duration_frames']}f, max_dur={cfg['max_duration_frames']}f, "
        f"merge_gap={cfg['merge_gap_frames']}f"
    )

    pitch_segments = _decode_pitch_segments(
        smoothed_predictions,
        fps=fps,
        cfg=cfg,
        start_scores=smoothed_start,
        end_scores=smoothed_end,
    )

    print(f"Found {len(pitch_segments)} pitch segments with constrained logic")
    return pitch_segments, predictions


def evaluate_model_performance(model, pose_file, video_file, label_file, fps=30,
                              threshold=None, device='cuda', inference_config=None):
    """
    Comprehensive evaluation of model performance with IoU metrics
    
    Args:
        model: Trained pitch detection model
        pose_file: Path to pose feature JSON file
        video_file: Path to video feature .npy file
        label_file: Path to ground truth label JSON file
        fps: Frames per second of the video
        threshold: Optional threshold override
        device: Device to run inference on
        inference_config: Optional calibrated inference settings dictionary
    
    Returns:
        Dictionary containing segments, metrics, timing errors, and frame predictions
    """
    # Make predictions
    pitch_segments, frame_predictions = predict_pitch_timestamps(
        model,
        pose_file,
        video_file,
        fps=fps,
        threshold=threshold,
        device=device,
        inference_config=inference_config,
    )
    
    # Load and parse ground truth labels
    with open(label_file, 'r') as f:
        raw_labels = json.load(f)
    
    # Parse the label format correctly
    true_segments = []
    if isinstance(raw_labels, list) and len(raw_labels) > 0:
        # Handle the actual label format: [{"videoLabels": [{"ranges": [...]}]}]
        video_data = raw_labels[0]  # First video in the list
        if 'videoLabels' in video_data:
            for label_group in video_data['videoLabels']:
                if 'ranges' in label_group and 'timelinelabels' in label_group:
                    # Only process "Pitch" labels
                    if 'Pitch' in label_group['timelinelabels']:
                        for range_data in label_group['ranges']:
                            true_segments.append((range_data['start'], range_data['end']))
    
    print(f"Ground truth: {len(true_segments)} pitch segments")
    for i, (start, end) in enumerate(true_segments):
        duration = (end - start) / fps
        print(f"  True Pitch {i+1}: frames {start}-{end} ({duration:.2f}s)")
    
    # Convert predicted segments to frame-based
    pred_segments = [(seg['start_frame'], seg['end_frame']) for seg in pitch_segments]
    
    # Calculate IoU-based metrics
    metrics = evaluate_pitch_detection(pred_segments, true_segments)
    
    # Calculate timing accuracy for matched segments
    timing_errors = []
    if len(pitch_segments) > 0 and len(true_segments) > 0:
        for pred_seg in pitch_segments:
            best_iou = 0
            best_match = None
            for i, (true_start, true_end) in enumerate(true_segments):
                iou = calculate_segment_iou(
                    pred_seg['start_frame'], pred_seg['end_frame'],
                    true_start, true_end
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match = (true_start, true_end)
            
            if best_match and best_iou > 0.3:
                start_error = abs(pred_seg['start_frame'] - best_match[0]) / fps
                end_error = abs(pred_seg['end_frame'] - best_match[1]) / fps
                timing_errors.append({'start_error': start_error, 'end_error': end_error})
    
    return {
        'segments': pitch_segments,
        'metrics': metrics,
        'timing_errors': timing_errors,
        'frame_predictions': frame_predictions
    }
