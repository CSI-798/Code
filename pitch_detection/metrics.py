"""
Evaluation metrics for pitch detection.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def calculate_segment_iou(pred_start, pred_end, true_start, true_end):
    """Calculate Intersection over Union for two segments"""
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (true_end - true_start) - intersection
    
    return intersection / union if union > 0 else 0.0


def extract_segments_from_predictions(predictions, threshold=0.5, min_duration=10):
    """Extract pitch segments from frame-level predictions with improved post-processing"""
    segments = []
    in_pitch = False
    start_frame = 0
    
    # Apply smoothing to reduce noise
    if len(predictions) > 5:
        smoothed = uniform_filter1d(predictions, size=3)
    else:
        smoothed = predictions
    
    # Use adaptive threshold if predictions are generally low
    pred_max = np.max(smoothed)
    pred_mean = np.mean(smoothed)
    
    if pred_max < 0.3:  # If all predictions are low, use adaptive threshold
        adaptive_threshold = pred_mean + 0.5 * (pred_max - pred_mean)
        threshold = min(threshold, adaptive_threshold)
    
    for frame_idx, confidence in enumerate(smoothed):
        if confidence > threshold and not in_pitch:
            in_pitch = True
            start_frame = frame_idx
        elif confidence <= threshold and in_pitch:
            in_pitch = False
            duration = frame_idx - start_frame
            if duration >= min_duration:  # Filter out very short segments
                segments.append((start_frame, frame_idx))
    
    # Handle pitch continuing to end
    if in_pitch:
        duration = len(smoothed) - start_frame
        if duration >= min_duration:
            segments.append((start_frame, len(smoothed)))
    
    return segments


def extract_segments_strict(
    predictions,
    threshold=0.5,
    min_duration=10,
    smoothing_size=3,
    low_threshold_ratio=0.7,
    max_duration=None,
    merge_gap=0,
):
    """Extract segments using fixed hysteresis thresholding (no adaptive threshold)."""
    segments = []
    in_pitch = False
    start_frame = 0

    if len(predictions) > smoothing_size:
        smoothed = uniform_filter1d(predictions, size=smoothing_size)
    else:
        smoothed = predictions

    high_threshold = float(threshold)
    low_threshold = float(threshold) * float(low_threshold_ratio)

    for frame_idx, confidence in enumerate(smoothed):
        if confidence > high_threshold and not in_pitch:
            in_pitch = True
            start_frame = frame_idx
        elif confidence <= low_threshold and in_pitch:
            in_pitch = False
            end_frame = frame_idx
            duration = end_frame - start_frame
            if duration >= min_duration and (max_duration is None or duration <= max_duration):
                segments.append((start_frame, end_frame))

    if in_pitch:
        end_frame = len(smoothed)
        duration = end_frame - start_frame
        if duration >= min_duration and (max_duration is None or duration <= max_duration):
            segments.append((start_frame, end_frame))

    if merge_gap > 0 and len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg[0] - prev[1]
            if gap <= merge_gap:
                merged[-1] = (prev[0], seg[1])
            else:
                merged.append(seg)
        segments = merged

    return segments


def evaluate_pitch_detection(predicted_segments, true_segments, iou_threshold=0.3):
    """Evaluate pitch detection using IoU-based metrics"""
    if len(true_segments) == 0:
        return {
            'precision': 1.0 if len(predicted_segments) == 0 else 0.0,
            'recall': 1.0,
            'f1': 1.0 if len(predicted_segments) == 0 else 0.0,
            'mean_iou': 0.0,
            'matched_segments': 0,
            'total_predicted': len(predicted_segments),
            'total_true': len(true_segments)
        }
    
    if len(predicted_segments) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mean_iou': 0.0,
            'matched_segments': 0,
            'total_predicted': len(predicted_segments),
            'total_true': len(true_segments)
        }
    
    # Match predicted segments to true segments using IoU
    matched_pred = set()
    matched_true = set()
    ious = []
    
    for i, (pred_start, pred_end) in enumerate(predicted_segments):
        best_iou = 0.0
        best_match = -1
        
        for j, (true_start, true_end) in enumerate(true_segments):
            if j in matched_true:
                continue
                
            iou = calculate_segment_iou(pred_start, pred_end, true_start, true_end)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = j
        
        if best_match >= 0:
            matched_pred.add(i)
            matched_true.add(best_match)
            ious.append(best_iou)
    
    # Calculate metrics
    precision = len(matched_pred) / len(predicted_segments)
    recall = len(matched_true) / len(true_segments)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'matched_segments': len(matched_pred),
        'total_predicted': len(predicted_segments),
        'total_true': len(true_segments)
    }


def analyze_dataset_distribution(dataset):
    """Analyze pitch vs non-pitch distribution in the dataset"""
    total_frames = 0
    pitch_frames = 0
    pitch_segments = 0
    non_pitch_segments = 0
    
    print("📊 Analyzing dataset distribution...")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        frame_labels = sample['frame_labels'].numpy()
        
        total_frames += len(frame_labels)
        pitch_frames += np.sum(frame_labels)
        
        # Count segments in this window
        segments = extract_segments_from_predictions(frame_labels, threshold=0.5, min_duration=1)
        pitch_segments += len(segments)
        
        # Count non-pitch segments (gaps between pitches)
        if len(segments) > 0:
            # Count gaps between segments and at beginning/end
            prev_end = 0
            for start, end in segments:
                if start > prev_end:
                    non_pitch_segments += 1
                prev_end = end
            if prev_end < len(frame_labels):
                non_pitch_segments += 1
        else:
            non_pitch_segments += 1
    
    pitch_ratio = pitch_frames / total_frames if total_frames > 0 else 0
    
    print(f"  Total frames: {total_frames:,}")
    print(f"  Pitch frames: {pitch_frames:,} ({pitch_ratio:.1%})")
    print(f"  Non-pitch frames: {total_frames - pitch_frames:,} ({1-pitch_ratio:.1%})")
    print(f"  Pitch segments: {pitch_segments}")
    print(f"  Non-pitch segments: {non_pitch_segments}")
    print(f"  Avg frames per pitch segment: {pitch_frames/pitch_segments:.1f}" if pitch_segments > 0 else "  No pitch segments found")
    print("-" * 50)
    
    return {
        'total_frames': total_frames,
        'pitch_frames': pitch_frames,
        'non_pitch_frames': total_frames - pitch_frames,
        'pitch_segments': pitch_segments,
        'non_pitch_segments': non_pitch_segments,
        'pitch_ratio': pitch_ratio
    }
