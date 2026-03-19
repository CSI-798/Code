"""
Loss functions for pitch detection training.
Includes IoU, Dice, Precision, False Positive Penalty, and Tversky losses.
"""

import torch
import torch.nn as nn


def calculate_iou_loss(predictions, targets, smooth=1e-6):
    """Calculate IoU loss for better segmentation performance"""
    # Flatten tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection and union
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Return 1 - IoU as loss (we want to minimize this)
    return 1 - iou


def calculate_dice_loss(predictions, targets, smooth=1e-6):
    """Calculate Dice loss as alternative to IoU"""
    # Flatten tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Calculate intersection
    intersection = (predictions * targets).sum()
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    # Return 1 - Dice as loss
    return 1 - dice


def calculate_precision_loss(predictions, targets, smooth=1e-6):
    """
    Calculate precision-focused loss to heavily penalize false positives.
    Precision = TP / (TP + FP)
    High precision means fewer false positives.
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # True positives: where both prediction and target are positive
    true_positives = (predictions * targets).sum()
    
    # Predicted positives: all frames where we predict positive
    predicted_positives = predictions.sum()
    
    # Precision
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    
    # Return 1 - precision as loss
    return 1 - precision


def calculate_false_positive_penalty(predictions, targets, penalty_weight=2.0):
    """
    Explicitly penalize false positives more heavily.
    FP = predictions > 0 where targets = 0
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # False positives: high predictions where target is 0
    false_positives = predictions * (1 - targets)
    
    # Average false positive score
    fp_penalty = false_positives.sum() / (targets.numel() + 1e-6)
    
    return penalty_weight * fp_penalty


def calculate_tversky_loss(predictions, targets, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky loss - generalization of Dice that allows weighting FP vs FN differently.
    alpha controls weight of false positives (higher = penalize FP more)
    beta controls weight of false negatives
    For alpha > beta: prioritizes precision (fewer false positives)
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # True positives
    tp = (predictions * targets).sum()
    
    # False positives (predict positive, actually negative)
    fp = (predictions * (1 - targets)).sum()
    
    # False negatives (predict negative, actually positive)
    fn = ((1 - predictions) * targets).sum()
    
    # Tversky index
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    
    return 1 - tversky


def segment_aware_loss(frame_preds, frame_targets, start_preds, start_targets, end_preds, end_targets, 
                       use_precision_focus=True, fp_penalty_weight=4.0,
                       pos_weight=1.0, hard_negative_ratio=0.0):
    """
    Enhanced loss function with multiple improvements:
    1. IoU loss for overall segment overlap
    2. Focal loss for class imbalance
    3. Precision loss to minimize false positives  
    4. False positive penalty for explicit FP penalization
    5. Tversky loss as alternative to IoU with FP/FN weighting
    6. Segment boundary detection
    
    Args:
        frame_preds: Frame-level predictions (logits)
        frame_targets: Frame-level ground truth labels
        start_preds: Start boundary predictions (logits)
        start_targets: Start boundary ground truth labels
        end_preds: End boundary predictions (logits)
        end_targets: End boundary ground truth labels
        use_precision_focus: If True, adds precision and FP penalty terms
        fp_penalty_weight: How heavily to penalize false positives (higher = stricter)
    
    Returns:
        Combined loss value
    """
    batch_size = frame_preds.size(0)
    
    total_iou_loss = 0
    total_focal_loss = 0
    total_precision_loss = 0
    total_fp_penalty = 0
    total_tversky_loss = 0
    total_segment_loss = 0
    total_weighted_bce_loss = 0
    total_hard_negative_loss = 0
    
    for b in range(batch_size):
        # Enhanced IoU loss with smoothing for empty predictions
        frame_probs = torch.sigmoid(frame_preds[b])
        targets = frame_targets[b]
        
        # Calculate IoU with better handling of edge cases
        intersection = (frame_probs * targets).sum()
        union = frame_probs.sum() + targets.sum() - intersection
        
        # Add smoothing and handle empty predictions better
        smooth = 1e-6
        iou = (intersection + smooth) / (union + smooth)
        
        # Enhanced IoU loss that penalizes poor overlap more heavily
        iou_loss = 1 - iou
        if union < 1e-3:  # Handle cases with no predictions or targets
            iou_loss = torch.tensor(1.0).to(frame_preds.device)
        
        total_iou_loss += iou_loss
        
        # Add focal loss to handle class imbalance better
        alpha = 0.35  # Lower positive bias to reduce false positives
        gamma = 2.0   # Focusing parameter
        
        bce = nn.functional.binary_cross_entropy_with_logits(frame_preds[b], targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = alpha * (1 - pt) ** gamma * bce
        total_focal_loss += focal_loss.mean()

        # Weighted BCE to explicitly use class-imbalance prior
        weighted_bce = nn.functional.binary_cross_entropy_with_logits(
            frame_preds[b],
            targets,
            reduction='none',
            pos_weight=torch.tensor(min(8.0, float(pos_weight)), device=frame_preds.device)
        )
        total_weighted_bce_loss += weighted_bce.mean()

        # Hard negative mining: focus on highest-loss negative frames
        if hard_negative_ratio > 0:
            negative_mask = targets < 0.5
            positive_count = int((targets >= 0.5).sum().item())
            negative_count = int(negative_mask.sum().item())

            if negative_count > 0:
                # If no positives in this window, still mine a small number of negatives
                if positive_count == 0:
                    top_k = min(32, negative_count)
                else:
                    top_k = min(negative_count, max(1, int(positive_count * float(hard_negative_ratio))))

                negative_losses = weighted_bce[negative_mask]
                top_neg_losses, _ = torch.topk(negative_losses, k=top_k)
                total_hard_negative_loss += top_neg_losses.mean()
        
        # NEW: Precision-focused components
        if use_precision_focus:
            # Precision loss - encourages high precision (fewer FPs)
            precision_loss = calculate_precision_loss(frame_probs, targets)
            total_precision_loss += precision_loss
            
            # Explicit false positive penalty
            fp_penalty = calculate_false_positive_penalty(frame_probs, targets, penalty_weight=fp_penalty_weight)
            total_fp_penalty += fp_penalty
            
            # Tversky loss with FP emphasis (alpha=0.7 means FPs weighted more than FNs)
            tversky_loss = calculate_tversky_loss(frame_probs, targets, alpha=0.8, beta=0.2)
            total_tversky_loss += tversky_loss
        
        # Simplified boundary loss - only if we have boundaries
        if start_targets[b].sum() > 0 or end_targets[b].sum() > 0:
            start_loss = nn.functional.binary_cross_entropy_with_logits(
                start_preds[b], start_targets[b], 
                pos_weight=torch.tensor(10.0).to(frame_preds.device)
            )
            end_loss = nn.functional.binary_cross_entropy_with_logits(
                end_preds[b], end_targets[b],
                pos_weight=torch.tensor(10.0).to(frame_preds.device)
            )
            total_segment_loss += (start_loss + end_loss)
    
    # Average losses
    avg_iou_loss = total_iou_loss / batch_size
    avg_focal_loss = total_focal_loss / batch_size
    avg_segment_loss = total_segment_loss / batch_size if total_segment_loss > 0 else 0
    
    if use_precision_focus:
        avg_precision_loss = total_precision_loss / batch_size
        avg_fp_penalty = total_fp_penalty / batch_size
        avg_tversky_loss = total_tversky_loss / batch_size
        avg_weighted_bce_loss = total_weighted_bce_loss / batch_size
        avg_hard_negative_loss = total_hard_negative_loss / batch_size if hard_negative_ratio > 0 else 0.0
        
        # Combine losses with strong emphasis on precision and avoiding false positives
        # Tversky already includes IoU-like behavior but with FP emphasis
        # We combine it with explicit precision terms for maximum FP control
        combined_loss = (
            0.22 * avg_tversky_loss +       # Primary segmentation with strong FP emphasis
            0.18 * avg_precision_loss +     # Explicit precision optimization
            0.22 * avg_fp_penalty +         # Direct FP penalization
            0.10 * avg_weighted_bce_loss +  # Imbalance-aware frame classification
            0.06 * avg_focal_loss +         # Class imbalance handling
            0.08 * avg_segment_loss +       # Boundary detection
            0.04 * avg_iou_loss +           # Basic overlap signal
            0.10 * avg_hard_negative_loss   # Strong focus on hardest negatives
        )
        
        return combined_loss
    else:
        # Original loss combination if precision focus is disabled
        return 0.7 * avg_iou_loss + 0.2 * avg_focal_loss + 0.1 * avg_segment_loss
