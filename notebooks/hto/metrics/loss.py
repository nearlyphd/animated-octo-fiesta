"""Masked heatmap loss for landmark heatmap regression."""
import torch


def masked_heatmap_loss(pred_hms, target_hms, visibility):
    """Per-channel MSE loss masked to visible keypoints."""
    mask        = visibility.unsqueeze(-1).unsqueeze(-1)
    masked_diff = (pred_hms - target_hms) ** 2 * mask
    H, W        = pred_hms.shape[-2], pred_hms.shape[-1]
    return masked_diff.sum() / (mask.sum().clamp(min=1) * H * W)
