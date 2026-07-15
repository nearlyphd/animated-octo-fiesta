"""Localization metrics: keypoint MSE and PCK."""
import torch


def calculate_mse(preds, gts):
    """Mean Squared Error (px\u00b2) over all valid keypoints."""
    valid_mask = gts[..., 0] >= 0
    if not torch.any(valid_mask): return 0.0
    return torch.mean(torch.sum((preds - gts) ** 2, dim=-1)[valid_mask]).item()


def calculate_pck(preds, gts, threshold=0.05, normalize_by=512):
    """Percentage of Correct Keypoints at *threshold* \u00d7 *normalize_by* px."""
    valid_mask = gts[..., 0] >= 0
    if not torch.any(valid_mask): return 0.0
    correct = torch.norm(preds - gts, dim=-1) < (threshold * normalize_by)
    return torch.mean(correct[valid_mask].float()).item()
