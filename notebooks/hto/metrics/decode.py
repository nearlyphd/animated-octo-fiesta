"""Heatmap → coordinate decoding (per-channel argmax)."""
import torch


def extract_coordinates(heatmaps, scale_factor=4.0):
    """
    Extracts the (x, y) coordinates of the maximum value in each heatmap.

    Args:
        heatmaps (torch.Tensor): Tensor of shape (Batch, Num_Keypoints, Height, Width)
        scale_factor (float): Multiplier to project the coordinates back to the
                              original image size. Default is 4.0 for a 1/4 scale heatmap.

    Returns:
        torch.Tensor: Coordinates of shape (Batch, Num_Keypoints, 2) formatted as [x, y]
    """
    batch_size, num_keypoints, h, w = heatmaps.shape

    # 1. Flatten the spatial dimensions (H, W) into a single 1D vector per keypoint
    heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)

    # 2. Find the index of the maximum value in that 1D vector
    max_vals, max_indices = torch.max(heatmaps_flat, dim=-1)

    # 3. Convert the 1D flat index back into 2D grid coordinates (y, x)
    y_coords = (max_indices // w).float()
    x_coords = (max_indices % w).float()

    # 4. Scale the coordinates back up to match the original input image resolution
    y_coords = y_coords * scale_factor
    x_coords = x_coords * scale_factor

    # 5. Stack the x and y coordinates together -> (Batch, Num_Keypoints, 2)
    coordinates = torch.stack([x_coords, y_coords], dim=-1)

    return coordinates
