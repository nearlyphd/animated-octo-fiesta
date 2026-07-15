"""Surgical geometry: letterbox-inverse transform and the Miniaci correction-angle construction."""
import math

import numpy as np


def map_global_to_orig(kp_final, orig_size, target_size=512):
    """Invert the letterbox transform back to original image coordinates."""
    orig_w, orig_h = orig_size
    scale    = min(target_size / orig_w, target_size / orig_h)
    pad_left = (target_size - int(orig_w * scale)) // 2
    pad_top  = (target_size - int(orig_h * scale)) // 2
    return np.array([(kp_final[0] - pad_left) / scale,
                     (kp_final[1] - pad_top)  / scale])


def calculate_intersection(p1, p2, target_y):
    """X-coordinate where line (p1 → p2) crosses y = *target_y*."""
    if p2[0] == p1[0]: return p1[0]
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    if m == 0:
        return p1[0] if abs(target_y - p1[1]) < 1e-9 else float("nan")
    return (target_y - p1[1]) / m + p1[0]


def evaluate_side_geometry(points):
    """Compute the Miniaci correction angle Alpha for one leg hemisphere.

    Parameters
    ----------
    points : dict with keys femur_head, knee_inner, ost_point,
             knee_outer, ankle_inner, ankle_outer (2-element arrays)

    Returns
    -------
    alpha, fujisawa, ankle_c, target_at_ankle
    """
    ankle_c  = (points["ankle_inner"] + points["ankle_outer"]) / 2.0
    fujisawa = points["knee_inner"] + 0.625 * (points["knee_outer"] - points["knee_inner"])

    tx              = calculate_intersection(points["femur_head"], fujisawa, ankle_c[1])
    target_at_ankle = np.array([tx, ankle_c[1]])

    v_orig   = ankle_c         - points["ost_point"]
    v_target = target_at_ankle - points["ost_point"]
    raw      = abs(math.atan2(v_orig[1], v_orig[0]) - math.atan2(v_target[1], v_target[0]))
    alpha    = min(raw, 2 * math.pi - raw) * 180.0 / math.pi
    return alpha, fujisawa, ankle_c, target_at_ankle
