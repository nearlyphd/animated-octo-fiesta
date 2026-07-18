"""hto — landmark-detection utilities for HTO long-leg radiographs.

Subpackages:
    hto.data        — dataset construction + image preprocessing
    hto.metrics     — masked heatmap loss, coordinate decode, MSE/PCK metrics
    hto.geometry    — letterbox-inverse transform + Miniaci correction-angle geometry
    hto.stats       — ICC agreement statistics
    hto.sensitivity — differentiable correction angle, landmark Jacobian, precision budget
"""

__all__ = ["data", "metrics", "geometry", "stats", "sensitivity"]
