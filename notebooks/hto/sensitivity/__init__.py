"""hto.sensitivity — measurement-sensitivity analysis of the correction angle.

Differentiates the Miniaci/Fujisawa construction with respect to its six landmarks, so
landmark-localisation error can be propagated into correction-angle error:

    dm  ~=  sum_i  J_i . dp_i ,        J_i = d(alpha)/d(p_i)  in  R^2

``||J_i||`` is the influence of landmark i in degrees per pixel. J is the sensitivity
object produced by Paper A / E1 and reused by Papers B and D.

Public API:
    SIDE_KEYS, HEMISPHERE_BASES, LANDMARK_LABELS, FUJISAWA_PCT, TOLERANCE_DEG
    points_to_tensor, miniaci_angle_signed, angle_and_jacobian, cohort_jacobians
    delta_method, precision_budget
    validate, check_channel_order

Validate before analysing -- the checks are cheap and gate everything downstream::

    from hto.sensitivity import validate, cohort_jacobians
    validate([h["points"] for h in hemispheres])
    J, alpha_signed, alpha = cohort_jacobians(h["points"] for h in hemispheres)
"""
from .jacobian import (
    FUJISAWA_PCT,
    HEMISPHERE_BASES,
    LANDMARK_LABELS,
    SIDE_KEYS,
    TOLERANCE_DEG,
    angle_and_jacobian,
    cohort_jacobians,
    delta_method,
    miniaci_angle_signed,
    points_to_tensor,
    precision_budget,
)
from .validation import check_channel_order, validate

__all__ = [
    "SIDE_KEYS",
    "HEMISPHERE_BASES",
    "LANDMARK_LABELS",
    "FUJISAWA_PCT",
    "TOLERANCE_DEG",
    "points_to_tensor",
    "miniaci_angle_signed",
    "angle_and_jacobian",
    "cohort_jacobians",
    "delta_method",
    "precision_budget",
    "validate",
    "check_channel_order",
]
