"""Differentiable Miniaci correction angle and its landmark Jacobian.

The construction implemented in :func:`hto.geometry.evaluate_side_geometry` is transcribed
here in torch, so the correction angle is differentiable with respect to the six landmarks
of a limb hemisphere. First-order (delta-method) propagation of a landmark perturbation is

    dm  ~=  sum_i  J_i . dp_i ,        J_i = d(alpha)/d(p_i)  in  R^2

where ||J_i|| is the *influence* of landmark i, in degrees of correction angle per pixel of
displacement. J is the sensitivity object produced by Paper A / E1 and reused by Papers B
and D; nothing here is specific to E1.

A transcription, not a wrapper
-----------------------------
``evaluate_side_geometry`` computes the angle with ``math.atan2`` on NumPy arrays, which
autograd cannot traverse. :func:`miniaci_angle_signed` is therefore a *second, independent
implementation* of the same construction. That duplication is a real maintenance hazard --
the two can silently drift -- which is why :mod:`hto.sensitivity.validation` checks parity
against the reference and should be run as a gate before any sensitivity analysis.

``hto.geometry`` is deliberately left untouched: it defines the correction angle and
produced the baseline's published numbers. Making it backend-generic would remove the
duplication at the source, but at the cost of editing code the published results depend on.

Known deviations from the reference, all intentional:

1. **Signed output.** ``abs()`` recovers the reference's unsigned value exactly (see below).
2. **No degenerate-slope branches.** ``calculate_intersection`` guards ``p2[0] == p1[0]``
   and ``m == 0``. The first is reproduced exactly by the algebra here (``d[0] == 0`` gives
   ``tx == fh[0]``). The second -- femoral head at the same height as the Fujisawa point --
   would return ``nan`` there and diverges here, but ``d[1]`` is the hip-to-knee drop, so
   it is unreachable on real anatomy.
3. **Fujisawa fraction is a parameter** (:data:`FUJISAWA_PCT`), defaulting to the
   reference's hard-coded 0.625.
4. **Returns the angle only**, not the reference's
   ``(alpha, fujisawa, ankle_c, target_at_ankle)`` tuple.

Signed versus unsigned angle
----------------------------
:func:`miniaci_angle_signed` returns the **signed** angle; ``abs()`` recovers
``evaluate_side_geometry``'s unsigned value exactly. Differentiating the signed quantity
avoids the kink that ``abs`` / ``min(raw, 2*pi - raw)`` introduces at ``alpha = 0``. Since

    d|alpha|/dp = sign(alpha) * d(alpha)/dp

the magnitudes ||J_i|| are identical either way -- this is numerical hygiene, not a change
of quantity. :mod:`hto.sensitivity.validation` asserts the equivalence against the NumPy
reference.
"""
import math

import numpy as np
import torch

#: Landmark order of the rows of P and of the Jacobian.
#:
#: MUST match the per-hemisphere channel order of ``GLOBAL_KEYPOINT_NAMES`` in
#: ``hto_unet_baseline.ipynb`` (channels 0-5 are the ``_lh`` hemisphere, 6-11 ``_rh``),
#: and the keys ``evaluate_side_geometry`` expects.
#: :func:`hto.sensitivity.validation.check_channel_order` verifies this against the
#: baseline notebook, and the malleoli-symmetry check catches a mis-ordering numerically.
SIDE_KEYS = ["femur_head", "knee_inner", "ost_point",
             "knee_outer", "ankle_inner", "ankle_outer"]

#: Channel base -> hemisphere name, following the baseline's ``_lh`` / ``_rh`` suffixes.
#: These are *image* hemispheres: the dataset assigns base 0 to the annotation with the
#: smaller representative x, not to an anatomical side.
HEMISPHERE_BASES = [(0, "left"), (6, "right")]

#: Display labels, so figures across papers name the landmarks identically.
LANDMARK_LABELS = {
    "femur_head":  "Femoral head centre",
    "knee_inner":  "Medial tibial plateau",
    "ost_point":   "Osteotomy hinge",
    "knee_outer":  "Lateral tibial plateau",
    "ankle_inner": "Medial malleolus",
    "ankle_outer": "Lateral malleolus",
}

#: Fujisawa point as a fraction of the medial-to-lateral plateau width.
FUJISAWA_PCT = 0.625

#: Clinical correction-angle tolerance in degrees (Jiang et al. 2022).
TOLERANCE_DEG = 1.63

DEFAULT_DTYPE = torch.float64

__all__ = [
    "SIDE_KEYS", "HEMISPHERE_BASES", "LANDMARK_LABELS", "FUJISAWA_PCT", "TOLERANCE_DEG",
    "points_to_tensor", "miniaci_angle_signed", "batched_angle_signed", "wrap_deg",
    "angle_and_jacobian", "cohort_jacobians", "delta_method", "precision_budget",
]


def points_to_tensor(points, requires_grad=False, dtype=DEFAULT_DTYPE):
    """Stack a landmark dict into a ``(6, 2)`` tensor in :data:`SIDE_KEYS` order.

    Parameters
    ----------
    points : dict[str, array-like]
        Keys are :data:`SIDE_KEYS`; values are ``(x, y)`` in original-image pixels.
    requires_grad : bool
        Set on the returned leaf tensor so a Jacobian can be taken with respect to it.
    """
    P = torch.tensor(np.stack([np.asarray(points[n], dtype=float) for n in SIDE_KEYS]),
                     dtype=dtype)
    return P.requires_grad_(requires_grad)


def miniaci_angle_signed(P, fujisawa_pct=FUJISAWA_PCT):
    """Signed Miniaci/Fujisawa correction angle in degrees, differentiable in ``P``.

    Parameters
    ----------
    P : torch.Tensor or dict
        ``(6, 2)`` tensor in :data:`SIDE_KEYS` order, or a landmark dict.

    Returns
    -------
    torch.Tensor
        Scalar, signed, degrees. ``abs()`` equals ``evaluate_side_geometry(points)[0]``.
    """
    if isinstance(P, dict):
        P = points_to_tensor(P)
    fh, ki, ost, ko, ai, ao = P[0], P[1], P[2], P[3], P[4], P[5]

    ankle_c = (ai + ao) / 2.0
    fujisawa = ki + fujisawa_pct * (ko - ki)

    # Target ankle position: the line femoral-head -> Fujisawa, evaluated at the ankle's
    # height. Algebraically identical to the reference's (target_y - y1)/m + x1, but
    # written so a near-vertical line (dx -> 0) is exact rather than a division by a
    # vanishing slope. d[1] is the hip-to-knee drop, so it is never near zero.
    d = fujisawa - fh
    tx = fh[0] + d[0] * (ankle_c[1] - fh[1]) / d[1]
    target_at_ankle = torch.stack([tx, ankle_c[1]])

    v_orig = ankle_c - ost
    v_target = target_at_ankle - ost
    diff = torch.atan2(v_orig[1], v_orig[0]) - torch.atan2(v_target[1], v_target[0])
    wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))      # -> (-pi, pi], smooth
    return wrapped * (180.0 / math.pi)


def batched_angle_signed(P, fujisawa_pct=FUJISAWA_PCT):
    """Vectorised, NumPy-only version of :func:`miniaci_angle_signed`.

    :func:`miniaci_angle_signed` differentiates one hemisphere at a time; a large
    perturbation study (E3, Monte-Carlo in Papers B/D) needs the *exact* angle for
    millions of configurations at once, where autodiff is not required. This computes the
    identical quantity over an arbitrary batch. It carries no gradient -- pair it with
    :func:`angle_and_jacobian` for J, which is evaluated once per hemisphere at the
    unperturbed landmarks.

    The formula is line-for-line the same as :func:`miniaci_angle_signed`;
    :func:`hto.sensitivity.validation.check_batched` asserts parity against it so the two
    cannot drift.

    Parameters
    ----------
    P : numpy.ndarray
        ``(..., 6, 2)`` landmarks in :data:`SIDE_KEYS` order, any leading batch shape.

    Returns
    -------
    numpy.ndarray
        ``(...)`` signed correction angle in degrees.
    """
    P = np.asarray(P, dtype=float)
    fh, ki, ost, ko, ai, ao = (P[..., i, :] for i in range(6))

    ankle_c = (ai + ao) / 2.0
    fujisawa = ki + fujisawa_pct * (ko - ki)

    d = fujisawa - fh
    tx = fh[..., 0] + d[..., 0] * (ankle_c[..., 1] - fh[..., 1]) / d[..., 1]
    target_at_ankle = np.stack([tx, ankle_c[..., 1]], axis=-1)

    v_orig = ankle_c - ost
    v_target = target_at_ankle - ost
    diff = (np.arctan2(v_orig[..., 1], v_orig[..., 0])
            - np.arctan2(v_target[..., 1], v_target[..., 0]))
    wrapped = np.arctan2(np.sin(diff), np.cos(diff))
    return wrapped * (180.0 / math.pi)


def wrap_deg(delta):
    """Wrap an angle difference into ``(-180, 180]`` degrees.

    A correction-angle *change* under a large landmark perturbation can wrap past the
    +-180 deg branch cut, at which point a raw ``a1 - a0`` overstates the true change by
    ~360 deg. Wrapping gives the principal change, which is what "how much did the angle
    move" means. For the small displacements of in-domain operation this is a no-op.
    """
    return (np.asarray(delta) + 180.0) % 360.0 - 180.0


def angle_and_jacobian(points, fujisawa_pct=FUJISAWA_PCT, dtype=DEFAULT_DTYPE):
    """Correction angle and its landmark Jacobian for one hemisphere.

    Parameters
    ----------
    points : dict[str, array-like]
        Landmarks in original-image pixels, keyed by :data:`SIDE_KEYS`.

    Returns
    -------
    alpha_signed : float
        Signed correction angle, degrees.
    alpha : float
        Unsigned correction angle, degrees -- equals ``evaluate_side_geometry(points)[0]``.
    J : numpy.ndarray
        ``(6, 2)`` Jacobian in :data:`SIDE_KEYS` row order, degrees per pixel.
    """
    P = points_to_tensor(points, requires_grad=True, dtype=dtype)
    alpha_signed = miniaci_angle_signed(P, fujisawa_pct)
    (grad,) = torch.autograd.grad(alpha_signed, P)
    a = float(alpha_signed)
    return a, abs(a), grad.detach().numpy()


def cohort_jacobians(points_iter, fujisawa_pct=FUJISAWA_PCT):
    """Vectorised :func:`angle_and_jacobian` over a cohort.

    Parameters
    ----------
    points_iter : iterable of dict
        One landmark dict per hemisphere.

    Returns
    -------
    J : numpy.ndarray
        ``(n, 6, 2)`` Jacobians, degrees per pixel.
    alpha_signed : numpy.ndarray
        ``(n,)`` signed correction angles, degrees.
    alpha : numpy.ndarray
        ``(n,)`` unsigned correction angles, degrees.
    """
    points_list = list(points_iter)
    J = np.zeros((len(points_list), len(SIDE_KEYS), 2))
    signed = np.zeros(len(points_list))
    for i, pts in enumerate(points_list):
        signed[i], _, J[i] = angle_and_jacobian(pts, fujisawa_pct)
    return J, signed, np.abs(signed)


def delta_method(J, delta):
    """First-order predicted correction-angle change for a landmark perturbation.

    Parameters
    ----------
    J : numpy.ndarray
        ``(6, 2)`` Jacobian from :func:`angle_and_jacobian`.
    delta : dict[str, array-like] or numpy.ndarray
        Per-landmark displacement in pixels, keyed by :data:`SIDE_KEYS` or as ``(6, 2)``.

    Returns
    -------
    float
        Predicted ``dm`` in degrees.
    """
    if isinstance(delta, dict):
        delta = np.stack([np.asarray(delta[n], dtype=float) for n in SIDE_KEYS])
    return float((np.asarray(J) * np.asarray(delta)).sum())


def precision_budget(J_norm, tolerance_deg=TOLERANCE_DEG, z=1.0, k=None):
    """Per-landmark localisation budget implied by a tolerance on the correction angle.

    Treating the ``k`` landmark errors as independent and isotropic with per-landmark
    scale ``sigma_i``, ``dm ~ N(0, sum_i ||J_i||^2 sigma_i^2)``. Allocating the budget
    equally across landmarks and requiring ``z * sd(dm) <= tolerance`` gives

        sigma_i <= tolerance / (z * sqrt(k) * ||J_i||)

    Parameters
    ----------
    J_norm : array-like
        Per-landmark ``||J_i||`` in degrees per pixel.
    z : float
        1.0 reproduces the form in Paper A section 3.2; use 1.96 for a 95% budget.
    k : int, optional
        Number of landmarks sharing the budget. Defaults to ``len(J_norm)``.

    Returns
    -------
    numpy.ndarray
        Allowed displacement per landmark, in the same length unit as ``J_norm``'s
        denominator (pixels).
    """
    J_norm = np.asarray(J_norm, dtype=float)
    k = len(J_norm) if k is None else k
    return tolerance_deg / (z * math.sqrt(k) * J_norm)
