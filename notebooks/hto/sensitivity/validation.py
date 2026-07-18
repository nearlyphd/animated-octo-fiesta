"""Correctness checks for the differentiable correction-angle geometry.

The Jacobian in :mod:`hto.sensitivity.jacobian` is only trustworthy if it agrees with
:func:`hto.geometry.evaluate_side_geometry`, the construction the detector is actually
evaluated against. These checks establish that, and are meant to be run as a gate before
any sensitivity analysis rather than as an afterthought.

Checks
------
1. **Parity** -- ``abs(miniaci_angle_signed)`` equals ``evaluate_side_geometry`` exactly.
2. **Autodiff** -- the autograd Jacobian equals central finite differences taken *on the
   NumPy reference*, so the derivative is validated against the shipped code rather than
   against another copy of itself.
3. **Delta-method** -- at a realistic 1.5 px perturbation, ``sum_i J_i . dp_i`` reproduces
   the exact angle change, bounding the linearisation error well inside the 1.63 deg
   clinical tolerance.
4. **Malleoli symmetry** -- the two malleoli enter ``g`` only through their midpoint, so
   their Jacobian rows must be identical. This is a property of the construction, not of
   the implementation, so it fails loudly if :data:`SIDE_KEYS` is ever mis-ordered.
5. **Channel order** -- :data:`SIDE_KEYS` matches ``GLOBAL_KEYPOINT_NAMES`` in the
   baseline notebook (optional; skipped when the notebook is not present).

Run standalone::

    python -m hto.sensitivity.validation

or from a notebook, against the real cohort::

    from hto.sensitivity import validate
    validate([h["points"] for h in hemispheres])
"""
import json
from pathlib import Path

import numpy as np

from ..geometry import evaluate_side_geometry
from .jacobian import (
    SIDE_KEYS,
    angle_and_jacobian,
    batched_angle_signed,
    miniaci_angle_signed,
    points_to_tensor,
)

__all__ = ["validate", "check_channel_order", "check_batched", "random_hemisphere"]

# Tolerances. Loose enough for float64 finite differences, tight enough to catch a real
# error: the delta-method bound is ~7000x smaller than the clinical tolerance.
TOL_PARITY = 1e-9      # deg
TOL_GRAD = 1e-5        # relative
TOL_LINEAR = 5e-3      # deg, at 1.5 px perturbation
TOL_SYMMETRY = 1e-6    # relative

_BASELINE_NOTEBOOK = "hto_unet_baseline.ipynb"


def random_hemisphere(rng):
    """A plausible long-leg hemisphere in original-image pixels (~2860 x 8000 canvas)."""
    x0 = 1400.0 + rng.normal(0, 120)
    y_hip = 900 + rng.normal(0, 90)
    y_knee = 4200 + rng.normal(0, 150)
    y_ankle = 7400 + rng.normal(0, 150)
    plateau_w = 420.0 + rng.normal(0, 35)
    x_knee = x0 + rng.normal(0, 70)
    malleolar_w = 330.0 + rng.normal(0, 30)
    x_ankle = x0 + rng.normal(0, 420)
    return {
        "femur_head":  np.array([x0, y_hip]),
        "knee_inner":  np.array([x_knee - plateau_w / 2, y_knee]),
        "ost_point":   np.array([x_knee + plateau_w * 0.30, y_knee + 300 + rng.normal(0, 40)]),
        "knee_outer":  np.array([x_knee + plateau_w / 2, y_knee]),
        "ankle_inner": np.array([x_ankle - malleolar_w / 2, y_ankle + rng.normal(0, 10)]),
        "ankle_outer": np.array([x_ankle + malleolar_w / 2, y_ankle + rng.normal(0, 10)]),
    }


def _fd_jacobian_reference(points, h=1e-4):
    """Central finite differences of the *NumPy reference* angle, ``(6, 2)``."""
    fd = np.zeros((len(SIDE_KEYS), 2))
    for i, name in enumerate(SIDE_KEYS):
        for j in range(2):
            plus = {k: np.array(v, dtype=float) for k, v in points.items()}
            minus = {k: np.array(v, dtype=float) for k, v in points.items()}
            plus[name][j] += h
            minus[name][j] -= h
            fd[i, j] = (evaluate_side_geometry(plus)[0]
                        - evaluate_side_geometry(minus)[0]) / (2 * h)
    return fd


def check_batched(n=2000, seed=0, rtol_deg=1e-9):
    """Assert :func:`batched_angle_signed` matches :func:`miniaci_angle_signed` exactly.

    The batched NumPy angle and the per-sample torch angle are separate transcriptions of
    the same formula; this gates them against drift. Perturbs random hemispheres well
    beyond the in-domain range so the check also covers the wrapped-difference regime.

    Returns the observed maximum absolute discrepancy in degrees.
    """
    rng = np.random.default_rng(seed)
    base = [random_hemisphere(rng) for _ in range(n)]
    P = np.stack([np.stack([b[k] for k in SIDE_KEYS]) for b in base])   # (n, 6, 2)
    P += rng.normal(0, 60, size=P.shape)                               # large perturbations

    batched = batched_angle_signed(P)
    per_sample = np.array([
        float(miniaci_angle_signed({k: P[i, j] for j, k in enumerate(SIDE_KEYS)}))
        for i in range(n)
    ])
    max_abs = float(np.abs(batched - per_sample).max())
    assert max_abs < rtol_deg, \
        f"batched vs per-sample angle mismatch: {max_abs:.3e} deg"
    return max_abs


def check_channel_order(notebook_path=None):
    """Assert :data:`SIDE_KEYS` matches ``GLOBAL_KEYPOINT_NAMES`` in the baseline notebook.

    Returns ``True`` if checked, ``False`` if the notebook could not be found.
    """
    if notebook_path is None:
        here = Path(__file__).resolve()
        candidates = [p / _BASELINE_NOTEBOOK for p in (here.parents[2], Path.cwd())]
        notebook_path = next((c for c in candidates if c.exists()), None)
        if notebook_path is None:
            return False
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        return False

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    if "GLOBAL_KEYPOINT_NAMES" not in src:
        return False

    block = src.split("GLOBAL_KEYPOINT_NAMES", 1)[1][:400]
    missing = [k for k in SIDE_KEYS if f'"{k}_lh"' not in block]
    assert not missing, f"landmarks absent from GLOBAL_KEYPOINT_NAMES: {missing}"

    positions = [block.index(f'"{k}_lh"') for k in SIDE_KEYS]
    assert positions == sorted(positions), (
        f"SIDE_KEYS order disagrees with GLOBAL_KEYPOINT_NAMES: {SIDE_KEYS}")
    return True


def validate(points=None, n_synthetic=500, fd_sample=50, seed=1234, verbose=True):
    """Run every check and return the observed worst-case errors.

    Parameters
    ----------
    points : list[dict], optional
        Real hemispheres to check. When omitted, ``n_synthetic`` random ones are used.
    fd_sample : int
        How many hemispheres get the (more expensive) finite-difference comparison.
        Parity and symmetry are checked on all of them.

    Returns
    -------
    dict
        ``max_parity``, ``max_grad``, ``max_linear``, ``max_symmetry``, ``n_checked``,
        ``channel_order_checked``, and the correction-angle range covered.

    Raises
    ------
    AssertionError
        On any failure -- callers should let this propagate.
    """
    rng = np.random.default_rng(seed)
    if points is None:
        points = [random_hemisphere(rng) for _ in range(n_synthetic)]
        source = f"{len(points)} synthetic hemispheres"
    else:
        points = list(points)
        source = f"{len(points)} supplied hemispheres"

    assert points, "no hemispheres to validate"

    i_in, i_out = SIDE_KEYS.index("ankle_inner"), SIDE_KEYS.index("ankle_outer")
    max_parity = max_grad = max_linear = max_symmetry = 0.0
    alphas = []

    for idx, pts in enumerate(points):
        a_signed, a_unsigned, J = angle_and_jacobian(pts)
        alphas.append(a_unsigned)

        # 1. parity with the reference construction
        max_parity = max(max_parity, abs(evaluate_side_geometry(pts)[0] - a_unsigned))

        # 4. structural identity of the two malleoli
        max_symmetry = max(max_symmetry,
                           np.abs(J[i_in] - J[i_out]).max()
                           / max(np.linalg.norm(J[i_in]), 1e-12))

        if idx < fd_sample:
            # 2. autodiff against finite differences on the NumPy reference
            fd = _fd_jacobian_reference(pts)
            scale = max(np.abs(fd).max(), 1e-12)
            max_grad = max(max_grad, np.abs(np.sign(a_signed) * J - fd).max() / scale)

            # 3. delta-method at a realistic perturbation
            delta = {n: rng.normal(0, 1.5, size=2) for n in SIDE_KEYS}
            moved = {n: np.asarray(pts[n], dtype=float) + delta[n] for n in SIDE_KEYS}
            exact = float(miniaci_angle_signed(points_to_tensor(moved))) - a_signed
            predicted = sum(float(J[i] @ delta[n]) for i, n in enumerate(SIDE_KEYS))
            max_linear = max(max_linear, abs(exact - predicted))

    assert max_parity < TOL_PARITY, \
        f"torch/numpy angle mismatch: {max_parity:.3e} deg"
    assert max_grad < TOL_GRAD, \
        f"autodiff disagrees with finite differences: {max_grad:.3e} rel"
    assert max_linear < TOL_LINEAR, \
        f"delta-method error {max_linear:.3e} deg too large at 1.5 px"
    assert max_symmetry < TOL_SYMMETRY, \
        f"malleoli rows differ by {max_symmetry:.3e} rel -- SIDE_KEYS may be mis-ordered"

    order_checked = check_channel_order()
    max_batched = check_batched()
    alphas = np.array(alphas)
    n_fd = min(fd_sample, len(points))

    if verbose:
        print(f"geometry validation | {source}")
        print(f"  alpha mean {alphas.mean():.2f} deg, "
              f"range {alphas.min():.2f}-{alphas.max():.2f}\n")
        print(f"  [1] parity vs evaluate_side_geometry, n={len(points):<4d} "
              f"{max_parity:.2e} deg   OK")
        print(f"  [2] autodiff vs finite differences,   n={n_fd:<4d} {max_grad:.2e} rel   OK")
        print(f"  [3] delta-method @ 1.5 px jitter,     n={n_fd:<4d} {max_linear:.2e} deg   OK")
        print(f"  [4] malleoli symmetry,                n={len(points):<4d} "
              f"{max_symmetry:.2e} rel   OK")
        print(f"  [5] SIDE_KEYS vs GLOBAL_KEYPOINT_NAMES     "
              f"{'OK' if order_checked else 'skipped (notebook not found)'}")
        print(f"  [6] batched vs per-sample angle           {max_batched:.2e} deg   OK")

    return {
        "max_parity": float(max_parity),
        "max_grad": float(max_grad),
        "max_linear": float(max_linear),
        "max_symmetry": float(max_symmetry),
        "max_batched": float(max_batched),
        "n_checked": len(points),
        "n_finite_difference": n_fd,
        "channel_order_checked": bool(order_checked),
        "alpha_deg": {"mean": float(alphas.mean()),
                      "min": float(alphas.min()),
                      "max": float(alphas.max())},
    }


if __name__ == "__main__":
    validate()
    print("\nAll checks pass.")
