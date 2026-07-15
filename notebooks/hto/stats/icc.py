"""Intraclass correlation (ICC) for correction-angle agreement.

compute_icc uses pingouin for the exact 95% CI and p-value when installed, else falls
back to the manual ICC(2,1) point estimate in icc21_manual (with a SciPy F-test p-value).
"""
import numpy as np
import pandas as pd


def icc21_manual(ratings, return_pvalue=False):
    """ICC(2,1): two-way random, absolute agreement, single measurement.

    ratings: array (n_targets, k_raters). Point estimate (Shrout & Fleiss 1979).

    If ``return_pvalue=True`` returns ``(icc, pval)``, where ``pval`` is the F-test
    p-value for H0: ICC = 0 (F = MS_row / MS_error, df = n-1, (n-1)(k-1)); it needs
    SciPy and is ``nan`` when SciPy is unavailable. Default returns just ``icc``.
    """
    ratings = np.asarray(ratings, dtype=float)
    n, k = ratings.shape
    grand     = ratings.mean()
    row_means = ratings.mean(axis=1, keepdims=True)
    col_means = ratings.mean(axis=0, keepdims=True)
    ss_total = ((ratings - grand) ** 2).sum()
    ss_row   = k * ((row_means - grand) ** 2).sum()
    ss_col   = n * ((col_means - grand) ** 2).sum()
    ss_err   = ss_total - ss_row - ss_col
    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_row + (k - 1) * ms_err + (k / n) * (ms_col - ms_err)
    icc = float((ms_row - ms_err) / denom)
    if not return_pvalue:
        return icc
    try:
        from scipy.stats import f as _f
        pval = float(_f.sf(ms_row / ms_err, n - 1, (n - 1) * (k - 1)))
    except Exception:
        pval = float("nan")
    return icc, pval


def compute_icc(gt, pred):
    """(icc, ci_low, ci_high, pval, method).

    pingouin gives the exact 95% CI and p-value if installed; otherwise a manual
    ICC(2,1) point estimate is used with a SciPy F-test p-value (CI is nan). The two
    'raters' are the mean-observer GT and the automatic method; ``pval`` tests H0: ICC = 0.
    """
    gt   = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(gt)
    try:
        import pingouin as pg
        long = pd.DataFrame({
            "target": list(range(n)) * 2,
            "rater":  ["mean_observer"] * n + ["auto"] * n,
            "angle":  np.concatenate([gt, pred]),
        })
        res = pg.intraclass_corr(data=long, targets="target", raters="rater", ratings="angle")
        row = res.loc[res["Type"].isin(["ICC2", "ICC(A,1)"])].iloc[0]
        ci_col = "CI95%" if "CI95%" in res.columns else "CI95"
        ci = row[ci_col]
        pval = float(row["pval"]) if "pval" in res.columns else float("nan")
        return float(row["ICC"]), float(ci[0]), float(ci[1]), pval, "pingouin ICC(2,1)"
    except ImportError:
        icc, pval = icc21_manual(np.column_stack([gt, pred]), return_pvalue=True)
        return (icc, np.nan, np.nan, pval,
                "manual ICC(2,1) -- pip install pingouin for the 95% CI")
