"""
The paired comparison used for every Delta: two conditions scored on the same clips.
"""

import numpy as np


def paired(a, b, n_boot=5000, seed=0):
    """
    Per-clip differences a - b. Returns the mean and median difference, a bootstrap 95%
    confidence interval for the mean, the Wilcoxon signed-rank p-value, and the share of
    clips where a > b.
    """
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    try:
        _, p = wilcoxon(d)
    except ValueError:
        p = np.nan
    return {"n": len(d), "mean_delta": float(d.mean()),
            "median_delta": float(np.median(d)), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "wilcoxon_p": float(p),
            "frac_positive": float((d > 0).mean())}


def by_recording(values, recordings, n_boot=5000, seed=0, level=0.95):
    """
    Mean of per-sample `values` with a bootstrap confidence interval (95% by default) that
    resamples whole recordings, because samples from one recording are not independent.
    """
    v = np.asarray(values, dtype=float)
    keys, inv = np.unique(np.asarray(recordings), return_inverse=True)
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    return {"n": len(v), "recordings": len(keys), "mean": float(v.mean()),
            "ci_lo": float(np.percentile(boot, 50 * (1 - level))),
            "ci_hi": float(np.percentile(boot, 50 * (1 + level)))}
