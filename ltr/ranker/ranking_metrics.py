"""Ranking-quality metrics for the LTR ranker (Track B).

Pure NumPy/SciPy (no torch), so these are always CPU-testable. Kendall's tau is
the ranking metric the prior work reports; we also expose pairwise accuracy
(fraction of request pairs whose predicted order matches the true order).

Important: the prior paper notes Kendall's tau vs the *predicted rank* is not
the same as alignment with *realised latency* — we keep both this rank metric
and (later, in bench.metrics) a tau-vs-actual-latency check.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def kendall_tau(pred_scores: Sequence[float], true_key: Sequence[float]) -> float:
    """Kendall's tau-b between predicted scores and the true ordering key.

    Both should order items the same way (higher-first). Returns tau in
    [-1, 1]; 1.0 = identical order, -1.0 = reversed. Falls back to a manual
    computation if SciPy is unavailable.
    """
    a = np.asarray(pred_scores, dtype=float)
    b = np.asarray(true_key, dtype=float)
    if a.size < 2:
        return 0.0
    try:
        from scipy.stats import kendalltau

        tau, _ = kendalltau(a, b)
        return 0.0 if np.isnan(tau) else float(tau)
    except Exception:
        return _kendall_tau_manual(a, b)


def _kendall_tau_manual(a: np.ndarray, b: np.ndarray) -> float:
    n = a.size
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = np.sign(a[i] - a[j])
            sb = np.sign(b[i] - b[j])
            if sa == 0 or sb == 0:
                continue
            if sa == sb:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return 0.0 if total == 0 else (concordant - discordant) / total


def pairwise_accuracy(pred_scores: Sequence[float], true_key: Sequence[float]) -> float:
    """Fraction of item pairs the prediction orders the same way as the truth.

    Ties in either sequence are excluded from the denominator.
    """
    a = np.asarray(pred_scores, dtype=float)
    b = np.asarray(true_key, dtype=float)
    n = a.size
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa = np.sign(a[i] - a[j])
            sb = np.sign(b[i] - b[j])
            if sa == 0 or sb == 0:
                continue
            total += 1
            if sa == sb:
                correct += 1
    return 1.0 if total == 0 else correct / total
