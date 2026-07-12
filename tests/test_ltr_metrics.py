"""CPU tests for ranking-quality metrics (Track B).

Covers README §8: ranking-metric computation on a known permutation.
Pure NumPy/SciPy — no torch, always runs in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from ltr.ranker.ranking_metrics import _kendall_tau_manual, kendall_tau, pairwise_accuracy


def test_kendall_tau_perfect_agreement():
    # predicted scores order items the same way as the true key
    assert kendall_tau([4, 3, 2, 1], [40, 30, 20, 10]) == pytest.approx(1.0)


def test_kendall_tau_reversed():
    assert kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_kendall_tau_single_item_is_safe():
    assert kendall_tau([1.0], [5.0]) == 0.0


def test_pairwise_accuracy_perfect_and_reversed():
    assert pairwise_accuracy([4, 3, 2, 1], [40, 30, 20, 10]) == 1.0
    assert pairwise_accuracy([1, 2, 3, 4], [40, 30, 20, 10]) == 0.0


def test_pairwise_accuracy_partial():
    # swap the top two predictions → 5 of 6 pairs still correct
    acc = pairwise_accuracy([3, 4, 2, 1], [40, 30, 20, 10])
    assert acc == pytest.approx(5 / 6)


def test_manual_kendall_matches_scipy_on_random():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20)
    b = rng.normal(size=20)
    assert _kendall_tau_manual(a, b) == pytest.approx(kendall_tau(a, b), abs=1e-9)
