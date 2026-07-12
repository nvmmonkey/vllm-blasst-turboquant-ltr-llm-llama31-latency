"""Placeholder test for the accuracy stub (Track C).

B0/B1 don't alter accuracy, so this only pins the stub's contract until the
C-tiers implement it (README §9).
"""

from __future__ import annotations

import pytest

from bench import accuracy


def test_accuracy_is_marked_not_implemented():
    assert accuracy.IMPLEMENTED is False


def test_compute_perplexity_raises_until_c_tiers():
    with pytest.raises(NotImplementedError):
        accuracy.compute_perplexity()


def test_evaluate_raises_until_c_tiers():
    with pytest.raises(NotImplementedError):
        accuracy.evaluate()


def test_accuracy_result_schema():
    r = accuracy.AccuracyResult(config="b1")
    assert r.config == "b1"
    assert r.perplexity is None
    assert r.task_scores is None
