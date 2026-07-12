"""CPU tests for LTR priority assignment (Track B)."""

from __future__ import annotations

from bench.datasets import Request
from ltr.scheduler.assign import assign_priorities_from_scores


def _reqs(n: int) -> list[Request]:
    return [Request(id=f"r{i}", messages=[], prompt="p") for i in range(n)]


def test_highest_score_gets_priority_zero():
    # scores: r1=9 (best/shortest) -> 0, r2=5 -> 1, r0=1 -> 2
    out = assign_priorities_from_scores(_reqs(3), [1.0, 9.0, 5.0])
    assert {r.id: r.priority for r in out} == {"r0": 2, "r1": 0, "r2": 1}


def test_descending_scores_map_to_ascending_priorities():
    out = assign_priorities_from_scores(_reqs(4), [4.0, 3.0, 2.0, 1.0])
    assert [r.id for r in out] == ["r0", "r1", "r2", "r3"]  # order preserved
    assert [r.priority for r in out] == [0, 1, 2, 3]


def test_originals_unchanged_and_copies_returned():
    reqs = _reqs(2)
    out = assign_priorities_from_scores(reqs, [1.0, 2.0])
    assert reqs[0].priority is None  # frozen originals untouched
    assert out[0].priority is not None and out is not reqs
