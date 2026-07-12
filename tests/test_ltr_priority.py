"""CPU tests for ranker-score → vLLM-priority mapping (Track B)."""

from __future__ import annotations

from ltr.scheduler.priority import score_to_priority_int, scores_to_priorities


def test_highest_score_gets_priority_zero():
    # idx0=10 (best) -> 0, idx2=8 -> 1, idx1=5 -> 2  (lower value = higher priority)
    assert scores_to_priorities([10.0, 5.0, 8.0]) == [0, 2, 1]


def test_priorities_are_a_permutation_of_range():
    prio = scores_to_priorities([3.1, -1.0, 2.5, 9.9, 0.0])
    assert sorted(prio) == [0, 1, 2, 3, 4]


def test_ties_break_by_input_order_stably():
    assert scores_to_priorities([1.0, 1.0, 1.0]) == [0, 1, 2]


def test_empty_scores():
    assert scores_to_priorities([]) == []


def test_single_request_mapping_is_monotonic():
    # higher score -> lower (better) integer priority
    assert score_to_priority_int(2.0) < score_to_priority_int(1.0) < score_to_priority_int(0.0)
