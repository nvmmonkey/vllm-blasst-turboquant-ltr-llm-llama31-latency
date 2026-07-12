"""CPU tests for ranker training-data construction (Track B)."""

from __future__ import annotations

import pytest

from bench.datasets import Request
from ltr.ranker.dataset import (
    LengthExample,
    examples_from_requests,
    make_lists,
    relevance_from_lengths,
)


def _req(rid: str, ref_tokens):
    return Request(
        id=rid, messages=[{"role": "user", "content": "q"}], prompt="q",
        n_reference_tokens=ref_tokens,
    )


def test_examples_filter_missing_or_zero_lengths():
    reqs = [_req("a", 10), _req("b", None), _req("c", 0), _req("d", 5)]
    ex = examples_from_requests(reqs)
    assert [e.output_length for e in ex] == [10, 5]
    assert all(isinstance(e, LengthExample) for e in ex)


def test_relevance_is_negative_length():
    # shorter output (5) must have the HIGHEST relevance (least negative)
    rel = relevance_from_lengths([10, 5, 20])
    assert list(rel) == [-10.0, -5.0, -20.0]
    assert rel.argmax() == 1


def test_make_lists_partitions_into_fixed_size_and_drops_remainder():
    ex = [LengthExample(prompt=f"p{i}", output_length=i) for i in range(10)]
    lists = make_lists(ex, 4, seed=0)  # 10 // 4 = 2 full lists; remainder of 2 dropped
    assert len(lists) == 2
    assert all(len(chunk) == 4 for chunk in lists)


def test_make_lists_is_deterministic_under_seed():
    ex = [LengthExample(prompt=f"p{i}", output_length=i) for i in range(12)]
    a = [[e.prompt for e in chunk] for chunk in make_lists(ex, 4, seed=1)]
    b = [[e.prompt for e in chunk] for chunk in make_lists(ex, 4, seed=1)]
    assert a == b


def test_make_lists_rejects_tiny_list_size():
    with pytest.raises(ValueError):
        make_lists([LengthExample("p", 1)], 1)


def test_examples_from_labels_file(tmp_path):
    import json

    from ltr.ranker.dataset import examples_from_labels_file

    p = tmp_path / "labels.json"
    p.write_text(json.dumps([
        {"prompt": "a", "output_length": 10},
        {"prompt": "b", "output_length": 0},   # zero -> filtered
        {"prompt": "c", "output_length": 5},
    ]))
    ex = examples_from_labels_file(p)
    assert [e.output_length for e in ex] == [10, 5]
    assert [e.prompt for e in ex] == ["a", "c"]
