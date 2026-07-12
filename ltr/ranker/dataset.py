"""Build ranker training data from LMSYS conversations (Track B).

The ranker learns to order requests by output length, so each training example
is (prompt, observed_output_length). For ListMLE the relevance is ``-length``
(shorter output ⇒ higher relevance ⇒ should run first, i.e. SJF). Examples are
partitioned into fixed-size *lists*, since ListMLE is a listwise loss.

Pure transforms over :class:`bench.datasets.Request` objects — CPU-testable,
no torch.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.datasets import Request


@dataclass(frozen=True)
class LengthExample:
    """One ranker training example."""

    prompt: str
    output_length: int


def examples_from_requests(requests: Sequence[Request]) -> list[LengthExample]:
    """Keep requests that carry a positive reference (assistant) output length."""
    out: list[LengthExample] = []
    for r in requests:
        if r.n_reference_tokens is not None and r.n_reference_tokens > 0:
            out.append(LengthExample(prompt=r.prompt, output_length=int(r.n_reference_tokens)))
    return out


def examples_from_labels_file(path: str | Path) -> list[LengthExample]:
    """Load target-model-sampled labels (from ltr.ranker.synthesize) as examples.

    Expects a JSON list of ``{"prompt": ..., "output_length": ...}`` — the real
    fix for the ranker (labels are the served model's own generation lengths).
    """
    data = json.loads(Path(path).read_text())
    return [
        LengthExample(prompt=d["prompt"], output_length=int(d["output_length"]))
        for d in data
        if int(d.get("output_length", 0)) > 0
    ]


def relevance_from_lengths(lengths: Sequence[int]) -> np.ndarray:
    """ListMLE relevance: shorter output ⇒ higher relevance (SJF)."""
    return -np.asarray(lengths, dtype=float)


def make_lists(
    examples: Sequence[LengthExample], list_size: int, *, seed: int = 0, drop_last: bool = True
) -> list[list[LengthExample]]:
    """Shuffle and partition examples into fixed-size lists for ListMLE.

    With ``drop_last`` (default) a trailing short list is dropped so every list
    has exactly ``list_size`` items.
    """
    if list_size < 2:
        raise ValueError("list_size must be >= 2 for a ranking loss")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))
    lists: list[list[LengthExample]] = []
    for start in range(0, len(idx), list_size):
        chunk = idx[start : start + list_size]
        if drop_last and len(chunk) < list_size:
            break
        lists.append([examples[i] for i in chunk])
    return lists
