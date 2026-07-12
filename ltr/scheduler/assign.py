"""Assign vLLM priorities to requests from the trained LTR ranker (Track B).

The client-side half of B1: score each request's prompt with the OPT-125M
ranker, map scores → priorities (highest score = shortest predicted output =
priority 0 = served first), and stamp the priority on each request. The
priorities then ride to a server launched with ``--scheduling-policy priority``
— so LTR ordering happens with **no engine patching**.

The pure mapping (:func:`assign_priorities_from_scores`) is CPU-testable; the
ranker inference (:func:`score_requests`) is the torch/GPU integration path.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from bench.datasets import Request
from ltr.scheduler.priority import scores_to_priorities


def assign_priorities_from_scores(
    requests: Sequence[Request], scores: Sequence[float]
) -> list[Request]:
    """Return copies of ``requests`` with ``priority`` set from ``scores``.

    Highest score → priority 0 (served first). Pure — no torch.
    """
    priorities = scores_to_priorities(scores)
    return [
        dataclasses.replace(r, priority=int(p))
        for r, p in zip(requests, priorities, strict=True)
    ]


def score_requests(  # pragma: no cover - torch/GPU integration path
    requests: Sequence[Request],
    ranker_dir: str,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 512,
) -> list[float]:
    """Score request prompts with the trained ranker (higher = shorter output)."""
    import json
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer

    from ltr.ranker.model import build_ranker
    from ltr.ranker.train import tokenize_prompts

    d = Path(ranker_dir)
    meta = json.loads((d / "ranker_meta.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(d)
    ranker = build_ranker(base=meta["base"])
    ranker.load_state_dict(torch.load(d / "ranker.pt", map_location=device))
    ranker.to(device).eval()

    prompts = [r.prompt for r in requests]
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            ids, attn = tokenize_prompts(tokenizer, prompts[start : start + batch_size], max_length=max_length)
            scores.extend(ranker(ids.to(device), attn.to(device)).reshape(-1).tolist())
    return scores


def assign_priorities(
    requests: Sequence[Request], ranker_dir: str, **kwargs
) -> list[Request]:  # pragma: no cover - torch/GPU integration path
    """Score with the ranker and stamp priorities (convenience wrapper)."""
    scores = score_requests(requests, ranker_dir, **kwargs)
    return assign_priorities_from_scores(requests, scores)
