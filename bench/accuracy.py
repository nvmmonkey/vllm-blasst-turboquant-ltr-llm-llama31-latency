"""Accuracy evaluation (perplexity + task scores) — STUB.

B0 and B1 change *scheduling*, not model outputs, so accuracy is identical to
vanilla vLLM and there is nothing to measure yet. This module is scaffolded now
and implemented alongside the C-tiers, where **quantization and sparsity** are
the only techniques that can move accuracy (offloading and lossless spec
decoding cannot). See README §9 / §11.

TODO(C-tiers): implement
  * perplexity on a held-out corpus (compare each C-tier vs B1), and
  * a small task battery (e.g. a few LongBench / MMLU subsets),
  reusing the model + dataset loaders in bench.datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

# Marker so callers/tests can assert the accuracy stage is planned-but-not-live.
IMPLEMENTED = False


@dataclass
class AccuracyResult:
    """Placeholder schema the real implementation will populate."""

    config: str
    perplexity: float | None = None
    task_scores: dict[str, float] | None = None


def compute_perplexity(*_args, **_kwargs) -> float:
    """Not implemented until the C-tiers (quant/sparsity) land."""
    raise NotImplementedError(
        "accuracy eval (perplexity) lands with the C-tiers; B0/B1 do not alter accuracy"
    )


def evaluate(*_args, **_kwargs) -> AccuracyResult:
    """Not implemented until the C-tiers (quant/sparsity) land."""
    raise NotImplementedError(
        "accuracy eval (task battery) lands with the C-tiers; B0/B1 do not alter accuracy"
    )
