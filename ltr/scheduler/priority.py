"""Map ranker scores → vLLM request priorities (Track B, crash-safe v1 path).

This is how B1 reaches modern vLLM v1 **without patching the scheduler or block
manager** (the two files that crashed the prior attempt, §IV-D of the paper):
run the server with ``--scheduling-policy priority`` and attach a per-request
``priority`` derived from the OPT-125M ranker's score.

vLLM priority convention: **lower value = higher priority** (served first). We
want the highest-scoring request (shortest predicted output) to run first, so
the top score maps to priority 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def scores_to_priorities(scores: Sequence[float]) -> list[int]:
    """Rank scores so the highest score → priority 0 (served first).

    Ties break by input order (stable). Returns integer priorities in
    ``[0, n)`` suitable for vLLM's ``priority`` request field.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return []
    # order[0] is the index of the highest score; its rank (priority) is 0.
    order = np.argsort(-arr, kind="stable")
    priorities = np.empty(arr.size, dtype=int)
    priorities[order] = np.arange(arr.size)
    return priorities.tolist()


def score_to_priority_int(score: float, *, scale: float = 1000.0) -> int:
    """Magnitude-preserving single-request mapping: higher score → lower int.

    Useful when requests arrive online (no full list to rank). Negated and
    scaled so a higher score yields a smaller (higher-priority) integer.
    """
    return int(round(-score * scale))
