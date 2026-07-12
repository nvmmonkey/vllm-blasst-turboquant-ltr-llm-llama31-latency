"""Track C — shared benchmark harness.

Engine-agnostic load generation, dataset shaping, metrics computation, and
plotting for every serving config (B0, B1, and the later C-tiers). Talks to
vLLM's OpenAI-compatible endpoint, so the same harness measures all configs.
"""

__all__ = ["datasets", "loadgen", "metrics", "run_sweep", "plots", "accuracy"]
