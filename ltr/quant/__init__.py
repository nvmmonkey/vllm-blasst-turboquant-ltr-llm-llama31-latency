"""KV-cache quantization for the C-tiers (C1 = TurboQuant rotation + KVmix).

See docs/C_TIERS.md. This package holds the committed, unit-tested math; the
vLLM 0.4.1 fork (git-ignored) imports it at the attention-backend hook points.
"""
