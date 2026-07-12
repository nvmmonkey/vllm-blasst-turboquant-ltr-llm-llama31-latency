"""OPT-125M output-length ranker (Track B).

A scalar scoring head over OPT's token representations. Given a prompt, it
predicts a score; higher score ⇒ shorter expected output ⇒ should run earlier
(SJF), matching the prior LTR work [11][12]. Trained with ListMLE
(:mod:`ltr.ranker.losses`).

The pooling is factored into :func:`masked_mean_pool` (pure torch, unit-tested
without any model); :func:`build_ranker` wires it to an OPT backbone. torch /
transformers are imported lazily so the rest of the harness stays import-light.
"""

from __future__ import annotations

DEFAULT_BASE = "facebook/opt-125m"


def masked_mean_pool(hidden, attention_mask):
    """Mean-pool ``hidden`` (B, T, H) over valid tokens given ``attention_mask``.

    Padding tokens (mask == 0) are excluded; a row with no valid token pools to 0.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)                     # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)                 # (B, 1)
    return summed / counts


_RANKER_CLS = None


def _ranker_cls():
    """Memoised OPTRanker class (built on first use so torch import is lazy)."""
    global _RANKER_CLS
    if _RANKER_CLS is not None:
        return _RANKER_CLS

    import torch.nn as nn

    class OPTRanker(nn.Module):
        """OPT backbone + linear scoring head → one score per prompt."""

        def __init__(self, base: str = DEFAULT_BASE, backbone=None):
            super().__init__()
            if backbone is None:
                from transformers import AutoModel

                # Many checkpoints (e.g. facebook/opt-125m) ship fp16 weights;
                # cast to fp32 so the fp32 scoring head matmul dtype matches.
                backbone = AutoModel.from_pretrained(base).float()
            self.backbone = backbone
            self.score_head = nn.Linear(backbone.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = masked_mean_pool(out.last_hidden_state, attention_mask)  # (B, H)
            # guard against a backbone that emits a different dtype than the head
            pooled = pooled.to(self.score_head.weight.dtype)
            return self.score_head(pooled).squeeze(-1)  # (B,)

    _RANKER_CLS = OPTRanker
    return _RANKER_CLS


def build_ranker(base: str = DEFAULT_BASE, backbone=None):
    """Construct an OPT ranker. Tests inject a tiny backbone to avoid a download."""
    return _ranker_cls()(base=base, backbone=backbone)
