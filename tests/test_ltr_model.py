"""CPU tests for the OPT ranker (Track B).

Covers README §8: ranker I/O shapes. Uses a tiny randomly-initialised OPT so no
weights are downloaded. torch/transformers guarded via importorskip.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ltr.ranker.model import build_ranker, masked_mean_pool  # noqa: E402


def test_masked_mean_pool_excludes_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]])  # (1, 3, 2)
    mask = torch.tensor([[1, 1, 0]])  # last token is padding
    pooled = masked_mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))  # mean of first two rows


def test_masked_mean_pool_no_valid_tokens_is_zero():
    hidden = torch.ones(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.long)
    assert torch.allclose(masked_mean_pool(hidden, mask), torch.zeros(1, 2))


def test_opt_ranker_outputs_one_score_per_prompt():
    from transformers import OPTConfig, OPTModel

    cfg = OPTConfig(
        vocab_size=100, hidden_size=32, word_embed_proj_dim=32,
        num_hidden_layers=2, num_attention_heads=2, ffn_dim=64, max_position_embeddings=64,
    )
    ranker = build_ranker(backbone=OPTModel(cfg))
    input_ids = torch.randint(0, 100, (4, 10))
    attention_mask = torch.ones(4, 10, dtype=torch.long)
    scores = ranker(input_ids, attention_mask)
    assert scores.shape == (4,)  # one scalar score per prompt


def test_opt_ranker_handles_fp16_backbone():
    # regression: facebook/opt-125m ships fp16 weights; a Half backbone with an
    # fp32 head must NOT raise "mat1 and mat2 must have the same dtype".
    from transformers import OPTConfig, OPTModel

    cfg = OPTConfig(
        vocab_size=100, hidden_size=32, word_embed_proj_dim=32,
        num_hidden_layers=2, num_attention_heads=2, ffn_dim=64, max_position_embeddings=64,
    )
    ranker = build_ranker(backbone=OPTModel(cfg).half())  # fp16 backbone, fp32 head
    scores = ranker(torch.randint(0, 100, (3, 8)), torch.ones(3, 8, dtype=torch.long))
    assert scores.shape == (3,)
