"""CPU smoke tests for the ranker train/eval scripts (Track B).

Full training needs a GPU + dataset access, so here we only pin that the
modules import and that the tokenisation helper shapes its output correctly
(with a fake tokenizer — no download).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ltr.ranker.train import tokenize_prompts  # noqa: E402


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, prompts, *, return_tensors, padding, truncation, max_length):
        n = len(prompts)
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }


def test_tokenize_prompts_returns_ids_and_mask():
    ids, mask = tokenize_prompts(_FakeTokenizer(), ["a", "b"], max_length=8)
    assert ids.shape == (2, 4)
    assert mask.shape == (2, 4)


def test_tokenize_prompts_sets_pad_token_when_missing():
    tok = _FakeTokenizer()
    tok.pad_token = None
    tokenize_prompts(tok, ["hi"], max_length=8)
    assert tok.pad_token == tok.eos_token  # helper fills a missing pad token


def test_train_and_eval_modules_import():
    import ltr.ranker.eval as eval_mod
    import ltr.ranker.train as train_mod

    assert hasattr(train_mod, "train")
    assert hasattr(eval_mod, "evaluate")
