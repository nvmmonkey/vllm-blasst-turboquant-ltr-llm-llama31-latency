"""CPU tests for the LMSYS-Chat-1M loader transforms (Track C).

Covers the three properties the README §6 calls out: template application is
deterministic, sampled subsets are disjoint, and token-length stats are
correct on a tiny fixture. Uses a fake tokenizer so nothing is downloaded.
"""

from __future__ import annotations

import pytest

from bench.datasets import (
    Request,
    build_requests,
    extract_single_turn,
    iter_requests,
    sample_disjoint_subsets,
    token_length_stats,
)


class FakeTokenizer:
    """Deterministic stand-in: chat template = tagged concat, encode = word split."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        return text if not tokenize else self.encode(text)

    def encode(self, text):
        return text.split()


# --------------------------------------------------------------------------- #
def test_extract_single_turn_basic():
    conv = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi back"},
        {"role": "user", "content": "second turn"},
    ]
    msgs, ref = extract_single_turn(conv)
    assert msgs == [{"role": "user", "content": "hello there"}]
    assert ref == "hi back"


def test_extract_single_turn_skips_leading_assistant():
    conv = [
        {"role": "assistant", "content": "unsolicited"},
        {"role": "user", "content": "the real prompt"},
        {"role": "assistant", "content": "the real reply"},
    ]
    msgs, ref = extract_single_turn(conv)
    assert msgs[0]["content"] == "the real prompt"
    assert ref == "the real reply"


def test_extract_single_turn_no_user():
    msgs, ref = extract_single_turn([{"role": "assistant", "content": "x"}])
    assert msgs == []
    assert ref is None


def test_apply_template_is_deterministic():
    tok = FakeTokenizer()
    convs = [[{"role": "user", "content": "explain kv cache"}]]
    r1 = build_requests(convs, tok)
    r2 = build_requests(convs, tok)
    assert r1[0].prompt == r2[0].prompt
    assert r1[0].prompt.endswith("<|assistant|>")
    assert "explain kv cache" in r1[0].prompt


def test_build_requests_skips_conversations_without_user_turn():
    tok = FakeTokenizer()
    convs = [
        [{"role": "user", "content": "keep me"}],
        [{"role": "assistant", "content": "no user turn -> dropped"}],
        [{"role": "user", "content": ""}],  # empty prompt -> dropped
    ]
    reqs = build_requests(convs, tok)
    assert len(reqs) == 1
    assert reqs[0].messages[0]["content"] == "keep me"


def test_build_requests_counts_tokens():
    tok = FakeTokenizer()
    convs = [[{"role": "user", "content": "one two three four five"}]]
    reqs = build_requests(convs, tok)
    # prompt = "<|user|>one two three four five\n<|assistant|>"; str.split() treats
    # "\n" as a separator, so tokens = [<|user|>one, two, three, four, five, <|assistant|>] = 6
    assert reqs[0].n_prompt_tokens == 6


def test_sample_disjoint_subsets_are_disjoint_and_sized():
    in_idx, held_idx = sample_disjoint_subsets(100, 30, 20, seed=0)
    assert len(in_idx) == 30
    assert len(held_idx) == 20
    assert set(in_idx).isdisjoint(set(held_idx))
    assert all(0 <= i < 100 for i in in_idx + held_idx)


def test_sample_disjoint_subsets_is_deterministic():
    a = sample_disjoint_subsets(100, 30, 20, seed=7)
    b = sample_disjoint_subsets(100, 30, 20, seed=7)
    c = sample_disjoint_subsets(100, 30, 20, seed=8)
    assert a == b
    assert a != c


def test_sample_disjoint_subsets_rejects_oversized_request():
    with pytest.raises(ValueError):
        sample_disjoint_subsets(10, 8, 5, seed=0)


def test_token_length_stats_on_known_fixture():
    reqs = [
        Request(id=f"r{i}", messages=[], prompt="", n_prompt_tokens=t)
        for i, t in enumerate([10, 20, 30, 40])
    ]
    stats = token_length_stats(reqs)["prompt_tokens"]
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(25.0)
    assert stats["p50"] == pytest.approx(25.0)
    assert stats["p90"] == pytest.approx(37.0)  # linear interp: 30 + 0.7*(40-30)
    assert stats["max"] == 40.0
    assert stats["min"] == 10.0


def test_token_length_stats_empty_is_safe():
    stats = token_length_stats([])["prompt_tokens"]
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


def test_iter_requests_synthetic_source_needs_no_download():
    tok = FakeTokenizer()
    reqs = list(iter_requests(12, seed=1, source="synthetic", tokenizer=tok))
    assert len(reqs) == 12
    assert all(r.n_prompt_tokens and r.n_prompt_tokens > 0 for r in reqs)
    # deterministic under fixed seed
    reqs2 = list(iter_requests(12, seed=1, source="synthetic", tokenizer=tok))
    assert [r.prompt for r in reqs] == [r.prompt for r in reqs2]


def test_iter_requests_unknown_source_raises():
    with pytest.raises(ValueError):
        list(iter_requests(1, source="nope", tokenizer=FakeTokenizer()))
