"""LMSYS-Chat-1M loader + prompt shaping (Track C).

Loads ``lmsys/lmsys-chat-1m`` (HF, gated), extracts a single-turn request from
each conversation, applies the model chat template, and samples two disjoint
subsets (in-distribution + held-out) to mirror the prior LTR paper's setup.

The *transforms* (single-turn extraction, template application, disjoint
subset sampling, token-length stats) are pure functions over plain Python
lists, so they are fully CPU-testable on a tiny fixture with a fake tokenizer
— no network, no gated access, no GPU. Only ``load_lmsys_chat_1m`` and the
real tokenizer touch the network.

A ``synthetic`` source is also provided so the load generator and metrics can
be smoke-tested end-to-end before gated dataset access is granted.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# Public dataset ids (both gated on HF).
LMSYS_CHAT_1M = "lmsys/lmsys-chat-1m"

Message = dict[str, str]  # OpenAI chat format: {"role": ..., "content": ...}


class Tokenizer(Protocol):
    """Minimal tokenizer surface the harness relies on.

    Both a real ``transformers`` tokenizer and the test fake satisfy this.
    """

    def apply_chat_template(
        self, messages: Sequence[Message], *, tokenize: bool, add_generation_prompt: bool
    ) -> str | list[int]: ...

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class Request:
    """One benchmark request derived from a source conversation.

    ``prompt`` is the chat-template-applied string (for ``/v1/completions``);
    ``messages`` is the raw chat list (for ``/v1/chat/completions``). Both are
    kept so the load generator can drive either endpoint.
    """

    id: str
    messages: list[Message]
    prompt: str
    n_prompt_tokens: int | None = None
    reference_output: str | None = None
    n_reference_tokens: int | None = None
    priority: int | None = None  # LTR scheduling priority (B1); lower = served first
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure transforms (CPU-testable)
# --------------------------------------------------------------------------- #
def extract_single_turn(conversation: Sequence[Message]) -> tuple[list[Message], str | None]:
    """Return ([first user message], first following assistant reply | None).

    Mirrors the ShareGPT/LMSYS serving-benchmark convention: prompt = first
    human turn, reference = first assistant turn (used later to derive an
    expected output length). Returns ``([], None)`` if there is no user turn.
    """
    prompt_msgs: list[Message] = []
    reference: str | None = None
    seen_user = False
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not seen_user:
            if role == "user":
                prompt_msgs = [{"role": "user", "content": content}]
                seen_user = True
            # skip any leading assistant/system turns
        else:
            if role == "assistant":
                reference = content
                break
    return prompt_msgs, reference


def apply_template(messages: Sequence[Message], tokenizer: Tokenizer) -> str:
    """Apply the chat template deterministically, adding a generation prompt."""
    out = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return out if isinstance(out, str) else str(out)


def _count_tokens(text: str | None, tokenizer: Tokenizer) -> int | None:
    if text is None:
        return None
    return len(tokenizer.encode(text))


def build_requests(
    conversations: Sequence[Sequence[Message]],
    tokenizer: Tokenizer,
    *,
    id_prefix: str = "req",
    min_prompt_chars: int = 1,
) -> list[Request]:
    """Turn raw conversations into templated, token-counted :class:`Request`s.

    Conversations without a user turn (or with an empty prompt) are skipped.
    """
    requests: list[Request] = []
    for i, conv in enumerate(conversations):
        msgs, reference = extract_single_turn(conv)
        if not msgs or len(msgs[0]["content"]) < min_prompt_chars:
            continue
        prompt = apply_template(msgs, tokenizer)
        requests.append(
            Request(
                id=f"{id_prefix}-{i}",
                messages=msgs,
                prompt=prompt,
                n_prompt_tokens=_count_tokens(prompt, tokenizer),
                reference_output=reference,
                n_reference_tokens=_count_tokens(reference, tokenizer),
            )
        )
    return requests


def sample_disjoint_subsets(
    n_total: int, n_in: int, n_held: int, *, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Deterministically sample two **disjoint** index subsets.

    A single permutation is split, so the in-distribution and held-out index
    lists never overlap by construction.
    """
    if n_in + n_held > n_total:
        raise ValueError(
            f"n_in + n_held ({n_in + n_held}) exceeds n_total ({n_total})"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    in_idx = sorted(int(x) for x in perm[:n_in])
    held_idx = sorted(int(x) for x in perm[n_in : n_in + n_held])
    return in_idx, held_idx


def token_length_stats(requests: Sequence[Request]) -> dict[str, dict[str, float]]:
    """Prompt/reference token-length summary (mean/p50/p90/max/min)."""

    def _stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "min": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
            "min": float(arr.min()),
        }

    prompt = [r.n_prompt_tokens for r in requests if r.n_prompt_tokens is not None]
    ref = [r.n_reference_tokens for r in requests if r.n_reference_tokens is not None]
    return {"prompt_tokens": _stats(prompt), "reference_tokens": _stats(ref)}


# --------------------------------------------------------------------------- #
# Synthetic source (no download) — smoke-tests the harness end to end
# --------------------------------------------------------------------------- #
def synthetic_conversations(
    n: int, *, seed: int = 0, min_words: int = 8, max_words: int = 128
) -> list[list[Message]]:
    """Random single-turn conversations for smoke-testing without a download."""
    rng = np.random.default_rng(seed)
    vocab = [
        "explain", "summarize", "the", "quantum", "cache", "latency", "please",
        "write", "code", "for", "a", "poisson", "process", "and", "attention",
        "kv", "scheduler", "gpu", "memory", "token", "prompt", "model", "vllm",
    ]
    convs: list[list[Message]] = []
    for _ in range(n):
        k = int(rng.integers(min_words, max_words + 1))
        words = rng.choice(vocab, size=k)
        prompt = " ".join(words.tolist()) + "?"
        convs.append([{"role": "user", "content": prompt}])
    return convs


_LONGCTX_UNIT = (
    "The key-value cache stores the keys and values of every previous token so "
    "that each newly generated token can attend to the entire history without "
    "recomputing it from scratch at every step. "
)  # ~35 Llama-3.1 tokens per copy


def longctx_conversations(
    n: int, *, seed: int = 0, target_tokens: int = 1024
) -> list[list[Message]]:
    """``n`` identical single-turn prompts padded to ~``target_tokens`` tokens.

    Controlled-length workload for the swap-vs-recompute long-context probe
    (`serving/grid_longctx.sh`): a neutral paragraph is repeated to the target
    context length, then a short question. Length is fixed across requests so the
    context at preemption is a *controlled* variable (content is deterministic by
    design, so ``seed`` is unused). ``target_tokens`` comes from ``LONGCTX_TOKENS``.
    """
    reps = max(1, target_tokens // 35)
    prompt = _LONGCTX_UNIT * reps + "\nBriefly, in one sentence, what does the cache store?"
    _ = seed
    return [[{"role": "user", "content": prompt}] for _ in range(n)]


# --------------------------------------------------------------------------- #
# Network / gated paths (real runs)
# --------------------------------------------------------------------------- #
def load_lmsys_chat_1m(
    n: int, *, split: str = "train", seed: int = 0, language: str | None = "English"
) -> list[list[Message]]:
    """Stream ``lmsys/lmsys-chat-1m`` and return ``n`` conversations.

    Gated on HF — requires an accepted license and ``HF_TOKEN``/login. Imported
    lazily so the module stays importable (and testable) without ``datasets``.
    """
    from datasets import load_dataset  # lazy: heavy + only needed for real runs

    ds = load_dataset(LMSYS_CHAT_1M, split=split, streaming=True)
    rng = np.random.default_rng(seed)
    # Reservoir-free simple skip: shuffle a streaming buffer for light mixing.
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    convs: list[list[Message]] = []
    for row in ds:
        if language is not None and row.get("language") != language:
            continue
        conv = row.get("conversation") or row.get("messages")
        if not conv:
            continue
        convs.append([{"role": m["role"], "content": m["content"]} for m in conv])
        if len(convs) >= n:
            break
    _ = rng  # reserved for future deterministic sampling knobs
    return convs


def load_tokenizer(model: str) -> Tokenizer:
    """Load a real chat tokenizer (lazy import of transformers)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def iter_requests(
    n: int,
    *,
    seed: int = 0,
    source: str = "lmsys",
    model: str | None = None,
    tokenizer: Tokenizer | None = None,
    language: str | None = "English",
) -> Iterator[Request]:
    """Yield ``n`` benchmark requests from ``lmsys`` or ``synthetic`` source.

    ``tokenizer`` may be injected (tests / reuse); otherwise ``model`` is
    loaded via transformers. The synthetic source needs no download and is
    used for pre-gated-access smoke tests.
    """
    if tokenizer is None:
        if model is None:
            raise ValueError("provide either `tokenizer` or `model`")
        tokenizer = load_tokenizer(model)

    if source == "synthetic":
        convs = synthetic_conversations(n, seed=seed)
    elif source == "longctx":
        import os

        convs = longctx_conversations(
            n, seed=seed, target_tokens=int(os.environ.get("LONGCTX_TOKENS", "1024"))
        )
    elif source == "lmsys":
        convs = load_lmsys_chat_1m(n, seed=seed, language=language)
    else:
        raise ValueError(f"unknown source: {source!r}")

    requests = build_requests(convs, tokenizer, id_prefix=source)
    yield from requests[:n]


if __name__ == "__main__":  # pragma: no cover - manual inspection
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the request loader.")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "lmsys"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("-n", type=int, default=8)
    args = ap.parse_args()

    reqs = list(iter_requests(args.n, source=args.source, model=args.model))
    stats = token_length_stats(reqs)
    print(f"loaded {len(reqs)} requests from {args.source}")
    print("prompt token stats:", stats["prompt_tokens"])
    for r in reqs[:3]:
        print(f"  [{r.id}] {r.n_prompt_tokens} tok :: {r.messages[0]['content'][:60]!r}")
