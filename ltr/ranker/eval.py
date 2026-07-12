"""Evaluate the LTR ranker's ranking quality on a held-out split (Track B).

Reports Kendall's τ and pairwise accuracy of the predicted scores against the
true output-length order (shorter = should rank first). Note (from the paper):
τ vs predicted *rank* is not the same as alignment with realised *latency*; the
latency-aligned check lives in the benchmark, not here.

    python -m ltr.ranker.eval --ranker results/ranker/opt125m-ltr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _predict_scores(ranker, tokenizer, prompts, *, device, batch_size, max_length):  # pragma: no cover
    import torch

    from ltr.ranker.train import tokenize_prompts

    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            input_ids, attn = tokenize_prompts(tokenizer, batch, max_length=max_length)
            out = ranker(input_ids.to(device), attn.to(device))
            scores.extend(out.reshape(-1).tolist())
    return scores


def evaluate(  # pragma: no cover - integration path (needs torch + data)
    *,
    ranker_dir: str,
    target_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    source: str = "lmsys",
    n: int = 1000,
    held_seed: int = 1,  # different split from training (seed 0)
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> dict:
    import torch
    from transformers import AutoTokenizer

    from bench.datasets import iter_requests
    from ltr.ranker.dataset import examples_from_requests
    from ltr.ranker.model import build_ranker
    from ltr.ranker.ranking_metrics import kendall_tau, pairwise_accuracy

    d = Path(ranker_dir)
    meta = json.loads((d / "ranker_meta.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(d)
    ranker = build_ranker(base=meta["base"])
    ranker.load_state_dict(torch.load(d / "ranker.pt", map_location=device))
    ranker.to(device).eval()

    requests = list(iter_requests(n, seed=held_seed, source=source, model=target_model))
    examples = examples_from_requests(requests)
    prompts = [e.prompt for e in examples]
    lengths = [e.output_length for e in examples]

    scores = _predict_scores(
        ranker, tokenizer, prompts, device=device, batch_size=batch_size, max_length=max_length
    )
    true_key = [-length for length in lengths]  # shorter output = higher relevance
    result = {
        "kendall_tau": kendall_tau(scores, true_key),
        "pairwise_accuracy": pairwise_accuracy(scores, true_key),
        "n": len(examples),
        "held_seed": held_seed,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Evaluate LTR ranker ranking quality.")
    ap.add_argument("--ranker", required=True, help="directory from ltr.ranker.train")
    ap.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--source", default="lmsys", choices=["lmsys", "synthetic"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--held-seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    evaluate(
        ranker_dir=args.ranker, target_model=args.target_model, source=args.source,
        n=args.n, held_seed=args.held_seed, device=args.device,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
