"""Synthesize target-model output-length labels for the ranker (Track B).

The audit's #1 fix: the prior paper trains the ranker on lengths *sampled from
the target model* (its ``synthesize_dataset.py``), not on LMSYS reference
replies written by other models. This module reproduces that: send prompts to a
**running target-model server** (natural EOS), capture each request's actual
generated length, and save ``(prompt, messages, output_length)`` records that
``ltr.ranker.train --labels-file`` trains on.

    # with a Llama-8B server up on :8000
    python -m ltr.ranker.synthesize --model meta-llama/Llama-3.1-8B-Instruct \
        --n 3000 --out results/ranker/lengths_llama8b.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


async def synthesize_lengths(  # pragma: no cover - needs a live server
    requests,
    *,
    base_url: str,
    model: str,
    request_rate: float,
    max_tokens: int,
) -> list[dict]:
    """Generate each request on the target model; return real output lengths."""
    from bench.loadgen import SamplingConfig, run_http_load

    # natural EOS (no ignore_eos) + a generous cap so most finish on their own
    sampling = SamplingConfig(max_tokens=max_tokens, temperature=0.0)
    records = await run_http_load(
        requests, base_url=base_url, model=model, request_rate=request_rate, sampling=sampling
    )
    by_id = {r.id: r for r in records}
    out: list[dict] = []
    for req in requests:
        rec = by_id.get(req.id)
        if rec and rec.success and rec.n_output_tokens > 0:
            out.append(
                {
                    "prompt": req.prompt,
                    "messages": req.messages,
                    "output_length": int(rec.n_output_tokens),
                }
            )
    return out


def main() -> None:  # pragma: no cover - CLI (needs a live server)
    ap = argparse.ArgumentParser(description="Sample target-model output lengths for ranker labels.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--rate", type=float, default=20.0, help="req/s for generation")
    ap.add_argument("--max-tokens", type=int, default=1024, help="generous cap; natural EOS")
    ap.add_argument("--source", default="lmsys", choices=["lmsys", "synthetic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from bench.datasets import iter_requests

    requests = list(iter_requests(args.n, seed=args.seed, source=args.source, model=args.model))
    labels = asyncio.run(
        synthesize_lengths(
            requests, base_url=args.base_url, model=args.model,
            request_rate=args.rate, max_tokens=args.max_tokens,
        )
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels, indent=2))
    lengths = [x["output_length"] for x in labels]
    mean = sum(lengths) / len(lengths) if lengths else 0
    print(f"wrote {len(labels)} target-sampled labels to {out} (mean length {mean:.0f} tok)")


if __name__ == "__main__":  # pragma: no cover
    main()
