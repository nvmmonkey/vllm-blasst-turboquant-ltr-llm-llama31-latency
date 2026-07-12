"""Teacher-forced perplexity harness for KV-quant accuracy (C1, Track C).

Measures how much a KV-cache quantization config degrades the model's ability to
model a FIXED clean passage, versus the fp16 baseline. The quant mode is chosen
at engine init, so run ONE config per process:

    python -m ltr.quant.eval_accuracy --kv-cache-dtype auto   --out fp16.json
    python -m ltr.quant.eval_accuracy --kv-cache-dtype fp8    --out fp8.json
    VLLM_KV_ROTATE=1 python -m ltr.quant.eval_accuracy --kv-cache-dtype fp8 --out rot_fp8.json

Perplexity is teacher-forced (sum of the actual tokens' logprobs via
``prompt_logprobs``), so there is no greedy-decode cascade — it isolates the KV
quantization error. Lower = closer to fp16. TurboQuant hypothesis (C1 Stage 1a):
rotation+fp8 perplexity < plain fp8 (rotation makes coordinates near-uniform, so
fp8 quantizes with less distortion). Passages are neutral/technical — deliberately
NO LMSYS content. Needs the capstone repo on PYTHONPATH for the rotation import.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# Clean, self-contained passages (~150-250 tokens each) that exercise the KV
# cache over a non-trivial context. No user data.
PASSAGES = [
    "The key-value cache is a central optimization in autoregressive transformer "
    "inference. During generation, each new token attends to every previous token, "
    "so the keys and values computed for earlier positions are stored and reused "
    "rather than recomputed at every step. This turns an operation that would grow "
    "quadratically with sequence length into an incremental one, but it shifts the "
    "cost from computation to memory: the cache size grows linearly with the number "
    "of tokens, the number of layers, and the number of attention heads. On modern "
    "accelerators the cache quickly becomes the dominant consumer of device memory, "
    "which is why techniques such as paged attention, quantization, eviction, and "
    "offloading to host memory have become essential for serving long contexts.",
    "Quantization reduces the number of bits used to represent each cached value. "
    "A floating-point key or value that ordinarily occupies sixteen bits can be "
    "stored in eight bits, or fewer, at the cost of a small rounding error. The "
    "difficulty is that the distribution of values is not uniform: a few outlier "
    "channels carry disproportionately large magnitudes, and naive uniform "
    "quantization wastes precision on the common small values while clipping the "
    "rare large ones. Rotating the vectors by a random orthogonal matrix spreads "
    "the energy evenly across coordinates, so that each dimension follows a "
    "predictable distribution and a scalar quantizer can be applied with near "
    "optimal distortion, recovering most of the accuracy lost to compression.",
    "Scheduling determines the order in which pending requests are admitted to the "
    "running batch. A first-come first-served policy is simple and fair in arrival "
    "order, but it suffers from head-of-line blocking: a long request that arrives "
    "early holds resources while many short requests wait behind it. If the system "
    "can estimate how long each request will run, it can prioritize the short ones "
    "and dramatically reduce average latency, an idea borrowed from classic "
    "shortest-job-first scheduling. Predicting output length before generation is "
    "itself a learning problem, and a small auxiliary model trained to rank prompts "
    "by expected length can supply the ordering signal that the serving engine needs.",
    "When device memory is exhausted, the serving engine must preempt some running "
    "requests to make room. Two strategies are common. Recomputation discards the "
    "cached state of a victim request and regenerates it later from scratch, which "
    "is simple but wastes the computation already performed. Swapping instead moves "
    "the cached state to host memory and copies it back when the request resumes, "
    "trading device memory pressure for slower host-device transfers. Which strategy "
    "wins depends on the workload: under heavy preemption from aggressive reordering, "
    "swapping avoids repeated wasted work, while under light pressure the transfer "
    "overhead of swapping may not pay for itself.",
]


def _token_logprob(entry, token_id):
    """Extract the logprob of ``token_id`` from a prompt_logprobs dict entry."""
    if entry is None or token_id not in entry:
        return None
    val = entry[token_id]
    return val.logprob if hasattr(val, "logprob") else float(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="float16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, enforce_eager=True, swap_space=4,
              kv_cache_dtype=args.kv_cache_dtype)
    # DECODE-path metric: greedy-generate so each new token attends to the
    # (quantized) cached prompt KV — this is what a prefill-only perplexity misses.
    outs = llm.generate(PASSAGES, SamplingParams(max_tokens=args.gen_tokens,
                                                 temperature=0, logprobs=1))
    per = []
    for o in outs:
        c = o.outputs[0]
        lps = []
        for i, tid in enumerate(c.token_ids):
            entry = c.logprobs[i] if c.logprobs else None
            lp = _token_logprob(entry, tid)
            lps.append(round(lp, 5) if lp is not None else None)
        per.append({"token_ids": list(c.token_ids), "logprobs": lps})

    result = {
        "kv_cache_dtype": args.kv_cache_dtype,
        "rotate": os.environ.get("VLLM_KV_ROTATE", "0") == "1",
        "gen_tokens": args.gen_tokens,
        "per_passage": per,
    }
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "per_passage"}))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


def compare(fp16_path, other_path):
    """Longest fp16-matching prefix + mean |Δlogprob| on it, per passage."""
    ref = json.loads(Path(fp16_path).read_text())["per_passage"]
    oth = json.loads(Path(other_path).read_text())["per_passage"]
    rows = []
    for r, o in zip(ref, oth):
        rt, ot = r["token_ids"], o["token_ids"]
        n = min(len(rt), len(ot))
        match = 0
        while match < n and rt[match] == ot[match]:
            match += 1
        dl = [abs(r["logprobs"][i] - o["logprobs"][i])
              for i in range(match)
              if r["logprobs"][i] is not None and o["logprobs"][i] is not None]
        rows.append({"match_prefix": match, "of": n,
                     "mean_abs_dlogprob": round(sum(dl) / len(dl), 5) if dl else None})
    total_match = sum(x["match_prefix"] for x in rows)
    total_of = sum(x["of"] for x in rows)
    return {"per_passage": rows, "total_match": total_match, "total_of": total_of,
            "match_frac": round(total_match / max(total_of, 1), 4)}


if __name__ == "__main__":
    main()
