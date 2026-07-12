"""LTR ranker head-to-head — ours vs LLM.pdf's predictor, on BOTH distributions.

Answers "should we use LLM.pdf's ranker?" FAIRLY: each ranker is evaluated on BOTH
its own home distribution and the other's, so home-field advantage is visible rather
than hidden. Metric = Kendall's |τ| of predicted score vs true output-length order.

  * LMSYS  = held-out LMSYS split, true length = reference-reply length (ours' home)
  * ShareGPT = the paper's own trace, true length = tokens in `generated` (paper's home)

Our rankers (opt125m-ltr / opt125m-real) load our `ranker.pt` format; LLM.pdf's is an
OPTForSequenceClassification (HF safetensors, trained on ShareGPT for Llama-3-8B).

    python -m ltr.ranker.headtohead --n 1000 --out results/summaries/ltr_headtohead.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER = ("ltr/vendor/vllm-ltr/benchmarks/MODEL/results/"
         "opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/finetuned")
SHAREGPT = "ltr/vendor/vllm-ltr/benchmarks/llama3-8b-sharegpt-test-t1-s0-8192.jsonl"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _score_ours(ranker_dir, prompts, device, max_length, batch_size=32):
    import torch
    from transformers import AutoTokenizer

    from ltr.ranker.model import build_ranker
    from ltr.ranker.train import tokenize_prompts

    d = Path(ranker_dir)
    meta = json.loads((d / "ranker_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(d)
    r = build_ranker(base=meta["base"])
    r.load_state_dict(torch.load(d / "ranker.pt", map_location=device))
    r.to(device).eval()
    out: list[float] = []
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            ids, attn = tokenize_prompts(tok, prompts[s : s + batch_size], max_length=max_length)
            out.extend(r(ids.to(device), attn.to(device)).reshape(-1).tolist())
    return out


def _score_paper(model_dir, prompts, device, max_length, batch_size=32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    m = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out: list[float] = []
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            enc = tok(prompts[s : s + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            logits = m(input_ids=enc.input_ids.to(device),
                       attention_mask=enc.attention_mask.to(device)).logits
            out.extend(logits.reshape(-1).tolist())
    return out


def _load_lmsys(n, seed):
    from bench.datasets import iter_requests
    from ltr.ranker.dataset import examples_from_requests

    ex = examples_from_requests(list(iter_requests(n, seed=seed, source="lmsys", model=MODEL)))
    return [e.prompt for e in ex], [e.output_length for e in ex]


def _load_sharegpt(n, seed):
    import random

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(line) for line in Path(SHAREGPT).read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    prompts = [r["prompt"] for r in rows]
    lengths = [len(tok(r["generated"], add_special_tokens=False)["input_ids"]) for r in rows]
    return prompts, lengths


def main() -> None:  # pragma: no cover - integration path
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--held-seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default="results/summaries/ltr_headtohead.json")
    args = ap.parse_args()

    from ltr.ranker.ranking_metrics import kendall_tau

    rankers = [
        ("opt125m-ltr (ours · LMSYS labels)", "ours", "results/ranker/opt125m-ltr"),
        ("opt125m-real (ours · real Llama-8B labels)", "ours", "results/ranker/opt125m-real"),
        ("LLM.pdf downloaded (ShareGPT · Llama-3-8B)", "paper", PAPER),
    ]
    datasets = {}
    print("loading LMSYS held-out…")
    datasets["LMSYS (ours' home)"] = _load_lmsys(args.n, args.held_seed)
    print("loading ShareGPT trace…")
    datasets["ShareGPT (paper's home)"] = _load_sharegpt(args.n, args.held_seed)

    rows = []
    for name, kind, path in rankers:
        entry = {"ranker": name}
        for dset, (prompts, lengths) in datasets.items():
            try:
                scores = (_score_ours if kind == "ours" else _score_paper)(
                    path, prompts, args.device, args.max_length)
                tau = kendall_tau(scores, [-L for L in lengths])
                entry[dset] = round(abs(tau), 4)
            except Exception as e:  # noqa: BLE001
                entry[dset] = f"ERR {type(e).__name__}"
        rows.append(entry)
        print(f"  {name}: " + " · ".join(f"{k}={v}" for k, v in entry.items() if k != "ranker"))
    result = {"n": args.n, "held_seed": args.held_seed, "metric": "Kendall |tau| vs output length",
              "note": "each ranker on BOTH distributions; the diagonal (home turf) is where each "
                      "should win. LMSYS length = reference-reply; ShareGPT length = tokens in `generated`.",
              "rankers": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
