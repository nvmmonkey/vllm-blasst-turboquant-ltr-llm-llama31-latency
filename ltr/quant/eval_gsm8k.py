"""Generation-based accuracy for KV-quant (C1, Track C) — the RIGHT metric.

Teacher-forced perplexity is BLIND to KV-cache quantization: prefill attends to
raw K/V, not the cached quantized KV, so fp16 and fp8-KV give bit-identical PPL
(confirmed 17.90024668528097 — results/RESULTS.md). A wikitext-PPL eval would
falsely report "zero degradation."

This measures accuracy correctly: GSM8K solved by GREEDY GENERATION, where each
decoded token attends to the (quantized) cached context — the path fp8 KV
actually corrupts. HTTP client (this file loads the data + scores) against a
RUNNING vLLM server whose --kv-cache-dtype fixes the quant mode; run one server
per dtype so the data-loading venv and the serving venv stay decoupled:

    # serve auto (fp16), then:  python -m ltr.quant.eval_gsm8k --out gsm8k_fp16.json
    # serve fp8,        then:  python -m ltr.quant.eval_gsm8k --out gsm8k_fp8.json --dtype-note fp8
    python -m ltr.quant.eval_gsm8k --compare gsm8k_fp16.json gsm8k_fp8.json

fp8-KV accuracy < fp16 accuracy is C1's COST side, the latency sweep's missing half.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

_INSTR = (
    "Solve the following grade-school math problem. Think step by step, then give "
    "the final numeric answer on its own line in the exact form '#### <number>'.\n\n"
    "Problem: {q}"
)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm(cand: str | None) -> str | None:
    if cand is None:
        return None
    c = cand.rstrip(".").replace(",", "")
    try:
        f = float(c)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def _extract(text: str) -> str | None:
    """Final numeric answer: prefer the '#### x' tag, else the last number."""
    tag = re.findall(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if tag:
        return _norm(tag[-1])
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


def _gold(ans: str) -> str | None:
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", ans)
    return _norm(m.group(1)) if m else None


async def _run(base_url: str, model: str, questions, gold, gen: int, conc: int):
    import aiohttp

    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(conc)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)

    async def one(session, q):
        payload = {"model": model, "messages": [{"role": "user", "content": _INSTR.format(q=q)}],
                   "max_tokens": gen, "temperature": 0.0}
        async with sem, session.post(url, json=payload) as r:
            d = await r.json()
            try:
                return d["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                return ""

    async with aiohttp.ClientSession(timeout=timeout) as s:
        texts = await asyncio.gather(*[one(s, q) for q in questions])
    rows, correct = [], 0
    for t, g in zip(texts, gold):
        pred = _extract(t)
        ok = pred is not None and g is not None and pred == g
        correct += int(ok)
        rows.append({"gold": g, "pred": pred, "ok": ok})
    return rows, correct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gen-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype-note", default="auto", help="label only: the server's kv-cache-dtype")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("FP16", "OTHER"))
    args = ap.parse_args()

    if args.compare:
        print(json.dumps(compare(*args.compare), indent=2))
        return

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=args.seed)
    ds = ds.select(range(min(args.n, len(ds))))
    questions = [r["question"] for r in ds]
    gold = [_gold(r["answer"]) for r in ds]

    rows, correct = asyncio.run(
        _run(args.base_url, args.model, questions, gold, args.gen_tokens, args.concurrency)
    )
    acc = correct / max(len(rows), 1)
    result = {"kv_cache_dtype": args.dtype_note, "n": len(rows), "correct": correct,
              "accuracy": round(acc, 4), "rows": rows}
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "rows"}))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


def compare(fp16_path: str, other_path: str) -> dict:
    """Accuracy delta + how often the quantized run flips a correct answer wrong."""
    a = json.loads(Path(fp16_path).read_text())
    b = json.loads(Path(other_path).read_text())
    cw = sum(1 for x, y in zip(a["rows"], b["rows"]) if x["ok"] and not y["ok"])
    wc = sum(1 for x, y in zip(a["rows"], b["rows"]) if not x["ok"] and y["ok"])
    return {"fp16_accuracy": a["accuracy"], "other_accuracy": b["accuracy"],
            "other_dtype": b["kv_cache_dtype"], "delta_acc": round(b["accuracy"] - a["accuracy"], 4),
            "n": a["n"], "correct_to_wrong": cw, "wrong_to_correct": wc}


if __name__ == "__main__":
    main()
