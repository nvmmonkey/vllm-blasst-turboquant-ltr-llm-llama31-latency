"""Fine-tune the OPT-125M output-length ranker with ListMLE (Track B).

Deterministic (fixed seed, no online learning). Uses the tested building blocks:
LMSYS length data (:mod:`ltr.ranker.dataset`), the OPT ranker
(:mod:`ltr.ranker.model`), and the ListMLE loss (:mod:`ltr.ranker.losses`).

Run on the GPU host:
    python -m ltr.ranker.train --out results/ranker/opt125m-ltr --n 4000 --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tokenize_prompts(tokenizer, prompts, *, max_length: int):
    """Pad/truncate a list of prompts to (input_ids, attention_mask) tensors."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    return enc["input_ids"], enc["attention_mask"]


def train(  # pragma: no cover - integration path (needs torch + data + GPU)
    *,
    out: str,
    base: str = "facebook/opt-125m",
    target_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    source: str = "lmsys",
    n: int = 4000,
    list_size: int = 16,
    epochs: int = 10,
    lr: float = 2e-5,
    max_length: int = 512,
    seed: int = 0,
    device: str = "cuda",
    resume: str | None = None,
    labels_file: str | None = None,
) -> dict:
    import torch
    from transformers import AutoTokenizer

    from bench.datasets import iter_requests
    from ltr.ranker.dataset import (
        examples_from_labels_file,
        examples_from_requests,
        make_lists,
        relevance_from_lengths,
    )
    from ltr.ranker.losses import listmle_loss
    from ltr.ranker.model import build_ranker

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base)

    if labels_file:  # target-model-sampled lengths (audit fix #1 — the real labels)
        examples = examples_from_labels_file(labels_file)
    else:  # LMSYS reference-reply lengths (a proxy — other models wrote them)
        requests = list(iter_requests(n, seed=seed, source=source, model=target_model))
        examples = examples_from_requests(requests)
    lists = make_lists(examples, list_size, seed=seed)
    if not lists:
        raise RuntimeError("no training lists built — check dataset access / n / list_size")

    ranker = build_ranker(base=base).to(device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=lr)

    history: list[float] = []
    if resume:  # continue from a saved checkpoint (e.g. epoch 5 -> 6..10)
        prior = json.loads((Path(resume) / "ranker_meta.json").read_text())
        history = list(prior.get("loss_history", []))
        ranker.load_state_dict(torch.load(Path(resume) / "ranker.pt", map_location=device))
        print(f"resumed from {resume}: {len(history)} prior epochs", flush=True)
    start_epoch = len(history)
    total_epochs = start_epoch + epochs
    for epoch in range(epochs):
        epoch_loss = 0.0
        for lst in lists:
            input_ids, attn = tokenize_prompts(
                tokenizer, [e.prompt for e in lst], max_length=max_length
            )
            scores = ranker(input_ids.to(device), attn.to(device))
            relevance = torch.tensor(
                relevance_from_lengths([e.output_length for e in lst]),
                dtype=torch.float32, device=device,
            )
            loss = listmle_loss(scores, relevance)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        mean_loss = epoch_loss / len(lists)
        history.append(mean_loss)
        # flush so per-epoch progress is visible live even when stdout is
        # redirected to a file (Python block-buffers a non-tty stdout).
        print(f"epoch {start_epoch + epoch + 1}/{total_epochs}  listmle={mean_loss:.4f}", flush=True)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ranker.state_dict(), out_dir / "ranker.pt")
    tokenizer.save_pretrained(out_dir)
    meta = {"base": base, "seed": seed, "epochs": total_epochs, "list_size": list_size,
            "resumed_from": resume, "labels_file": labels_file, "loss_history": history}
    (out_dir / "ranker_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved ranker to {out_dir}", flush=True)
    return meta


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Fine-tune the OPT-125M LTR ranker (ListMLE).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="facebook/opt-125m")
    ap.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--source", default="lmsys", choices=["lmsys", "synthetic"])
    ap.add_argument("--n", type=int, default=4000,
                    help="training samples (LLM.pdf spec: 23800; use less for quick local runs)")
    ap.add_argument("--list-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None,
                    help="continue from a saved ranker dir (e.g. resume a 5-epoch run for 5 more)")
    ap.add_argument("--labels-file", default=None,
                    help="JSON of target-model-sampled lengths from ltr.ranker.synthesize "
                         "(audit fix #1); overrides LMSYS reference lengths")
    args = ap.parse_args()
    train(
        out=args.out, base=args.base, target_model=args.target_model, source=args.source,
        n=args.n, list_size=args.list_size, epochs=args.epochs, lr=args.lr,
        seed=args.seed, device=args.device, resume=args.resume, labels_file=args.labels_file,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
