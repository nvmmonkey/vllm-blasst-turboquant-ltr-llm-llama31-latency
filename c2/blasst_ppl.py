"""C2 quality metric — perplexity cost of BLASST block-skip (all layers, real forward).

Monkeypatches Llama's eager attention to apply the per-query online-softmax block
skip in EVERY layer, then measures wikitext-2 perplexity vs dense at several tau.
This is the accuracy side of C2 (complements C2a's attention-output error and
C2b's kernel speedup): does dropping low-value blocks hurt language modeling?

    .venv/bin/python -m c2.blasst_ppl --taus inf,8,6,4 --block-n 64
"""
from __future__ import annotations
import argparse, json, math
import torch

STATE = {"tau": float("inf"), "block_n": 64}


def _blasst_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import repeat_kv
    k = repeat_kv(key, module.num_key_value_groups)
    v = repeat_kv(value, module.num_key_value_groups)
    S = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        S = S + attention_mask[..., : k.shape[-2]]
    tau, BN = STATE["tau"], STATE["block_n"]
    if not math.isfinite(tau):                                   # dense reference
        A = F.softmax(S, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(A, v).transpose(1, 2).contiguous(), None
    Sf = S.float()
    B, H, Nq, Nk = Sf.shape
    D = v.shape[-1]
    m = torch.full((B, H, Nq), float("-inf"), device=Sf.device)
    l = torch.zeros((B, H, Nq), device=Sf.device)
    acc = torch.zeros((B, H, Nq, D), device=Sf.device, dtype=torch.float32)
    vf = v.float()
    for b in range(0, Nk, BN):
        hi = min(b + BN, Nk)
        Sb = Sf[..., b:hi]                                       # [B,H,Nq,bw]
        bmax = Sb.max(dim=-1).values                            # [B,H,Nq]
        valid = torch.isfinite(bmax)
        skip = valid & (bmax < m - tau) & (b > 0)               # never skip block 0
        keep = valid & (~skip)
        m_new = torch.where(keep, torch.maximum(m, bmax), m)
        alpha = torch.exp(m - m_new)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        p = torch.where(keep[..., None], torch.exp(Sb - m_new[..., None]), torch.zeros_like(Sb))
        l = l * alpha + p.sum(-1)
        acc = acc * alpha[..., None] + torch.matmul(p, vf[..., b:hi, :])
        m = m_new
    O = (acc / l[..., None].clamp_min(1e-9)).to(query.dtype)
    return O.transpose(1, 2).contiguous(), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--taus", default="inf,8,6,4")
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--out", default="results/summaries/c2_blasst_ppl.json")
    args = ap.parse_args()
    STATE["block_n"] = args.block_n

    import transformers.models.llama.modeling_llama as M
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    M.eager_attention_forward = _blasst_eager

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager", device_map="cuda")
    model.eval()
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    except Exception:                                            # robust fallback: coherent gsm8k prose
        ds = load_dataset("openai/gsm8k", "main", split="test")
        text = "\n\n".join(f"{r['question']} {r['answer']}" for r in ds.select(range(400)))
    ids = tok(text, return_tensors="pt").input_ids[0]
    chunks = [ids[i * args.seq:(i + 1) * args.seq] for i in range(args.chunks)]
    chunks = [c for c in chunks if len(c) == args.seq]

    def ppl(tau):
        STATE["tau"] = tau
        tot_nll, tot_tok = 0.0, 0
        with torch.no_grad():
            for c in chunks:
                x = c.unsqueeze(0).cuda()
                out = model(x, labels=x)
                tot_nll += out.loss.item() * (x.shape[1] - 1)
                tot_tok += x.shape[1] - 1
        return math.exp(tot_nll / tot_tok)

    taus = [float("inf") if t.strip() == "inf" else float(t) for t in args.taus.split(",")]
    base = ppl(float("inf"))
    rows = [{"tau": "inf(dense)", "ppl": round(base, 4), "delta_ppl": 0.0}]
    print(f"{'tau':>10} {'PPL':>9} {'ΔPPL':>8}")
    print(f"{'inf(dense)':>10} {base:>9.3f} {0.0:>8.3f}")
    for t in taus:
        if not math.isfinite(t):
            continue
        p = ppl(t)
        rows.append({"tau": t, "ppl": round(p, 4), "delta_ppl": round(p - base, 4)})
        print(f"{t:>10.0f} {p:>9.3f} {p - base:>8.3f}")

    out = {"model": args.model, "block_n": args.block_n, "seq": args.seq,
           "chunks": len(chunks), "dense_ppl": round(base, 4), "rows": rows}
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
