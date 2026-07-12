"""C2a — BLASST algorithmic screen (no vLLM kernel yet).

Validates the online-softmax block-skip on REAL Llama-3.1-8B prefill attention:
capture (post-RoPE) Q/K/V from one layer, simulate the BLASST skip in torch,
sweep the threshold tau -> (achieved block sparsity, output error vs dense).

Also contrasts the *per-query* skip (best case) with the *block-uniform* skip
(BLOCK_M tile must all agree) that a Triton kernel actually needs (§1 of
docs/C2_BLASST_PLAN.md) — quantifying the "CTA-uniform tax" the audits warned of.

    .venv/bin/python -m c2.blasst_screen --layer 16 --block-n 64 --block-m 64
"""
from __future__ import annotations
import argparse, json


def _long_text() -> str:
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        txt = "".join(t for t in ds["text"] if t.strip())
        if len(txt) > 4000:
            return txt
    except Exception:
        pass
    para = ("The King Penguin (Aptenodytes patagonicus) is the second largest species of "
            "penguin, smaller only than the Emperor Penguin. There are two subspecies. King "
            "penguins eat small fish and squid and rely less on krill than other penguins. On "
            "foraging trips they repeatedly dive to over 100 metres, and have been recorded at "
            "depths greater than 300 metres. ")
    return para * 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1536)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--taus", default="2,4,6,8,10,12,16")
    ap.add_argument("--out", default="results/summaries/c2a_blasst_screen.json")
    args = ap.parse_args()

    import torch
    import transformers.models.llama.modeling_llama as M
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cap: dict = {}
    orig = M.eager_attention_forward
    counter = [0]

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        if counter[0] == args.layer and "q" not in cap:
            cap["q"] = query.detach()[0].float()          # [H, N, D]
            cap["k"] = key.detach()[0].float()            # [KVH, N, D]
            cap["v"] = value.detach()[0].float()
            cap["scaling"] = float(scaling)
            cap["groups"] = module.num_key_value_groups
        counter[0] += 1
        return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)

    M.eager_attention_forward = patched
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager", device_map="cuda")
    ids = tok(_long_text(), return_tensors="pt", truncation=True, max_length=args.seq).input_ids.cuda()
    counter[0] = 0
    with torch.no_grad():
        model(ids)
    N = ids.shape[1]
    print(f"captured layer {args.layer}: seq={N} heads={cap['q'].shape[0]} d={cap['q'].shape[2]} groups={cap['groups']}")

    q, k, v = cap["q"].cuda(), cap["k"].cuda(), cap["v"].cuda()
    g = cap["groups"]; scaling = cap["scaling"]
    k = k.repeat_interleave(g, dim=0); v = v.repeat_interleave(g, dim=0)   # GQA -> [H,N,D]
    H, N, D = q.shape
    S = torch.einsum("hqd,hkd->hqk", q, k) * scaling
    causal = torch.triu(torch.ones(N, N, dtype=torch.bool, device=S.device), 1)
    S = S.masked_fill(causal[None], float("-inf"))
    A = torch.softmax(S, dim=-1)
    O_dense = torch.einsum("hqk,hkd->hqd", A, v)
    dnorm = O_dense.norm().item()

    def blasst(tau: float, tile_m: int):
        BN = args.block_n
        m = torch.full((H, N), float("-inf"), device=S.device)
        l = torch.zeros((H, N), device=S.device)
        acc = torch.zeros((H, N, D), device=S.device)
        skipped = torch.zeros((H, N), device=S.device)
        total = torch.zeros((H, N), device=S.device)
        for b in range(0, N, BN):
            hi = min(b + BN, N)
            Sb = S[:, :, b:hi]                                   # [H,N,bw]
            bmax = Sb.max(dim=-1).values                        # [H,N] (-inf if fully masked)
            valid = torch.isfinite(bmax)
            total += valid.float()
            pred = bmax < (m - tau)                             # per-query skip test
            if tile_m > 1:                                      # CTA-uniform: whole M-tile must agree
                nt = (N + tile_m - 1) // tile_m
                pad = nt * tile_m - N
                bmax_p = torch.nn.functional.pad(bmax, (0, pad), value=float("-inf"))
                m_p = torch.nn.functional.pad(m, (0, pad), value=float("inf"))
                tile_bmax = bmax_p.view(H, nt, tile_m).max(-1).values                   # loudest in tile
                tile_m_min = m_p.view(H, nt, tile_m).min(-1).values                     # lowest running-max
                tile_skip = tile_bmax < (tile_m_min - tau)                              # [H,nt]
                pred = tile_skip.repeat_interleave(tile_m, dim=1)[:, :N]
            skip = valid & pred & (b > 0)                       # never skip block 0
            skipped += skip.float()
            keep = valid & (~skip)
            m_new = torch.where(keep, torch.maximum(m, bmax), m)
            alpha = torch.exp(m - m_new)
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
            p = torch.exp(Sb - m_new[:, :, None])
            p = torch.where(keep[:, :, None], p, torch.zeros_like(p))
            l = l * alpha + p.sum(-1)
            acc = acc * alpha[:, :, None] + torch.einsum("hnk,hkd->hnd", p, v[:, b:hi, :])
            m = m_new
        O = acc / l[:, :, None].clamp_min(1e-9)
        rel = (O - O_dense).norm().item() / dnorm
        sparsity = (skipped.sum() / total.sum().clamp_min(1)).item()
        return sparsity, rel

    taus = [float(x) for x in args.taus.split(",")]
    rows = []
    print(f"{'tau':>5} | {'per-query':>20} | {'block-uniform(M='+str(args.block_m)+')':>24}")
    print(f"{'':>5} | {'sparsity':>9} {'rel-err':>10} | {'sparsity':>9} {'rel-err':>10}")
    for t in taus:
        sq, eq = blasst(t, 1)
        sb, eb = blasst(t, args.block_m)
        rows.append({"tau": t, "perquery_sparsity": round(sq, 4), "perquery_relerr": round(eq, 5),
                     "blockuniform_sparsity": round(sb, 4), "blockuniform_relerr": round(eb, 5)})
        print(f"{t:>5.0f} | {sq:>9.3f} {eq:>10.5f} | {sb:>9.3f} {eb:>10.5f}")

    out = {"model": args.model, "layer": args.layer, "seq": N, "block_n": args.block_n,
           "block_m": args.block_m, "rows": rows}
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
