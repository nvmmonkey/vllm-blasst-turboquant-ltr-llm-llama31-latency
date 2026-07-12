"""C2 — does BLASST decode sparsity survive vLLM's GQA-grouped decode kernel?

vLLM's `kernel_unified_attention` shares ONE KV load across a GQA group
(num_queries_per_kv query heads), so to skip a KV block ALL heads in the group
must agree — a cross-head tax C2a never measured (it only tested cross-position).
This decides whether the ~1.3x standalone speedup can
transfer to the production kernel, or whether it needs a per-head rewrite.

We reuse C2a's capture, treat every query position as a decode query (attends to
its causal prefix), and compare:
  * per-head skip   (each (head, pos) decides alone) = C2b's standalone kernel
  * GQA-group skip  (all num_queries_per_kv heads of a KV group must agree) = the
                     production grouped-decode kernel

    .venv/bin/python -m c2.blasst_gqa_decode --layer 16 --seq 1536 --block-n 64
"""
from __future__ import annotations
import argparse, json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1536)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--taus", default="4,6,8")
    ap.add_argument("--out", default="results/summaries/c2_gqa_decode.json")
    args = ap.parse_args()

    import torch
    import transformers.models.llama.modeling_llama as M
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cap: dict = {}
    orig = M.eager_attention_forward
    ctr = [0]

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        if ctr[0] == args.layer and "q" not in cap:
            cap["q"] = query.detach()[0].float()        # [H,N,D]
            cap["k"] = key.detach()[0].float()          # [KVH,N,D]
            cap["scaling"] = float(scaling)
            cap["groups"] = module.num_key_value_groups  # queries per kv head
        ctr[0] += 1
        return orig(module, query, key, value, attention_mask, scaling, dropout, **kw)

    M.eager_attention_forward = patched
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager", device_map="cuda")
    txt = ("The King Penguin is the second largest penguin. It eats fish and squid, dives "
           "past 100 metres, and breeds on subantarctic islands in large colonies. ") * 40
    ids = tok(txt, return_tensors="pt", truncation=True, max_length=args.seq).input_ids.cuda()
    ctr[0] = 0
    with torch.no_grad():
        model(ids)

    q, k = cap["q"].cuda(), cap["k"].cuda()
    g = cap["groups"]; scale = cap["scaling"]
    H, N, D = q.shape
    KVH = k.shape[0]
    BN = args.block_n
    nblk = (N + BN - 1) // BN
    # scores per head vs its own kv head: S[h] = q[h] @ k[h//g].T
    kk = k.repeat_interleave(g, dim=0)                                    # [H,N,D]
    S = torch.einsum("hqd,hkd->hqk", q, kk) * scale                      # [H,N,N]
    causal = torch.triu(torch.ones(N, N, dtype=torch.bool, device=S.device), 1)
    S = S.masked_fill(causal[None], float("-inf"))

    # block maxima per (head, query, block): [H,N,nblk]
    pad = nblk * BN - N
    Sp = torch.nn.functional.pad(S, (0, pad), value=float("-inf"))
    bmax = Sp.view(H, N, nblk, BN).max(-1).values                        # [H,N,nblk]

    def sparsity(tau, grouped):
        # online running-max with skip; skip decision per-head or per-GQA-group.
        # vectorized over (H, query); loop nblk (~24).
        m = torch.full((H, N), float("-inf"), device=S.device)
        skipped = torch.zeros((H, N), device=S.device)
        total = torch.zeros((H, N), device=S.device)
        for b in range(nblk):
            bm = bmax[:, :, b]                                           # [H,N]
            valid = torch.isfinite(bm)
            total += valid.float()
            pred = bm < (m - tau)                                        # per-head skip test
            if grouped:                                                 # all heads in a KV group must agree
                pg = pred.view(KVH, g, N)
                vg = valid.view(KVH, g, N)
                # a group can skip a block only where every VALID head agrees; heads with no
                # valid key at this (block,pos) don't block the decision
                grp = (pg | ~vg).all(dim=1) & vg.any(dim=1)             # [KVH,N]
                pred = grp.repeat_interleave(g, dim=0)                   # [H,N]
            skip = valid & pred & (b > 0)
            skipped += skip.float()
            keep = valid & (~skip)
            m = torch.where(keep, torch.maximum(m, bm), m)
        return (skipped.sum() / total.sum().clamp_min(1)).item()

    taus = [float(x) for x in args.taus.split(",")]
    rows = []
    print(f"seq={N} heads={H} kv_heads={KVH} queries_per_kv={g} blocks={nblk}")
    print(f"{'tau':>4} | {'per-head':>9} | {'GQA-group':>9} | {'tax (grp/head)':>14}")
    for t in taus:
        ph = sparsity(t, False)
        gr = sparsity(t, True)
        ratio = gr / ph if ph > 0 else 0.0
        rows.append({"tau": t, "perhead_sparsity": round(ph, 4),
                     "gqa_group_sparsity": round(gr, 4), "retained_frac": round(ratio, 3)})
        print(f"{t:>4.0f} | {ph:>9.3f} | {gr:>9.3f} | {ratio:>13.2f}x")

    out = {"model": args.model, "layer": args.layer, "seq": N, "block_n": BN,
           "queries_per_kv": g, "kv_heads": KVH, "rows": rows}
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
