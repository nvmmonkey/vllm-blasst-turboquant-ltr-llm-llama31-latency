"""C2b — standalone Triton decode-attention kernel with the BLASST skip guard,
and a real 3090 latency benchmark: does skipping low-value KV blocks actually
speed up decode attention on Ampere? (docs/C2_BLASST_PLAN.md §8 open question.)

This is the kernel-level proof, decoupled from vLLM production integration. One
program per (seq, query-head) — the decode case where the skip predicate is
naturally scalar (C2a showed this is where BLASST realizes its sparsity). We
construct a controllable block-sparsity scenario and time dense vs skip.

    .venv-v025/bin/python -m c2.blasst_decode_kernel
"""
from __future__ import annotations
import argparse, json
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q, K, V, O, scale, N, TAU,
            sq, skb, skn, svb, svn, so,
            D: tl.constexpr, BLOCK_N: tl.constexpr, SKIP: tl.constexpr):
    b = tl.program_id(0)
    d = tl.arange(0, D)
    q = tl.load(Q + b * sq + d)                                   # [D]
    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        k = tl.load(K + b * skb + offs[:, None] * skn + d[None, :], mask=mask[:, None], other=0.0)  # [BN,D]
        qk = tl.sum(q[None, :] * k, axis=1) * scale               # [BN]
        qk = tl.where(mask, qk, -float("inf"))
        block_max = tl.max(qk, axis=0)                            # scalar -> real branch
        do = True
        if SKIP:
            do = block_max >= m - TAU                            # BLASST: keep only non-negligible blocks
        if do:                                                   # skip => no exp, no V-load, no P·V
            m_new = tl.maximum(m, block_max)
            alpha = tl.exp(m - m_new)
            p = tl.where(mask, tl.exp(qk - m_new), 0.0)          # [BN]
            v = tl.load(V + b * svb + offs[:, None] * svn + d[None, :], mask=mask[:, None], other=0.0)  # [BN,D]
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = m_new
    tl.store(O + b * so + d, acc / l)


def run(Q, K, V, tau, skip, BLOCK_N):
    BH, N, D = K.shape
    O = torch.empty((BH, D), device=Q.device, dtype=torch.float32)
    scale = 1.0 / (D ** 0.5)
    _decode[(BH,)](Q, K, V, O, scale, N, tau,
                   Q.stride(0), K.stride(0), K.stride(1), V.stride(0), V.stride(1), O.stride(0),
                   D=D, BLOCK_N=BLOCK_N, SKIP=skip)
    return O


def make_scene(BH, N, D, sparsity, BLOCK_N, device):
    """Block 0 hot (qk~10); of the rest, `sparsity` fraction cold (qk~0, skippable at
    tau=4), the others warm (qk~8, kept). Controls achieved sparsity directly."""
    g = torch.Generator(device=device).manual_seed(0)
    u = torch.randn(BH, D, generator=g, device=device)
    u = u / u.norm(dim=-1, keepdim=True)                          # query direction
    peaks = torch.full((BH, N), 8.0, device=device)              # warm default
    nblk = N // BLOCK_N
    for b in range(1, nblk):                                      # block 0 stays hot
        cold = (torch.rand((), generator=g, device=device) < sparsity)
        peaks[:, b * BLOCK_N:(b + 1) * BLOCK_N] = 0.0 if cold else 8.0
    peaks[:, 0:BLOCK_N] = 10.0
    K = u[:, None, :] * (peaks[:, :, None] * (D ** 0.5))          # so q·k*scale ≈ peak
    K = K + 0.01 * torch.randn(BH, N, D, generator=g, device=device)
    V = torch.randn(BH, N, D, generator=g, device=device)
    return u.contiguous(), K.contiguous(), V.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh", type=int, default=256, help="programs = seqs*query-heads (e.g. 8*32)")
    ap.add_argument("--n", type=int, default=2048, help="context length")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--sparsities", default="0,0.25,0.5,0.75,0.9")
    ap.add_argument("--out", default="results/summaries/c2b_decode_bench.json")
    args = ap.parse_args()
    dev = "cuda"

    # correctness: dense torch reference vs skip kernel at a mid sparsity
    u, K, V = make_scene(args.bh, args.n, args.d, 0.5, args.block_n, dev)
    scale = 1.0 / (args.d ** 0.5)
    S = torch.einsum("bd,bnd->bn", u, K) * scale
    ref = torch.einsum("bn,bnd->bd", torch.softmax(S, -1), V)
    o_skip = run(u, K, V, args.tau, True, args.block_n)
    rel = (o_skip - ref).norm().item() / ref.norm().item()
    print(f"correctness (skip vs dense-softmax, sparsity 0.5, tau {args.tau}): rel-err {rel:.2e}")

    rows = []
    print(f"\n{'target':>7} {'dense ms':>9} {'skip ms':>8} {'speedup':>8}")
    for sp in [float(x) for x in args.sparsities.split(",")]:
        u, K, V = make_scene(args.bh, args.n, args.d, sp, args.block_n, dev)
        t_dense = triton.testing.do_bench(lambda: run(u, K, V, args.tau, False, args.block_n))
        t_skip = triton.testing.do_bench(lambda: run(u, K, V, args.tau, True, args.block_n))
        speed = t_dense / t_skip
        rows.append({"target_sparsity": sp, "dense_ms": round(t_dense, 4),
                     "skip_ms": round(t_skip, 4), "speedup": round(speed, 3)})
        print(f"{sp:>7.2f} {t_dense:>9.4f} {t_skip:>8.4f} {speed:>7.2f}x")

    out = {"shape": {"bh": args.bh, "n": args.n, "d": args.d, "block_n": args.block_n},
           "tau": args.tau, "correctness_relerr": rel, "rows": rows,
           "gpu": torch.cuda.get_device_name(0)}
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
