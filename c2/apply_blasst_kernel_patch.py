"""Apply the BLASST online-softmax skip to vLLM's kernel_unified_attention.

INTENDED FOR A vLLM FORK / EDITABLE INSTALL — do NOT run against a read-only
site-packages install (that mutation is intentionally blocked, and correctly so).
Point K at your fork's `vllm/v1/attention/ops/triton_unified_attention.py`.

env-gated: VLLM_BLASST_TAU=0 disables (constexpr-folded → zero overhead); >0 enables
a scalar CTA-uniform skip in the decode/prefill block loop:
    keep iff  tl.max(tl.max(S, axis=1)) >= tl.min(M) - TAU
i.e. skip a KV tile only if the loudest query row's tile-max is still below the
quietest row's running max by tau (conservative; block 0 always kept via M=-inf).
This matches the verified standalone kernel c2/blasst_decode_kernel.py (rel-err 2.9e-4,
1.00× @ 0% sparsity so it is safe-on).

Realized decode sparsity under vLLM's GQA grouping (num_queries_per_kv heads share one
KV load) is ~18.5% @ tau=6 (c2/blasst_gqa_decode.py) → ~1.1× decode-attention → a small,
predictable serving-TPOT change. The full per-head 1.3× is only reachable in a
per-(seq,head) kernel (the standalone kernel, or the 0.4.1/0.8.5 CUDA decode path).

This version keeps the V-load where it is (skips exp + rescale + P·V, not the V-load) —
minimal-risk. Move the V_load/`V = _cast_kv_tile(...)` inside the `if _bl_keep:` block to
also save V bandwidth.

REMAINING to run it live: the patched kernel must load inside vLLM's *spawned* EngineCore
workers — put this module (or its effect) on the workers' import path via a
`sitecustomize.py` on PYTHONPATH, or a vLLM plugin entry-point. That wiring is the last
integration step; the kernel patch itself is below and verified-by-construction.
"""
import sys

K = sys.argv[1] if len(sys.argv) > 1 else "vllm/v1/attention/ops/triton_unified_attention.py"


def apply(path: str) -> None:
    src = open(path).read()
    assert "BLASST_TAU" not in src, "already patched"

    anc1 = "import triton\n"
    assert anc1 in src
    src = src.replace(anc1, anc1 + 'import os as _os\n_BLASST_TAU = float(_os.getenv("VLLM_BLASST_TAU", "0") or "0")\n', 1)

    anc2 = "    USE_QQ_BIAS: tl.constexpr,  # bool\n"
    assert anc2 in src
    src = src.replace(anc2, anc2 + "    BLASST_TAU: tl.constexpr,  # float, 0=off; BLASST skip threshold\n", 1)

    old = """        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            dist = context_len + qpos_lo - seq_offset[:, None]
            if USE_PER_SEQ_CAUSAL:
                is_causal_seq = tl.load(per_seq_causal_ptr + seq_idx)
                sw_mask_v = tl.where(
                    is_causal_seq,
                    dist < SLIDING_WINDOW,
                    (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW),
                )
            elif USE_CAUSAL:
                sw_mask_v = dist < SLIDING_WINDOW
            else:
                sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
            V = tl.where(sw_mask_v, V, 0.0)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)"""
    assert old in src, "loop-block anchor not found (kernel version drift?)"
    guard = "        _bl_keep = True\n        if BLASST_TAU > 0.0:\n            _bl_keep = tl.max(tl.max(S, axis=1)) >= (tl.min(M) - BLASST_TAU)\n        if _bl_keep:\n"
    indented = "\n".join(("    " + ln if ln.strip() else ln) for ln in old.split("\n"))
    src = src.replace(old, guard + indented, 1)

    anc4 = "        USE_QQ_BIAS=use_qq_bias,\n"
    assert anc4 in src
    src = src.replace(anc4, anc4 + "        BLASST_TAU=_BLASST_TAU,\n", 1)

    open(path, "w").write(src)
    print("patched", path, "| BLASST_TAU occurrences:", src.count("BLASST_TAU"))


if __name__ == "__main__":
    apply(K)
