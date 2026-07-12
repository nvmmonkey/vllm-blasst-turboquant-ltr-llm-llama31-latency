"""Generate serving/v025_blasst/turboquant_patched_kernel.py = a real-file copy of
vLLM 0.25's _tq_decode_stage1 (the TurboQuant per-head flash-decoding kernel) with the
BLASST skip. This is the C1+C2 path.

Why this kernel and not kernel_unified_attention: _tq_decode_stage1 is launched with
grid=(batch, q_head, kv_split) — ONE program per (seq, q_head) decoding a single query,
so `scores` is 1D [BLOCK_KV] and the skip predicate is a plain scalar
`tl.max(scores) >= m_prev - tau`. No BLOCK_Q packing, no GQA-shared KV load => no
CTA-uniform tile tax (which crushed realized sparsity to <7% and made BLASST net-NEGATIVE
on the bf16 unified kernel). Here BLASST reaches its full ~42% decode sparsity, and a
skipped block also skips the expensive value dequant (MSE unpack / centroid gather / norm).

env-gated: VLLM_BLASST_TAU=0 => constexpr-folded off (dense, pure C1); >0 => C1+C2.
Run once against the pinned .venv-v025 (needs the installed vLLM to read source)."""
import inspect
import vllm.v1.attention.ops.triton_turboquant_decode as TQ

ksrc = inspect.getsource(TQ._tq_decode_stage1.fn)  # carries its own @triton.jit line

# 1) add BLASST_TAU constexpr after the last defaulted constexpr param
anc = "    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)\n"
assert anc in ksrc, "signature anchor drift"
ksrc = ksrc.replace(
    anc, anc + "    BLASST_TAU: tl.constexpr = _blasst_tau_default,  # 0=off (dense/pure-C1)\n", 1
)

# 2) wrap the per-block contribution (online softmax -> value dequant -> acc/l/m update)
#    in a scalar BLASST skip guard. scores is 1D [BLOCK_KV] => scalar CTA-uniform predicate.
#    block 0 keeps automatically: m_prev=-inf => (m_prev - tau)=-inf => max(scores) >= -inf.
old_start = "        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)"
old_end = "        m_prev = n_e_max"
assert old_start in ksrc and old_end in ksrc, "loop-body anchor drift"
i = ksrc.index(old_start)
j = ksrc.index(old_end) + len(old_end)
block = ksrc[i:j]
guard = ("        _bl_keep = True\n"
         "        if BLASST_TAU > 0.0:\n"
         "            _bl_keep = tl.max(scores, 0) >= (m_prev - BLASST_TAU)\n"
         "        if _bl_keep:\n")
indented = "\n".join(("    " + ln if ln.strip() else ln) for ln in block.split("\n"))
ksrc = ksrc[:i] + guard + indented + ksrc[j:]

header = '''"""BLASST-patched copy of vLLM 0.25's _tq_decode_stage1 (TurboQuant per-head flash-decode).
Generated from build_turboquant_patch.py against the pinned vLLM 0.25.0 install and committed
for review + reproducibility. per-head program + 1D scores gives a scalar skip predicate (no
GQA/tile tax), so a skipped block also skips the value dequant. Do not hand-edit — regenerate
with build_turboquant_patch.py. (No @triton.jit here: getsource below carries the original
decorator; double-decorating breaks getsourcelines.)"""
import os
import triton
import vllm.v1.attention.ops.triton_turboquant_decode as _TQ
globals().update({k: v for k, v in _TQ.__dict__.items() if not k.startswith("__")})
_blasst_tau_default = float(os.getenv("VLLM_BLASST_TAU", "0") or "0")


'''
out = "serving/v025_blasst/turboquant_patched_kernel.py"
open(out, "w").write(header + ksrc + "\n")
print("wrote", out,
      "| BLASST_TAU:", "BLASST_TAU" in ksrc,
      "| guard:", "_bl_keep" in ksrc,
      "| scalar-pred:", "tl.max(scores, 0) >= (m_prev - BLASST_TAU)" in ksrc)
