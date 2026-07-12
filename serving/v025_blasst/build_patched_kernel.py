"""Generate serving/v025_blasst/patched_kernel.py = a real-file copy of vLLM 0.25's
kernel_unified_attention with the BLASST skip, so Triton's source parser works
(re-JIT-from-string breaks getsource). Run once against the pinned .venv-v025."""
import inspect, re, os
import vllm.v1.attention.ops.triton_unified_attention as U

_OLD = '''        M, L, P, alpha = softmax_step(S, M, L)
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
            acc += tl.dot(P.to(V.dtype), V)'''

ksrc = inspect.getsource(U.kernel_unified_attention.fn)   # def ... (undecorated)
m = re.search(r"\n\):", ksrc)
assert m
ksrc = ksrc[:m.start()] + "\n    BLASST_TAU: tl.constexpr = _blasst_tau_default," + ksrc[m.start():]
assert _OLD in ksrc, "loop anchor drift"
guard = ("        _bl_keep = True\n"
         "        if BLASST_TAU > 0.0:\n"
         "            _bl_keep = tl.max(tl.max(S, axis=1)) >= (tl.min(M) - BLASST_TAU)\n"
         "        if _bl_keep:\n")
indented = "\n".join(("    " + ln if ln.strip() else ln) for ln in _OLD.split("\n"))
ksrc = ksrc.replace(_OLD, guard + indented, 1)

header = '''"""BLASST-patched copy of vLLM 0.25's kernel_unified_attention (bf16 decode path).
Generated from build_patched_kernel.py against the pinned vLLM 0.25.0 install and
committed for review + reproducibility. Helpers/globals are pulled from the installed
module so Triton's source parser sees a real file; VLLM_BLASST_TAU sets the default.
Do not hand-edit — regenerate with build_patched_kernel.py."""
import os
import triton
import vllm.v1.attention.ops.triton_unified_attention as _U
globals().update({k: v for k, v in _U.__dict__.items() if not k.startswith("__")})
_blasst_tau_default = float(os.getenv("VLLM_BLASST_TAU", "0") or "0")

# NOTE: no @triton.jit here — inspect.getsource(...fn) below already carries the
# original @triton.jit decorator line; adding another double-decorates the kernel
# (outer jit wraps the inner JITFunction -> getsourcelines(JITFunction) TypeError).
'''
out = "serving/v025_blasst/patched_kernel.py"
open(out, "w").write(header + ksrc + "\n")
print("wrote", out, "| BLASST_TAU in it:", "BLASST_TAU" in ksrc, "| guard:", "_bl_keep" in ksrc)
