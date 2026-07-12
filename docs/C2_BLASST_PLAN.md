# C2_BLASST_PLAN.md — building BLASST (Skip-Softmax) block sparsity in vLLM (v1 + 0.4.1)

**C2 = attention *compute* sparsity = BLASST** (Yuan et al., arXiv:2512.12087). This is
the design/implementation plan for a **Triton reimplementation** we build ourselves so
it runs on the **RTX 3090 (Ampere sm_86)** and composes with our LTR + FP8-KV stack.

> **Audited.** §2/§4/§9 were rewritten after a close kernel audit against the real
> vLLM source (both engines). The audit **overturned the original headline target**
> (`triton_flash_attention.py` — it is dead code on the 3090) and re-grounded every
> file/line below in the kernels that actually execute. See §9 for the finding log.

> **Axis (important, and where an earlier doc revision was wrong).** BLASST is a
> **COMPUTE / attention-bandwidth** optimization — it skips the softmax + P·V work for
> negligible KV blocks; it does **NOT** shrink the KV cache. So it is **complementary to
> C1** (quantization = memory/capacity → fewer preemptions): C1 wins the **memory-bound**
> regime, C2 wins the **compute/attention-bound** regime (long context, where attention
> is O(n²)). Do not describe C2 as "shrinking the cache."

## 1. What BLASST does (the algorithm)

Inside a FlashAttention-style **online-softmax** loop over KV blocks, BLASST uses the
already-computed running statistics to skip blocks whose post-softmax contribution is
provably negligible — **no separate scoring pass, no extra cache, training-free**:

```
running max  m_i,  running denom  l_i,  accumulator acc
for each KV block n:
    qk        = Q · Kn^T                 # block scores — ALWAYS computed (can't skip)
    block_max = max(qk over keys)        # local max (already computed)
    if block_max < m_i - τ:              # BLASST: block is negligible vs the best so far
        SKIP exp(qk), SKIP V-load, SKIP P·Vn   # exp() decays fast → contribution ≈ 0
        # keep m_i, l_i, acc unchanged (alpha = 1)
    else:
        m_ij = max(m_i, block_max); p = exp(qk - m_ij); l_ij = sum(p)
        acc  = acc·exp(m_i - m_ij) + p·Vn ;  m_i, l_i = update
```

`τ` (threshold) trades sparsity vs approximation error. TRT-LLM exposes it as
`threshold_scale_factor` (per-phase prefill/decode); we mirror that. Because softmax
weights decay exponentially, a block far below the running max contributes ~0, so
skipping it changes the output negligibly (paper preserves benchmark accuracy).

**Two hard constraints the audit surfaced — they shape everything in §2:**
1. **The `Q·Kn^T` matmul is never skippable** (you need it to compute `block_max`). Only
   the `exp`, the **V-load**, and the `P·Vn` matmul are skippable → the ceiling is
   **≈ half the attention FLOPs + the V bandwidth**, not "all the block's work."
2. **The skip must be a scalar, CTA-uniform predicate** so Triton emits a *real branch*.
   The natural per-row form (`block_max[BLOCK_M] < m_i[BLOCK_M] - τ`) is a **vector** and
   Triton lowers it to `tl.where(..., 0)` = *mask-to-zero* = the matmul still runs = **no
   savings**. The predicate must reduce to one scalar over the whole tile:
   `skip ⟺ tl.max(block_max) < tl.min(m_i) - τ`. Consequence: every row in the tile (and,
   under GQA, every query head sharing the KV block) must agree — one "loud" row/head
   keeps the block. This **caps achievable sparsity**, most in prefill (BLOCK_M=128).

**Reference implementation = NVIDIA TensorRT-LLM "Skip Softmax"**
(`SkipSoftmaxAttentionConfig(threshold_scale_factor=…)`), **officially Hopper/Blackwell
only**, kernel is TRT-LLM-engine-bound (no reusable ABI, not callable from vLLM). The
*algorithm* is portable, so we **reimplement it in vLLM's Triton attention path** for
Ampere. We cite BLASST/TRT-LLM as prior art; the contribution here is the vLLM-native
port + its interaction with LTR scheduling and FP8 KV.

## 2. The exact insertion points (audited — the two engines do NOT share the kernel)

**The original plan's premise was false.** `triton_flash_attention.py` — the file the
first draft targeted — is **dead code on the RTX 3090 in both engines**: it is imported
only by `rocm_flash_attn.py` (`is_hip()`, AMD) and by the MLA backends (DeepSeek).
Standard GQA on NVIDIA never executes it. Likewise `triton_decode_attention.py` is
MLA-only. Editing either produces a benchmark byte-identical to baseline. The real,
executed kernels are below, and 0.4.1 ≠ v1 (0.4.1 predates vLLM's Triton-backend
unification), so they are plumbed **separately**.

**Shared truths (both engines):** the live kernels use **`tl.exp` (natural-log units)**,
not `exp2` → `τ` is a natural-log threshold, **no `log2(e)` scaling** (the `exp2`/log2
form only exists in the dead `triton_flash_attention.py`). The skip guard wraps the
`exp`, the `l/m` update, **and** the V-load + `tl.dot(p,v)`. Always keep **block 0** and
the **recent/current-query window**; make the predicate scalar (§1 constraint 2).

### 2.1 — v1 (vllm 0.8.5.post1, `.venv-v1fp8`) — recommended primary target

The audit favours **v1 as the more tractable engine for C2**: in the V1 engine, *both*
the prefill-context and the decode kernels are modifiable **Triton** (0.4.1's are
Triton-prefill-only + CUDA-decode, see §2.2). **But a runtime-forensics check
(§9.1) overturned the "we already run v1" assumption — read it before trusting this
section.** In short: our grid never actually ran the V1 engine, so v1-first is a
*target to stand up and validate at Gate 0*, not a place we already live.

> **fp8 ⇒ Triton *if you run V1* — but on the 3090, FlashAttention rejects fp8, and our
> runs fell back off V1 entirely.** `flash_attn_supports_fp8()` (`fa_utils.py:51-54`)
> needs FA v3 + `major==9` (Hopper); the 3090 is `major==8`, so V1's FlashAttention
> raises on fp8 KV (`flash_attn.py:642-647`). *Were we on V1*, fp8 would force the Triton
> backend. **We were not** (§9.1): `--scheduling-policy priority` (the LTR knob) and
> `VLLM_ATTENTION_BACKEND=XFORMERS` each force a **V0 fallback**, and fp8 then runs on the
> **V0 CUDA E5M2 path** (sm_86-OK), not Triton. Note our 0.8.5 Triton backend has **no**
> fp8 arch-guard (newer vLLM added an sm_89 check — not in 0.8.5.post1), so V1+Triton+fp8
> *should* work on the 3090 once actually selected — but **we have never run it**, so it
> is a Gate-0 experiment, not a premise.

> **⚠ LTR ↔ V1 tension (new, blocking for LTR+C2).** LTR (B1) needs
> `--scheduling-policy priority`, which the V1 engine **rejects** (forces V0, §9.1). So on
> 0.8.5 you cannot naively have *LTR + V1-Triton* together. Gate 0 must resolve this:
> either V1 exposes an equivalent priority/LTR hook, or LTR+C2 lives on the V0/0.4.1
> surface (XFORMERS prefill + CUDA decode, like §2.2) while *B0+C2* (no LTR) is where the
> clean all-Triton V1 kernels are exercised.

**Enable the backend (get this exactly right):**
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` — **not** `TRITON`. The selector matches
  the enum *name* exactly (`selector.py:29`, member `TRITON_ATTN_VLLM_V1` in
  `platforms/interface.py:39`); a bare `TRITON` returns `None` and **silently falls back
  to FlashAttention** (`cuda.py:212-223`; default on sm≥80 is FlashAttention, and there is
  no XFORMERS backend in v1 at all). sm_86 is supported with no arch gate
  (`triton_attn.py:24` head 128; `prefix_prefill.py:13,17` even has a Turing fallback).

**Prefill / context kernel — `vllm/attention/ops/prefix_prefill.py::_fwd_kernel`**
(the same kernel `context_attention_fwd`, `:718`, that we FP8-patched):

| line | code (v1) | BLASST action |
|---|---|---|
| ~134–216 | `for start_n in range(0, cur_batch_ctx_len, BLOCK_N):` | the KV-block loop |
| ~189 | `m_ij = tl.maximum(m_i, tl.max(qk, 1))` | `block_max = tl.max(qk,1)` here |
| ~191 | `p = tl.exp(qk - m_ij[:, None])` | **guard** (scalar predicate on `block_max` vs `m_i`) |
| ~199 | V load `v = tl.load(...)` | skip the V load |
| ~213 | `acc += tl.dot(p.to(v.dtype), v)` | skip the `tl.dot` (the cost) |

(ALiBi variant `_fwd_kernel_alibi:562-632` needs the same edit if ALiBi models are used.)

**Decode kernel — `vllm/attention/ops/chunked_prefill_paged_decode.py::kernel_paged_attention_2d`**
(inline at `:25`; this is the real GQA decode kernel — *not* `triton_decode_attention.py`):

| line | code (v1) | BLASST action |
|---|---|---|
| ~114–191 | the KV-block loop | — |
| ~143–150 | V load + scaled dequant | **must be MOVED** (see below) |
| ~159 | `S = tl.dot(Q, K) …` | `block_max = tl.max(S)` |
| ~172 | `m_j = tl.maximum(m_i, block_max)` | **guard** here |
| ~175 | `P = tl.exp(S - m_j[:, None])` | skip if guarded |

> **Decode needs a V-load reorder to actually save bandwidth.** As written the kernel
> loads V (`:143`) *before* it computes `S`/`m_j` (`:159`/`:172`), so a naive guard skips
> only the `tl.dot`, not the (dominant) V **bandwidth**. Move the V load + dequant to
> *after* `m_j` and inside `if not skip`. Decode's silver lining: each program handles one
> query row per (seq, head), so the predicate is naturally scalar — the "all rows agree"
> tax that caps prefill sparsity largely vanishes here (the GQA-group agreement still
> applies: `num_queries_per_kv_padded ≥ 16` rows, `:280`).

**τ plumbing (v1):** there is **no `--sparse-attention` EngineArg** in v1; thread `τ` as
env `VLLM_BLASST_TAU_PREFILL` / `VLLM_BLASST_TAU_DECODE`, read in the kernel wrappers
(`triton_attn.py:59/107` → `chunked_prefill_paged_decode(...):180` → the two kernels).
This avoids surgery on `VllmConfig`.

### 2.2 — 0.4.1 fork (`ltr/vendor/vllm-ltr/`) — swap-stack integration, narrower

On the 3090 the 0.4.1 active backend is **XFORMERS**, and a **fresh-prompt prefill goes
to compiled `xops.memory_efficient_attention_forward`** (`xformers.py:388`) — **not
modifiable** without patching xformers. So BLASST-in-Triton on 0.4.1 covers only:

**(a) cached-context / resumed prefill — `vllm/attention/ops/prefix_prefill.py::_fwd_kernel`**
(the one live, modifiable Triton kernel; the same one we FP8-patched):

| line | code (0.4.1) | BLASST action |
|---|---|---|
| ~109 | `for start_n in range(block_min, block_max, BLOCK_N):` | KV-block loop |
| ~121 | `m_ij = tl.max(qk, 1)` | `block_max` here |
| ~125 | `m_i_new = tl.maximum(m_i, m_ij)` | **guard** here |
| ~137–145 | `p = tl.exp(...)`, V load, `acc += tl.dot(p, v)` | skip `exp` + V-load + dot |

This kernel runs **only when a cached prefix exists** (`block_tables.numel()!=0`,
`xformers.py:280-305`): chunked-prefill / prefix-cache / **swap-preemption resume**. Under
the LTR **swap** regime that resume path *is* exercised (PATCHES.md §4e), so C2 composes
with the swap stack — but it does **not** accelerate a fresh long prefill (that's
xformers). **Never skip the second loop `:159-201`** (the fresh current-query K/V — the
"recent window" rail in this kernel).

**(b) decode — CUDA, not Triton — `csrc/attention/attention_kernels.cu`**
There is **no Triton decode kernel in 0.4.1** (only `paged_attn.py`, `prefix_prefill.py`,
and the dead `triton_flash_attention.py`). Decode is the CUDA `paged_attention_kernel`
(`:97`). It is actually the *cleaner* place for the skip — one query row per (seq, head)
⇒ a naturally scalar predicate. Add the guard to the **V-accumulation loop `:333-380`**,
comparing each block's max logit to the global `qk_max` already computed at `:281`. This
is a **modest CUDA edit + recompile** (we build from source already), *not* a "reuse the
Triton guard" and *not* a from-scratch Triton decode.

**τ plumbing (0.4.1):** prefill = 4 Python layers
(`xformers.py:293` → `paged_attn.forward_prefix:177` → `context_attention_fwd:633` →
`_fwd_kernel[grid]:713`). Decode (CUDA) = extend the C++ op signature + bindings:
`csrc/ops.h`, `csrc/pybind.cpp`, `_custom_ops.py`, launcher/kernel — a **rebuild**, not a
Python edit.

### 2.3 — v0.25.0 (the current stack — smoke-validated on the 3090, `docs/V025_SMOKE.md`)

vLLM **unified** its Triton attention in the 2026-03 rewrite: v0.25's `TRITON_ATTN` backend
no longer dispatches to `prefix_prefill.py` + `chunked_prefill_paged_decode.py` (the 0.8.5
world in §2.1) — it calls **one** kernel, `unified_attention` (prefill *and* decode).
**This is the current BLASST insertion point:**

- File: `vllm/v1/attention/ops/triton_unified_attention.py`, `def kernel_unified_attention`
  (`:179`); imported + launched by `v1/attention/backends/triton_attn.py:43,:668`.
- KV-block loop: `for j in range(loop_lo, loop_hi):` (`:420`).
- Scores: `S += tl.dot(Q, K) * …` (`:540-542`) → take `block_max = tl.max(S, 1)` here.
- **Guard** the `P = tl.exp(S - m_ij)`, the `L`/`m` update, the `acc *= alpha` (`:562`) and
  the `acc += tl.dot(P, V)` (`:582-584`) — the same scalar/CTA-uniform skip idiom as §1.
- `tl.exp` (natural log; `:740,:756` confirm) → τ is natural-log, no `log2(e)`.
- To actually *reach* this kernel on the 3090 you must select `TRITON_ATTN` (default is
  `FLASH_ATTN`, compiled) and use **non-fp8** KV (fp8 on Triton is SM89-gated, §9.1) — i.e.
  compose C2 with **bf16 or TurboQuant** KV, not fp8 (matches the C1⊕C2 constraint).

**Route B — INT8K–V4 Ampere-compatible TurboQuant variant** (avoids the `float8e4nv` that
kills `turboquant_k8v4` on sm_86, per the external review). The machinery already exists:
`vllm/v1/attention/ops/triton_turboquant_store.py` has `_tq_fused_store_fp8` (the fp8 **key**
path — the sm_86 blocker) *and* `_store_quantized_value` (uniform **INT** quant: min/max →
scale → `(v-min)/scale` → clamp → bit-pack → store scale/zero — pure integer, runs on
Ampere). INT8K-V4 = clone the value-path INT quant to **8-bit for keys** in the store kernel
+ the matching unpack in `triton_turboquant_decode.py`, add per-group key scale/zero. Keeps
"high-fidelity key + low-bit value" without hardware fp8. It is a *new* codec (re-measure
accuracy/latency), not the paper's `k8v4`.

## 3. Config surface (mirror TRT-LLM)

- **v1:** env `VLLM_BLASST_TAU_PREFILL` / `VLLM_BLASST_TAU_DECODE` (no EngineArg exists).
  **0.4.1:** may add `--sparse-attention skip-softmax` +
  `--skip-softmax-threshold-{prefill,decode}` (0.4.1 arg-utils is easy to extend), or the
  same two env vars for parity. `τ` is a **natural-log** scalar per phase (kernels use
  `tl.exp` / CUDA `expf`) — **do not** apply a `log2(e)` factor.
- Report **achieved** skipped-block fraction (a kernel counter or a profiling pass), not
  just the target `τ`.

## 4. FP8-KV interaction (compose with C1) — differs by engine

BLASST's skip decision uses `qk` (post-dequant), so it composes cleanly in both engines;
but the dequant code the plan cites differs:

- **v1:** dequant is **native + scaled** — `(_load.to(f32) * tl.load(k_scale)).to(q.dtype)`
  (`prefix_prefill.py:163-166` K / `:207-210` V; decode `chunked_prefill_paged_decode.py:137-150`).
  The uint8→fp8 `.view` is in the wrapper (`:747-759`); `kv_cache_dtype=='auto'` is rejected
  (`:761-764`). We add **no** dequant — just the skip guard after the K dequant/`qk`. And
  because fp8 forces Triton here (§2.1), the **2×2 ablation FP16/FP8-KV × BLASST off/on**
  runs entirely on one backend.
- **0.4.1:** the *prefix* (Triton) path uses our **bare-bitcast** patch
  (`x.to(tl.float8e5, bitcast=True).to(q.dtype)`, `prefix_prefill.py:111-112,141-142`). The
  *decode* FP8 dequant is **CUDA** (`fp8_e5m2_unscaled::vec_conversion`,
  `attention_kernels.cu:226-241` K / `:351-360` V) — so the fp8×BLASST decode cell needs
  the CUDA edit, not the Triton patch. (`ENABLE_FP8_E5M2` is compiled — CUDA 12.1 build,
  `cmake/utils.cmake:101-102` — so the CUDA fp8 decode path is live, not the assert-false
  stub.)

Order (both): dequant-K → `qk` → skip test → (if kept) dequant-V → `p·V`. This yields the
2×2 ablation to test whether the memory (C1) and compute (C2) levers stack.

## 5. Correctness tests (before any speed claim)

- **Dense vs sparse output error** at several `τ`: max/mean abs diff of attention output
  and of final logits vs `τ=∞` (dense). Must be small and NaN-free.
- **Long-context retrieval** (needle-in-haystack / RULER-lite) — the case where wrongly
  skipping a block actually loses information.
- **Numerical stability**: running-max/denom correctness when leading blocks are skipped
  (the sink/first-block effect — never skip block 0 / recent window as a safety rail);
  in chunked prefill the running-max resets per chunk, so the rail is per-chunk.
- Unit-test the **scalar skip predicate** (CPU) separately from the Triton/CUDA kernel —
  this is the piece most likely to silently degrade to mask-to-zero (§1 constraint 2).
- **Causal mask & partial final block.** `block_max` and the skip bound must be taken over
  **legal positions only** — the ragged final KV block, causal masking (different valid
  key ranges per query row in prefill), and sliding-window/ALiBi all change which keys are
  in-bounds. A `block_max` that includes masked (−inf) or out-of-range keys will either
  over-skip a live block or wrongly admit a dead one. Test the diagonal/last-tile case
  explicitly; the existing masked positions must stay masked, not be counted in the max.

## 6. Benchmark (what to report)

- **Prefill vs decode separately**: TTFT (prefill-dominated), TPOT (decode) — BLASST helps
  both but differently; sweep `τ`.
- **Achieved sparsity** (skipped-block fraction) on the x-axis, not target `τ`.
- **Quality**: GSM8K (generation) + a long-context task; reuse `ltr/quant/eval_gsm8k.py`.
- **Regime**: BLASST helps most when attention is the bottleneck (long context / large
  batch) — i.e. the **compute-bound** regime where C1 was neutral. Test there, and in the
  2×2 with C1 (§4). Same ladder as C1: B1 → C2 → (C1⊕C2).

## 7. Staged plan (v1-first — audit-revised)

The audit inverts the original decode-first-on-0.4.1 ordering: prototype on **v1** (both
kernels are Triton), then integrate into the 0.4.1 swap stack — **but only after Gate 0
proves the V1 engine actually stands up on the 3090** (our prior "v1" grid was V0, §9.1).

| stage | scope | gate |
|---|---|---|
| **Gate 0** — stand up the real V1 engine — ✅ **DONE (v0.25, `docs/V025_SMOKE.md`)** | vLLM **0.25.0** boots a real V1 engine on the 3090; **V1 priority scheduling now works** (PR #19057) so LTR+V1 is NOT deadlocked (the 0.8.5 blocker is gone); native TurboQuant runs. C2's insertion point is v0.25 `kernel_unified_attention` (§2.3), and the exact env-gated BLASST patch is written (`c2/apply_blasst_kernel_patch.py`). Measured GQA-realized decode sparsity ~18.5% @ τ=6 → ~1.1×. | ✅ V1 boots · ✅ priority/LTR works · ✅ TurboQuant works · ⧗ patch-into-spawned-workers (sitecustomize/plugin) is the last step |
| **C2a** — algorithmic screen | Python/Triton microbench of the scalar skip predicate on captured Q/K; sweep `τ` → sparsity vs output-error/quality. **No serving speed claim.** Engine-agnostic — can run before Gate 0. | Is there a τ with high sparsity + acceptable error, *given the scalar/GQA agreement tax*? |
| **C2b** — v1 decode Triton kernel | `kernel_paged_attention_2d` (reorder V-load); FP16 KV first; correctness vs dense; then real 3090 decode latency | dense-vs-sparse error small + **real** decode speedup? |
| **C2c** — v1 prefill/context | `prefix_prefill.py::_fwd_kernel`; long-context prefill + retrieval intact | prefill TTFT speedup + retrieval intact |
| **C2d** — v1 FP8 compose (2×2) | thread `τ` through the fp8-forced Triton path; FP16/FP8 × on/off | do C1 (memory) and C2 (compute) stack? |
| **C2e** — 0.4.1 swap-stack integration | `prefix_prefill.py` (resumed prefill) + the **CUDA** decode edit; run under LTR swap | does C2 compose with swap/LTR end-to-end? |

**Start at C2a.** Do NOT begin with FP8 + sparse + swap/recompute at once — isolate
errors. On 0.4.1, remember fresh-prompt prefill stays on xformers (out of scope).

> **C2a DONE (2026-07-11) — `c2/blasst_screen.py`, real Llama-3.1-8B L16, 1536-tok prefill.**
> Sweeping τ on captured Q/K/V: **the algorithm works** (per-query skip = **42% of blocks
> at 3.2% output rel-error**, τ=6; 72% at 12%, τ=4 — a real knee). **But the CTA-uniform tax
> is severe and scales with BLOCK_M:** at τ=6 realizable sparsity collapses **per-query 42%
> → M=16 6.7% → M=64 1.3%** (a causal-prefill tile's queries span too many positions to
> agree). This **empirically confirms §9's audit warning** and **fixes the kernel strategy:
> decode-first** — decode is one query/program, so per-query *is* the achievable (full 42%,
> no tile tax); prefill only pays off with a **small BLOCK_M (16)** and even then needs a
> looser τ. So **C2b (v0.25 decode guard in `kernel_unified_attention`) is the highest-value
> first kernel**, not prefill. Data: `results/summaries/c2a_blasst_screen{,_m16}.json`.

> **C2b DONE (2026-07-11) — `c2/blasst_decode_kernel.py`, §8's open risk RESOLVED (positive).**
> A standalone Triton decode-attention kernel with the scalar skip guard, benchmarked on the
> RTX 3090 (256 programs, N=2048, BLOCK_N=64, τ=4): **skip gives a real speedup that scales
> with sparsity — 1.33× @ 50%, 1.75× @ 90% — with NO penalty at 0% sparsity (1.00×)** and
> **correctness rel-err 2.9e-4**. Sub-linear (ceiling ~2×) because Q·K still runs every block
> (half-FLOP ceiling, §1). At C2a's realistic ~42% decode sparsity → **~1.3× decode-attention
> speedup**. (Triton 3.x rejects `continue`; use a scalar `if` around the block body.) So the
> Ampere risk in §8 is answered: **BLASST does net-speedup on the 3090 for decode.** Remaining
> = production: thread τ into vLLM's `unified_attention` decode path (§2.3) + e2e TPOT.
> Data: `results/summaries/c2b_decode_bench.json`.

## 8. Honest risks (Ampere)

- **Scalar-predicate / GQA agreement caps sparsity.** Because the skip must be tile-uniform
  (§1), a single loud row or GQA head keeps the block. Prefill (BLOCK_M=128) is hit hardest;
  decode fares better (single query row). Realized sparsity may be well below the paper's.
- **Half-FLOP ceiling.** `Q·K` can't be skipped, so the compute ceiling is ~½ attention
  FLOPs + V bandwidth — don't model it as "skip the whole block."
- **Triton on Ampere may not net-speedup** even when correct: branch divergence, the extra
  max/compare, and lost `tl.dot` tensor-core / `num_stages` pipelining can offset skipped
  work at low sparsity. The v1 decode kernel is already a small (BLOCK_SIZE=16), non-split
  dot — bandwidth-bound — so the win rides on skipped **V bandwidth**, not FLOPs. **Measure
  real latency; report the achieved-sparsity threshold at which it wins.** If it never wins
  on the 3090, that is itself a reportable result (the method wants the newer FMHA / A100+
  kernels TRT-LLM targets).
- **0.4.1 fresh-prefill is unreachable** (xformers, compiled). 0.4.1 C2 = resumed-prefill
  (Triton) + decode (CUDA) only; don't claim fresh-prefill prefill savings there.
- The paper's/TRT-LLM's headline speedups (1.36–1.4×) are on **GB200 at 128K context** —
  do not port those numbers; measure our own.
- Skipping the wrong block breaks long-context retrieval → always keep sinks + recent
  window; validate on a retrieval task.

## 9. Audit & forensics

### 9.1 Runtime forensics — the "v1" grid was actually V0 + XFORMERS (external-review trigger)

**Before trusting any "we already run v1/Triton/fp8" claim: we don't.** A runtime check
(prompted by an external review questioning whether the 3090 can run fp8 on the v1 Triton
backend) read the actual server startup banners of the pressured grid
(`/tmp/grid_pressured/srv_*_v1p.log`):

- **Engine = V0**, not V1 — every "v1p" server logged `Initializing a V0 LLM engine (v0.8.5.post1)`.
- **Backend = XFORMERS** (CUDA), not Triton: `Using XFormers backend` (`cuda.py:228`).
- **Forced fallback**, two independent causes: `VLLM_ATTENTION_BACKEND=XFORMERS is not
  supported by the V1 Engine. Falling back to V0` (B0) and `--scheduling-policy is not
  supported by the V1 Engine. Falling back to V0` (B1/C1 — the LTR knob).
- **fp8 was real but ran on the V0 CUDA E5M2 path**, not Triton (`c1_v1p`:
  `kv_cache_dtype=fp8`, `num_gpu_blocks 4986→1024`). On sm_86 this path works fine.
- **Preemption = RECOMPUTE** (confirmed: `is preempted by PreemptionMode.RECOMPUTE …
  total_num_cumulative_preemption=201`), the default for single-seq sampling
  (`scheduler.py:1758`). No log ever shows a V1 engine or a Triton/Flash backend.

**Consequences.** (i) The *science survives*: the "modern recompute stack" contrast is
genuine — RECOMPUTE preemption is real and fp8 capacity reduces it — but the **engine
label "vLLM v1" is wrong**; it is **vLLM 0.8.5 *V0* + XFORMERS (recompute)**. Results/docs
that say "v1" must be relabeled (follow-up). (ii) C2's "C1 fp8 is already on the Triton
path" was **false** — corrected in §2.1. (iii) The **LTR-priority ⇄ V1 tension** (§2.1) is
now a hard, evidenced constraint, not a hypothetical: making C2 run on real V1-Triton is a
Gate-0 experiment we have **not** yet performed.

### 9.2 Kernel audit — findings folded in

Two agents read the real source, one per engine (0.4.1 fork; v1 `.venv-v1fp8`,
`vllm-0.8.5.post1` / `triton-3.2.0`). Verdict: **algorithmically feasible and fp8-composable
on the 3090, but §2's original file/loop/env targets were almost all wrong.** Applied above:

1. **[Blocker, both] `triton_flash_attention.py` is dead code** on the 3090 (ROCm + MLA
   only) — retargeted to `prefix_prefill.py` (+ v1 `chunked_prefill_paged_decode.py`, +
   0.4.1 CUDA `attention_kernels.cu`).
2. **[Blocker, v1] env is `TRITON_ATTN_VLLM_V1`**, not `TRITON` (bare form silently falls
   back to FlashAttention). Default 3090 backend is FlashAttention, not XFORMERS.
3. **[Blocker, v1] fp8 forces Triton** — v1 FlashAttention raises on fp8-KV without sm_90;
   turned from an unstated assumption into C2's motivation (C1 already runs on Triton).
4. **[Blocker, 0.4.1] fresh prefill = xformers (compiled, unmodifiable); no Triton decode**
   — decode is CUDA. Scope narrowed to resumed-prefill (Triton) + CUDA decode.
5. **[Correctness, both] real kernels use `tl.exp` (natural log), not `exp2`** — `τ` is
   natural-log, no `log2(e)`.
6. **[Correctness, both] predicate must be scalar/CTA-uniform** or it degrades to
   mask-to-zero (no FLOP savings); GQA agreement caps sparsity; `Q·K` unskippable (~½ ceiling).
7. **[Correctness, v1] decode must reorder the V-load** (currently before `m_j`) to save V
   bandwidth.
8. **[Fact, both] FP8 dequant differs** — v1 native *scaled*; 0.4.1 prefix = bare bitcast
   (Triton), 0.4.1 decode = CUDA. §4 split accordingly.
9. **[Confirmed] sm_86 support** for the v1 Triton backend (no arch gate; Turing fallback).

Full agent reports (file:line evidence tables) are in the task outputs; key files:
v1 `triton_attn.py` / `chunked_prefill_paged_decode.py` / `prefix_prefill.py` /
`platforms/cuda.py` / `attention/utils/fa_utils.py`; 0.4.1
`ops/prefix_prefill.py` / `csrc/attention/attention_kernels.cu` / `backends/xformers.py` /
`backends/selector.py`.

## References
BLASST arXiv:2512.12087 · TRT-LLM Skip-Softmax (Hopper/Blackwell, `SkipSoftmaxAttentionConfig`):
nvidia.github.io/TensorRT-LLM/features/sparse-attention.html + developer.nvidia.com blog.
Our FP8 prefix-kernel patch: `ltr/vendor/PATCHES.md §4e`. C-tier context: `docs/C_TIERS.md`.
