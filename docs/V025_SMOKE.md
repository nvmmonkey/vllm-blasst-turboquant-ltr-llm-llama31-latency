# V025_SMOKE.md — vLLM 0.25.0 on the RTX 3090: can we host LTR + TurboQuant-C1 + BLASST-C2 on one V1 engine?

**Answer (2026-07-11): yes at the boot+generation level.** A fresh, isolated
`.venv-v025` (vLLM 0.25.0 + torch 2.11.0+cu130 + FlashInfer 0.6.13) runs the **real V1
engine** on the 3090 (sm_86 / WSL2), with **V1 priority scheduling** (the LTR knob) and the
**native TurboQuant KV backend** both working. Scripts: `serving/v025_smoke/`.

This does **not** touch `.venv` / `.venv-v1fp8` / the 0.4.1 fork. It is why the C-tier
platform is now per-tier (see `docs/C_TIERS.md §0`): B1a stays on 0.4.1 (swap); the *new*
C1/C2 work has a cleaner home on v0.25 V1.

## Why this matters

Our prior "modern" stack (`.venv-v1fp8`, vLLM 0.8.5) never actually ran V1 — it fell back
to **V0 + XFORMERS** because `--scheduling-policy priority` (LTR) and `XFORMERS` are both
V1-unsupported in 0.8.5 (`v1/engine/processor.py:212` even hard-raises "V1 does not support
priority yet"). See `docs/C2_BLASST_PLAN.md §9.1`. vLLM merged **V1 priority** in PR #19057
(2025-06-23) and **upstreamed a native TurboQuant KV backend** — so on v0.25 the LTR
deadlock is gone and C1 can be the *real* TurboQuant codec instead of our fp8 2× proxy.

## Environment blockers fixed (all non-hardware, no sudo)

| blocker | cause | fix |
|---|---|---|
| `torchcodec` fails to load `libavutil.so.57` | vLLM pulls a multimodal video dep; no system FFmpeg, no sudo | `uv pip uninstall torchcodec` — vLLM tolerates its absence for text models |
| `RuntimeError: freeze_support / bootstrapping phase` | V1 spawns the EngineCore as a child process | guard the entry point with `if __name__ == '__main__'` |
| `RuntimeError: UVA is not available` | `is_uva_available()` = `is_pin_memory_available()`; WSL2 reports pin memory off | `export VLLM_WSL2_ENABLE_PIN_MEMORY=1` (same knob we used on 0.8.5) |

## Smoke matrix — results (RTX 3090, Llama-3.1-8B-Instruct, max_model_len=2048, gpu_mem_util=0.85, enforce_eager)

| gate | `--kv-cache-dtype` / flag | result | evidence |
|---|---|---|---|
| V1 engine | `auto` (bf16) | ✅ | `Initializing a V1 LLM engine (v0.25.0)`; `Using FLASH_ATTN`; **36,640 tok** KV; coherent gen |
| V1 priority | `scheduling_policy=priority`, `priority=[1,0]` | ✅ | accepted priority, coherent gen → **LTR deadlock solved** |
| **C1 native** | `turboquant_4bit_nc` | ✅ | `Using TURBOQUANT backend`; **103,488 tok** KV = **2.82× bf16**; coherent gen, no NaN |
| fp8 (control) | `fp8` | ⚠️ | correctly routes to **FlashInfer** (confirms the backend-priority analysis), but FlashInfer JIT needs `nvcc`/`CUDA_HOME` (not installed) → needs a CUDA toolkit to run here |
| TQ k8v4 | `turboquant_k8v4` | ✗ | Triton compile error: `float8e4nv` unsupported on sm_86 (k8v4 has an fp8-key component) → on Ampere use the pure-low-bit **`_nc`** variants (`4bit_nc`/`3bit_nc`) |

**Backend priority observed** (matches `platforms/cuda.py:153`): non-fp8 →
`['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']` picks FLASH_ATTN; a
`turboquant_*` dtype forces `['TURBOQUANT']`.

## Honest boundaries (do not overclaim)

- This is a **boot + short-generation** smoke, **not** a quality or latency benchmark. The
  2.82× is a **KV-capacity** measurement (token budget at fixed GPU mem), not a speedup.
- `turboquant_4bit_nc` has **open upstream issues** (init/hybrid-alloc #41560, long-prefill
  OOM #40420). Validate **GSM8K quality + long-context + real TTFT/TPOT** before making it
  C1's default. Per vLLM's own TurboQuant study, `4bit_nc` is the memory-for-throughput
  sweet spot; `3bit_nc`/`k3v4_nc` risk reasoning/long-ctx accuracy.
- fp8 on this box additionally needs a CUDA toolkit for FlashInfer's JIT. TurboQuant did
  **not** need nvcc (its Triton kernels compiled fine), which is a point in its favour here.

## Reproduce

```bash
bash serving/v025_smoke/install_v025.sh          # fresh .venv-v025 (vllm 0.25 + flashinfer)
uv pip uninstall --python .venv-v025 torchcodec  # drop the multimodal dep
bash serving/v025_smoke/smoke_driver.sh          # runs the 5-gate matrix, greps evidence
```

Next (planned): wire our OPT-125M LTR ranker → V1 `priority` (B1 on v0.25); GSM8K + capped-KV
pressure micro-bench for `turboquant_4bit_nc` vs bf16 (C1-native); then BLASST-C2 on the
v0.25 Triton kernels. See `docs/C2_BLASST_PLAN.md` and `docs/C_TIERS.md §0`.
