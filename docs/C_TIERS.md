# C_TIERS.md — the C1–C4 KV-optimization plan (design for group review)

The C-tiers add KV-cache techniques **on top of B1** (LTR scheduling), each
ablated on/off, tracking **latency + preemptions + accuracy**. This doc is the
implementation plan + integration map for the group to review and split before
we write code. Prior context: `results/RESULTS.md` (B0/B1/B1a) and the roadmap in README.

## 0. Platform decision (per-tier — revised after the C2 kernel audit)

The C-tiers do **not** all live on one engine. The right platform depends on what each
tier needs to touch; forcing everything onto 0.4.1 for swap would sacrifice C2's
feasibility. Plan of record:

| tier | primary engine | why |
|---|---|---|
| **B1a** | vLLM **0.4.1** fork (swap) | the paper's 2.1× reproduction — needs **swap** preemption |
| **C1** | **three** stacks (updated) | cross-stack proof — 0.4.1 swap + 0.8.5-V0 recompute (both fp8 proxy) **+ v0.25 V1 native TurboQuant** (real `turboquant_4bit_nc`, 2.83× cap, GSM8K-neutral; see `docs/V025_SMOKE.md` + RESULTS §C1-native). NB: KVmix remains an accuracy probe only, not a serving codec |
| **C2** | **"v1"-first** (see below) — modifiable Triton prefill+decode; then 0.4.1 swap-stack integration | BLASST needs an editable attention kernel; 0.4.1's fresh-prefill is compiled xformers. **Pending Gate 0** (`docs/C2_BLASST_PLAN.md §7`) |
| **C3** | TBD (offloading API) | depends on the connector/offload hook |
| **C4** | v1-primary, 0.4.1 compat appendix | coordinator over the v1 KV-connector API |

> **⚠ "v1" here is a label, and it is currently inaccurate — read this.** Our `.venv-v1fp8`
> "v1" stack has, to date, **only ever run vLLM 0.8.5.post1's *V0* engine + XFORMERS +
> recompute** (forced off V1 by `--scheduling-policy priority` and
> `VLLM_ATTENTION_BACKEND=XFORMERS`; the V1 engine / Triton were never exercised —
> `docs/C2_BLASST_PLAN.md §9.1`, RESULTS.md §Setup). So "C2 on v1" means *stand up the
> real V1 + `TRITON_ATTN_VLLM_V1` engine at Gate 0 first*. Note the **LTR ⇄ V1 tension**:
> `--scheduling-policy priority` (B1's LTR knob) forces V0, so *LTR + V1-Triton* may not
> coexist on 0.8.5 — `B0+C2` may be where the clean all-Triton kernels run, while `LTR+C2`
> stays on the V0/0.4.1 surface (XFORMERS prefill + CUDA decode).

Where an engine lacks a method's support we **backport/patch** (e.g. C4's coordinator
would backport a KV-connector hook). Build/patch recipe: `ltr/vendor/PATCHES.md`.

> **Update (post-C1).** C1's *latency* study was ultimately run **cross-stack** — the
> 0.4.1 swap engine AND a modern recompute stack (isolated `.venv-v1fp8` = 0.8.5
> **V0**+XFORMERS, recompute — the "v1" label is shorthand, see §0 note) — under
> an **emulated memory-bound pool cap**. This is the *stronger* result: KV capacity
> (fp8) avoids saturation on **both** engines and converges swap and recompute to the
> same latency (RESULTS.md §"Thesis" / §"C1 cross-stack grid"). The 0.4.1-only plan
> here still governs the C2/C3/C4 build-out; C1 additionally carries the cross-stack
> evidence. **All C1 *latency* numbers use fp8 KV (the naive-quant baseline below) as
> a conservative proxy — the TurboQuant+KVmix kernels remain unported to 0.4.1** (§3
> Port reality), so full-C1 latency is future work; only its *accuracy* is characterized.

## 1. What each C-tier is

| tier | technique | source | calibration? |
|---|---|---|---|
| **C1** | KV **quantization** = TurboQuant [1] (random-rotation + scalar quant, data-oblivious) + KVmix [2] (per-layer gradient-importance mixed precision, 1–4 bit) | [1] arXiv:2504.19874, [2] AAAI'26 `github.com/LfLab-AI/KVmix` | TurboQuant: **no**; KVmix: **yes** (gradient profiling) |
| **C2** | attention **sparsity = BLASST** — online-softmax block skipping inside the attention kernel; **COMPUTE axis** (speeds up prefill/decode; does NOT shrink KV — complementary to C1's memory axis). We reimplement it in vLLM's Triton kernel for the 3090 — see **`docs/C2_BLASST_PLAN.md`** | [4] arXiv:2512.12087 | no (training-free) |
| C3 | head-wise **offloading** = HeadInfer | [5] arXiv:2502.12574 | no |
| C4 | all three under a KV coordinator (backport v1 KV-connector API [7][10]) | — | — |

`--kv-cache-dtype fp8` (0.4.1 built-in, uniform E5M2) is **NOT C1** — it is only a
**naive-quant baseline** to show the value of TurboQuant's rotation + KVmix's
importance-awareness. It's already wired into `serving/bench_reference.sh`
(`KVDTYPE=fp8`).

> **C2 = BLASST (compute-axis sparsity).** BLASST (arXiv 2512.12087) skips the softmax
> + P·V for negligible KV blocks *inside* the online-softmax loop — **compute/bandwidth
> savings, NOT KV reduction** (the full cache stays resident). So it is **complementary
> to C1**, not the same axis: C1 (quant) relieves the **memory-bound** regime (fewer
> preemptions); C2 (BLASST) attacks the **compute / attention-bound** regime (long
> context, attention O(n²)) — exactly where C1 was neutral. Availability (verified):
> BLASST is productized in **NVIDIA TensorRT-LLM as "Skip Softmax"**
> (`SkipSoftmaxAttentionConfig`), **officially Hopper/Blackwell only** and
> TRT-LLM-engine-bound (no reusable ABI — not callable from vLLM). The *algorithm* is
> portable (softmax thresholding, not hardware-gated by tensor cores), so **we
> reimplement it in vLLM's Triton attention kernel** to run on the 3090 and compose with
> LTR + FP8-KV. Full kernel design + files + staging + risks: **`docs/C2_BLASST_PLAN.md`**.
> BLASST/Skip-Softmax is prior art we cite (`literature/summaries/kv-4-blasst.md`); our
> contribution is the vLLM-native Triton port + its LTR/FP8 interaction.

## 2. Integration surface in vLLM 0.4.1 (the fp8 path is our template)

fp8 threads one `kv_cache_dtype` **string** + one `kv_scale` **float** from Python
to two CUDA dispatchers. A new codec follows the same seam.

**Python only — no recompile (3 spots to accept a new codec string):**
- `vllm/engine/arg_utils.py:176` — add to `--kv-cache-dtype` `choices=['auto','fp8']`.
- `vllm/config.py:327-345` — accept it in `_verify_cache_dtype`.
- `vllm/utils.py:26-31` — map codec → storage torch dtype (`torch.uint8`).

**Write path (quantize-on-store):**
- CUDA kernel `csrc/cache_kernels.cu:201-214`; host dispatch `:257-275` (add `else if (kv_cache_dtype=="turboquant")`).
- Python glue (no recompile): `vllm/attention/ops/paged_attn.py:62-80` `write_to_paged_cache`; XFormers call site `vllm/attention/backends/xformers.py:193-204`.

**Read path (dequant-during-attention):**
- CUDA kernel `csrc/attention/attention_kernels.cu:226-241` (K) + `:351-366` (V); host dispatch `:755-777` (v1) / `:952-974` (v2).
- Python glue: `vllm/attention/ops/paged_attn.py:82-160` `forward_decode`; XFormers read `xformers.py:253-266`.

**The fp8 codec header to clone:** `csrc/quantization/fp8_e5m2_kvcache/quant_utils.cuh`
(unscaled, NVIDIA) or the **scaled** `csrc/quantization/fp8/amd_detail/quant_utils.cuh`
(`scaled_vec_conversion(x, scale)` — the closer analog for a scalar-quant codec).

**Per-layer seam (for KVmix):** KV cache is already allocated per-layer
(`vllm/worker/cache_engine.py:64-71`); the per-layer `kv_scale` mechanism
(`vllm/model_executor/models/llama.py:126,:430-448` → `model_runner.py:195-198`)
is the precedent — a per-layer codec/bit-width follows the identical path plus
per-layer allocation + a CUDA dispatch switch.

**Caveat:** the Triton prefix kernel doesn't handle quantized KV
(`xformers.py:234-236`); we already run with `enable_prefix_caching=False`, so OK.

## 3. Staged plan for C1

**Stage 1a — TurboQuant "rotation + fp8", ZERO CUDA changes (start here).**
TurboQuant's key idea is a random orthogonal rotation that makes coordinates
Beta-distributed so scalar quant is near-optimal. The rotation mixes `head_size`
dims — do it in **torch/Python**: rotate K & Q (dot-product invariant) and V
before `write_to_paged_cache` (`xformers.py:189-204`), inverse-rotate the decode
output (`:254`); store the rotated KV with the **existing fp8 path**. Deliverable:
"rotated-fp8" vs "plain-fp8" accuracy at equal 2× compression — validates the
rotation with no kernel work. Reuse a Python TurboQuant ref (yashkc2025/turboquant).

**Stage 1b — full TurboQuant scalar quant in CUDA.** Clone the fp8 header →
implement Lloyd-Max per-coordinate quant at 2.5–3.5 bit; add the `"turboquant"`
dispatch branch; recompile (we build from source already). Gets 4–7× compression
vs fp8's 2×.

**Stage 2 — KVmix per-layer mixed precision.** Offline gradient profiling →
per-layer bit allocation; extend `cache_engine.py:64-71` to per-layer dtype/shape
and thread a per-layer codec id via the `kv_scale` precedent. Combine with the
Stage-1 rotation → C1 = rotation + per-layer scalar quant.

## 4. Metrics — C-tiers add ACCURACY

B0/B1 metrics (TTFT/TPOT/Nlatency/throughput/preemptions/KV-GB) **plus**:
- **Accuracy vs fp16 baseline** — quantization degrades quality. **Use a
  GENERATION-based metric, NOT teacher-forced perplexity:** we confirmed prefill PPL
  is *blind* to KV-cache quant (fp16 and fp8 give bit-identical PPL 17.900 — prefill
  attends raw K/V, not the cached quantized KV; RESULTS.md §C1 Stage-1a). Measure via
  greedy generation where each token attends the quantized cache — token-level
  agreement vs fp16, or downstream accuracy (`ltr/quant/eval_gsm8k.py`: GSM8K fp16
  86.0 % vs fp8 87.0 %, within noise). This is the C-tier's cost side; the win side
  is more GPU KV blocks → fewer preemptions → lower latency at high load.
- **KV compression ratio** (bits/token vs fp16) and the resulting **GPU-block
  count** (fp16 2145 → target 2× for fp8, 4–7× for TurboQuant on the 3090).

## 5. Bench integration

Add a **C1 arm** to `serving/bench_reference.sh` (alongside `fcfs`/`fifo`/`opt`):
the `opt` scheduler + the C1 codec, e.g. `SCHED=opt-xxx KVDTYPE=turboquant`.
Sweep the same rates (2/4/8 on 3090; 5–60 on A100) and compare **C1 vs B1a-opt**
at each rate, on latency AND accuracy. Record in `results/RESULTS.md` as `C1`.

## 6. Track split (group, trackable)

- **CUDA codec** — clone the fp8 header, Stage-1b/2 kernels, recompile, dtype plumbing (§2 CUDA spots).
- **Python integration** — the 3 Python plumbing spots, XFormers rotation hook (Stage 1a), per-layer cache_engine (Stage 2), the C1 bench arm.
- **Evaluation** — accuracy harness (perplexity/logit-MSE vs fp16), compression measurement, RESULTS.md write-up, ablations (rotation on/off, per-layer vs uniform).

## References
[1] TurboQuant arXiv:2504.19874 · [2] KVmix AAAI'26 `github.com/LfLab-AI/KVmix` ·
[4] BLASST MLSys'26 · [5] HeadInfer arXiv:2502.12574 · [7] PagedAttention/vLLM ·
[10] LMCache arXiv:2510.09665. Full list: `docs/REFERENCES.md`.
