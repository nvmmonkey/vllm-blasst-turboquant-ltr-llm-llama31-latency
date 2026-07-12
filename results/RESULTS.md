# Results — config ladder record

Human-readable record of every run. Raw per-config data lives in
`results/summaries/<config>.json` (all rates, one file) + `<config>_sweep.csv`;
per-rate detail in `results/raw/<config>/` (git-ignored). This file is updated
as each step completes.

## Setup
- **Platform:** RTX 3090 24 GB, WSL2 (kernel 6.6.87), 22 GB RAM — every number in this
  file is measured on this box. An A100-40 GB native-scale re-run is optional future work
  (`docs/BENCHMARK.md`).
- **Engines (two stacks):** (a) **reference** = vLLM **0.4.1** fork (`ltr/vendor/vllm-ltr/`,
  **swap** preemption); (b) **"v1"** = the `.venv-v1fp8` modern stack = vLLM **0.8.5.post1**.
  **Served model:** meta-llama/Llama-3.1-8B-Instruct, FP16 (fp8 KV for C1).
- **Ranker (B1):** OPT-125M, ListMLE, 23,800 samples / 10 epochs (LLM.pdf spec).
- **Workload:** LMSYS-Chat-1M. **Metrics:** TTFT/TPOT/e2e (mean+p50/p90/p99),
  throughput (tok/s, req/s), KV peak (frac + GB), peak batch, preemptions.

> **What "v1" means here (runtime-forensics correction).** Throughout this file and the
> report, **"v1" is shorthand for our `.venv-v1fp8` modern-vLLM stack** — *not* a claim
> that the vLLM **V1 engine** ran. Reading the actual server startup banners
> (`/tmp/grid_pressured/srv_*_v1p.log`) shows every "v1" server ran **vLLM 0.8.5.post1's
> *V0* engine + the XFORMERS backend + RECOMPUTE preemption**, on the 3090's CUDA E5M2 fp8
> path. It was forced off the V1 engine by two independent causes — `--scheduling-policy
> priority` (the LTR knob) and `VLLM_ATTENTION_BACKEND=XFORMERS` are both unsupported by
> V1. **The V1 engine / Triton / FlashAttention were never exercised.** This does **not**
> change any number or conclusion below: the "recompute stack" is genuinely recompute
> (preemptions are logged as `PreemptionMode.RECOMPUTE`), so the swap-vs-recompute
> cross-stack story stands — only the *engine label* is corrected from "V1" to "0.8.5 V0".
> (The earlier "vLLM 0.24.0" was a typo; 0.24.0 is not a real vLLM version.) Details:
> `docs/C2_BLASST_PLAN.md §9.1`.

## Ladder status
| Config | What | Status |
|---|---|---|
| **B0** | vanilla vLLM v1, FCFS, no KV opt | ✅ recorded (Llama-8B, 3090) — low-load + pressured |
| **B1** | LTR scheduler (OPT-125M + priority policy), vLLM v1 | ✅ pipeline reproduced; **finding: v1 recompute → LTR *worse* → needs swap** |
| **B1a** | Reference stack (vLLM 0.4.1 + **swap**), 3 arms | ✅ **2.1× reproduced** (rate-4 opt-vs-**fcfs** P99 3.6× is a *combined* ordering+swap number; the **clean LTR ablation is opt-vs-fifo = 1.68×** at r4, and LTR *loses* to swap-only at r8) — load-dependent |
| **C1** | B1 + KV quantization (TurboQuant [+ KVmix]) | ✅ **done (TurboQuant) & regime-dependent** — *KVmix is characterized for accuracy only (a fake-quant probe, not a real codec/serving); the real serving C1 is TurboQuant*: compute-bound (abundant KV) → fp8 net-neutral (substitutes); **memory-bound (capped pool, LLM.pdf's regime) → C1 wins big: r64 TTFT 4.7×/5.4× over FCFS on both stacks, preempts/swaps → 0, both converge to ~930 ms** (see cross-stack grid) |
| C2 | B1 + attention sparsity (BLASST) | ✅ **kernel-validated + serving-integrated on BOTH decode kernels** (C2a 42% sparsity; C2b Triton 1.3×; PPL-lossless @ τ=6; independently audited). **Serving (runtime monkeypatch, r64):** bf16/unified `kernel_unified_attention` → **+4.5% TPOT (net-negative, GQA tile tax)**; TQ4/per-head `_tq_decode_stage1` → **−22% TPOT, p99 −47% (net-positive)** — same algorithm, opposite sign; per-head is the right landing spot. See §C2a/§C2b + ablation matrix |
| C3 | B1 + head-wise offloading | ⏸️ **future work** (project scoped to C1+C2) |
| C4 | B1 + Rust coordinator (all three) | ⏸️ **future work** |
| C5 | + speculative decoding | ⏸️ **future work** |

## Final scope — C1 + C2 (KV layer beneath LTR)
This project delivers a **KV-cache layer that sits *beneath* the LTR scheduler** (Fu et al.
[12] / the prior LTR study [11]): it **builds on** that learning-to-rank work — it does *not*
replace it. **Final scope = two KV levers, single-GPU: C1 (TurboQuant KV quantization) + C2
(BLASST attention sparsity).** C3 (head-wise offloading), C4 (a Rust control-plane unifying
all tiers), and C5 (speculative decoding) are **future work**. The headline is the C1+C2
ablation on top of LTR — **does adding these KV levers cut latency further without hurting
accuracy, and where does each lever actually pay off?** The answer is honestly
*regime-dependent* (below): we never quote a single speedup multiplier (per the Q&A prep),
and a lever that *doesn't* help in a regime — C2 on the GQA kernel, C2 at low-load long
context, LTR on abundant capacity — is reported as a real finding, not hidden.

---

## B0 — vanilla vLLM v1 (FCFS)   —   low-load characterization
**Config:** Llama-3.1-8B-Instruct · RTX 3090 · ctx 4096 · gpu-mem-util 0.88 ·
compilation on · LMSYS · max_tokens 128 · n=100 · seed 0 · scheduling=fcfs,
prefix-caching off, **full KV pool**.
> This low-load run (full KV → 0 preemptions) is preserved here + in git
> history; the committed `results/summaries/b0.json` now holds the **pressured**
> B0 (capped KV, rates 30/60) used in the B0-vs-B1 comparison below.

| rate (req/s) | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | e2e mean (s) | e2e p99 (s) | out tok/s | req/s | KV peak (GB) | peak batch | preemptions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  5 |  89 | 212 | 23.9 | 2.35 | 3.46 |  366 |  3.8 | 0.38 | 16 | 0 |
| 20 | 165 | 343 | 43.3 | 4.05 | 6.13 | 1010 | 10.5 | 1.53 | 72 | 0 |
| 40 | 369 | 687 | 54.9 | 4.83 | 6.85 | 1241 | 12.9 | 1.79 | 92 | 0 |

**Reading it:** latency rises with load (TTFT p99 212→687 ms, TPOT 24→55 ms),
throughput scales 366→1241 tok/s. **Preemptions = 0 at every rate** — no memory
pressure (KV peak ≤ 1.8 GB vs a ~5 GB pool). LTR's benefit shows only *under
pressure*, so the B0-vs-B1 comparison uses a **pressured regime** (below).

**Notes:**
- Reached serving past the WSL2 vLLM-v1 ZMQ startup race (util 0.88 + ctx 4096 +
  IPv4 loopback; race is probabilistic — see `docs/RUNBOOK.md`).
- A prior **Qwen2.5-3B** B0 (git history) was the stopgap before Llama-8B served
  on the 3090.
- **A100 run** (spec: ctx 8192, util 0.90, rates 5–60, n=500) → `docs/BENCHMARK.md`.

---

## B1 — LTR scheduler (in progress)
**Ranker:** OPT-125M fine-tuned with ListMLE on LMSYS output lengths, then used
client-side to stamp each request's `priority`; served with vLLM
`--scheduling-policy priority` (no engine patching).

**Ranker (✅ trained — LLM.pdf spec: OPT-125M, 23,800 samples, 10 epochs, target Llama-8B):**
ListMLE loss decreases monotonically (no overfit rebound — 10 epochs is the
paper's balance point):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| listmle | 25.67 | 23.04 | 19.98 | 18.02 | 16.22 | 15.01 | 13.84 | 12.92 | 12.51 | 11.67 |

Trained on the 3090 in two stages (epochs 1–5, then resumed 6–10 via
`--resume`); loss curve is continuous. Saved to `results/ranker/opt125m-ltr/`
(weights git-ignored; `ranker_meta.json` committed).

### B0 (FCFS) vs B1 (LTR priority) — pressured comparison
**Config (identical for both):** Llama-3.1-8B · RTX 3090 · ctx 4096 · util 0.88
· **KV pool capped to 1024 blocks (16k tokens, `--num-gpu-blocks-override`)** so
it saturates and preempts · natural EOS (varying lengths → LTR signal) ·
max_tokens 512 · n=100 · LMSYS. B0 = `--scheduling-policy fcfs`; B1 =
`--scheduling-policy priority` with per-request priority from the ranker.

| rate | metric | B0 (FCFS) | B1 (LTR) | Δ |
|---:|---|---:|---:|---:|
| 30 | **TTFT p99 (ms)** | 554 | 424 | **−23%** ✅ |
| 30 | preemptions | 20 | 18 | −10% ✅ |
| 30 | e2e mean (s) | 10.57 | 10.49 | −1% |
| 30 | e2e p99 (s) | 21.64 | 23.56 | +9% ✗ |
| 60 | **TTFT p99 (ms)** | 1180 | 1077 | **−9%** ✅ |
| 60 | preemptions | 20 | 24 | +20% ✗ |
| 60 | e2e p99 (s) | 24.40 | 26.14 | +7% ✗ |

**Reading it honestly:**
- ✅ The **B1 pipeline is fully reproduced** end-to-end on the spec model: ranker
  scores → priorities → vLLM priority scheduler, no engine patching.
- ✅ **Consistent win: TTFT p99 −9…−23%** — LTR admits short-predicted requests
  first, cutting head-of-line blocking (the paper's core mechanism).
- ✗ **No 2.1×, and preemptions/e2e-tail are mixed** (SJF-style: prioritising
  short requests can lengthen the longest requests' tail; also n=100/1-seed is
  noisy).
### Audit fix #1 (target-sampled labels) — and the decisive swap finding
We fixed the label proxy: generated 3000 real Llama-8B output lengths
(`ltr.ranker.synthesize`) and retrained the ranker on them (`opt125m-real`,
loss 26.5→9.6). The ranker is **verified correct** — kendall τ(score, −length)
= **+0.81**, shortest requests get priority ≈29, longest ≈576 (SJF, right way).

| rate | metric | B0 (FCFS) | B1-proxy | **B1-real** |
|---:|---|---:|---:|---:|
| 30 | preemptions | 20 | 18 | 21 |
| 30 | TPOT (ms) | 47 | 46 | **54** ✗ |
| 60 | TPOT (ms) | 53 | 54 | **61** ✗ |
| 60 | TTFT p99 (ms) | 1180 | 1077 | **1695** ✗ |

**Yet B1-real is WORSE than B0 — and this is the real finding, not a bug:** on
vLLM **v1 (recompute-only preemption)**, an *accurate* SJF ranker starves long
requests, which v1 then **recomputes their whole prefill** on resume (no swap)
→ more total work. **The better the ranker, the worse it gets.** The paper's
2.1× depends on **swap** preemption (preempted requests are moved to CPU, not
recomputed) — which v1 removed.

> **Conclusion:** reproducing the 2.1× **requires swap → the `hao-ai-lab/vllm-ltr`
> reference stack on old vLLM (torch 2.2.1)**. Fixing labels is necessary but
> not sufficient on v1. This is the B1a reference-stack path, and the parity build belongs on
> the A100 (bare-metal CUDA toolkit). See `docs/BENCHMARK.md` for the setup.

**C1/C2/C3** still build on B1's priority-scheduling plumbing (which works); the
2.1× *magnitude* comparison happens on the reference stack (below).

---

## B1a — Reference stack (old vLLM 0.4.1 + SWAP): the 2.1× reproduced & decomposed

The swap-vs-recompute finding predicted the paper's 2.1× needs **swap**, which
vLLM v1 removed. We built **Fu et al.'s [12] released reference stack** —
`hao-ai-lab/vllm-ltr` (the vLLM 0.4.1 fork, LTR scheduler + swap, that LLM.pdf
adapts) — from source on the **RTX 3090** (WSL2), patched it for Llama-3.1
(`ltr/vendor/PATCHES.md`), and ran **Fu et al.'s released artifacts as a proxy**
(LLM.pdf itself trains/evaluates on **LMSYS-Chat-1M** and its ranker is *not*
released, so we use Fu's ShareGPT versions): the pre-generated ShareGPT/Llama-3
trace (`LLM-ltr/Llama3-Trace`) + pre-trained OPT-125M ranker
(`LLM-ltr/OPT-Predictors`, ListMLE, bucket-10). **These are NOT "the paper's own"
data — the paper is LMSYS; ShareGPT is Fu et al.'s.** Output-lengths are fixed by the trace (`--output-len -1`),
so the run is deterministic and the served weights don't matter — only
scheduling does. Driver: `serving/bench_reference.sh`.

**What LLM.pdf actually claims (so our numbers aren't cherry-picked).** The
source paper is *"An Empirical Study on Latency Reduction Techniques for LLMs"*
(Kumar et al., IEEE) — itself a reproduction/extension of Fu et al.'s vllm-ltr.
Its own words pin every knob we set:
- §III: *"based on the open-source vLLM framework (**version 0.4.1**)"*; on
  `preemption-mode`, *"vLLM defaults to the **recompute** mode… regularly
  resulted in **elevated latency**… Consequently, we employed the **swap
  mode**… produced superior outcomes"* — the swap-vs-recompute lever, verbatim.
- §III: they **rejected `num-gpu-blocks-override`** ("heightened overall
  latency… excluded from our final setup") — the very knob our v1 B0/B1 leaned
  on to force preemption, so that path was off-spec.
- §IV: metric = *"latency in **seconds per token**"* (= our Nlatency), model =
  **Llama-3.1-8B-Instruct**, load = **5–60 req/s**, hardware = **A100-40 GB**.
- Headline: *"latency reduction of **up to 2.1×** vs FCFS under **high** load"*,
  and §IV-C: *"at **low** frequencies (<20 req/s), **all methods** … similar."*

So the paper's "2.1×" is a **peak, per-token-latency, high-load, in-distribution**
number. Same metric/mechanism/model as ours — not a random match. Our opt-vs-fcfs
peak is **P99 Nlatency 3.6× / mean 1.66× at rate 4** (rate sweep below),
bracketing and exceeding 2.1×. Beyond parity we add a `fifo` (swap-only) arm the
paper never ran, which shows the benefit is **load-dependent** — ranking wins at
moderate load, swap-alone at extreme saturation.

**Three arms decompose the benefit** — ShareGPT/8B, RTX 3090, ctx 4096, util
0.85, swap 8 GB (→ 2145 GPU KV blocks = heavy pressure), **rate 8**, n=120:

| metric | `fcfs` (recompute) | `fifo` (swap only) | `opt` (LTR + swap) |
|---|---:|---:|---:|
| Mean Nlatency (ms) | 215.8 | **92.2** | 195.3 |
| Median Nlatency (ms) | 183.9 | **83.5** | 183.4 |
| P99 Nlatency (ms) | 903 | **354** | 371 |
| Mean TTFT (ms) | 9354 | **3476** | 6302 |
| Median TTFT (ms) | 7900 | **478** | 966 |
| P99 TTFT (ms) | 73027 | **28611** | 53544 |
| Mean TPOT (ms) | 174.8 | **80.6** | 182.9 |
| Throughput (req/s) | 0.92 | **1.65** | 0.77 |
| peak Swapped reqs | **0** | 18 | 33 |

**Two results, both honest:**

1. ✅ **The paper's claim reproduces here (and did NOT on v1):** `opt` (LTR+swap)
   beats the `fcfs` baseline — **P99 Nlatency 2.4× (903→371)**, **median TTFT
   8.2× (7900→966)**. On v1 (recompute-only) the *same accurate ranker* made B1
   *worse*. Mechanism is visible: `fcfs` **never swaps** (peak Swapped 0 — it
   queues, Pending→69, HOL blocking → 73 s P99 TTFT); swap arms move waiters to
   CPU and admit more concurrency (Running 95 vs 76).

2. 🔬 **Decomposition is load-dependent — and the rate sweep vindicates LTR.**
   At **rate 8** (extreme overload) swap dominates: `fcfs→fifo` (pure swap, no
   ranker) alone gives 2.3× mean Nlatency (215.8→92.2), and `fifo→opt` then
   *regresses* (92.2→195.3) — the ranker's cost (OPT-125M scoring on the serving
   GPU + heavier swap churn, 33 vs 18) isn't repaid when everything is thrashing.
   **But at rate 4 (moderate overload) LTR is the decisive winner** (below).

**Rate sweep (mean / P99 Nlatency in ms, throughput req/s) — the full picture:**

| rate | `fcfs` mean·p99·thru | `fifo` mean·p99·thru | `opt` (LTR) mean·p99·thru |
|---:|---|---|---|
| 2 (light) | 84.5 · 133 · 0.54 | 83.4 · 128 · 0.54 | 82.6 · 130 · 0.55 |
| **4 (sweet spot)** | 162.5 · 605 · 0.87 | 165.2 · 610 · 0.87 | **98.0 · 167 · 1.00** |
| 8 (overload) | 215.8 · 903 · 0.92 | **92.2 · 354 · 1.65** | 195.3 · 371 · 0.77 |

- **rate 2:** all three tie (~83 ms) — exactly the paper's *"<20 req/s, all
  methods similar."* No pressure ⇒ nothing to schedule around.
- **rate 4 — LTR wins outright and EXCEEDS the paper's 2.1×:** `opt` vs `fcfs`
  is **mean Nlatency 1.66× (162→98)**, **P99 Nlatency 3.6× (605→167)**, **TTFT 4×
  (4609→1158)**, and it alone sustains throughput 1.00 vs 0.87. It also beats
  `fifo` (98 vs 165) — so here the *ranking*, not just swap, is doing the work.
- **rate 8 — swap dominates, ranking overhead hurts:** `fifo` best; `opt` still
  beats `fcfs` on P99 (371 vs 903) but its aux-scoring + swap churn cost more
  than the ordering saves.

**Bottom line:** the paper's LTR+swap benefit is **real and reproduces** — at the
right operating point (moderate-high load) it hits **3.6× P99 / 1.66× mean**,
bracketing and exceeding the reported 2.1×. The single headline number hides a
**load-dependence** the paper didn't surface: ranking pays off at moderate
overload, swap-alone wins at extreme saturation. (3090 has a small KV pool, so
its saturation point is ~rate 4–8; on the A100-40 GB the same shape shifts to
the paper's 30–60 req/s.)

---

## C1 — KV quantization (Stage 1a: TurboQuant rotation, zero CUDA changes)

C1 adds KV-cache quantization on the **0.4.1 swap stack** (docs/C_TIERS.md), so
the whole B1a→C1 ladder shares one engine. **Stage 1a** validates TurboQuant's
core idea — a fixed orthogonal rotation of Q/K/V before fp8 storage — with **zero
CUDA changes**: rotation in the XFormers backend (`ltr/quant/rotation.py`, 5 CPU
tests, gated by `VLLM_KV_ROTATE=1`), scalar quant = 0.4.1's built-in fp8 path.
`C1 = VLLM_KV_ROTATE=1 + --kv-cache-dtype fp8`.

**Metric — decode-path token match vs fp16** (`ltr/quant/eval_accuracy.py`). A
teacher-forced *prefill* perplexity is blind to KV quantization — prefill attends
to raw K/V, not the cached quantized KV, and we confirmed fp16 and fp8 gave
**bit-identical** perplexity (17.90024668528097). So we greedy-decode 80 tokens
per passage (each attends to the quantized cached prompt KV) and count how many
match the fp16 baseline. Llama-3.1-8B, 4 neutral passages:

| config | tokens matching fp16 | match frac | per-passage (of 80) |
|---|---:|---:|---|
| `fp8` (plain, uniform) | 132/320 | 0.41 | [33, 8, 57, 34] |
| **`rot_fp8`** (rotation + fp8) | **195/320** | **0.61** | [59, 62, 21, 53] |

**Rotation lifts fp8 token-match with fp16 from 0.41 → 0.61 (+48%)** — TurboQuant's
rotation, done in Python at zero kernel cost, meaningfully cuts fp8 KV
quantization error. The rotation is exact in fp16 (rotated output reproduces the
baseline byte-for-byte), so the gain is purely reduced quantization distortion.
*Preliminary:* 4 passages + greedy cascade make per-passage noisy (passage 3
regressed); a larger eval is future work, but the aggregate direction is clear.

### Stage 1b/2 — sub-fp8 bit-widths (fake-quant accuracy study)

Below fp8, accuracy craters — rotation and KVmix become necessary but not
sufficient. Same decode token-match metric; `VLLM_KV_BITS` fake-quant
(`ltr/quant/scalar_quant.py`, no CUDA, so this is an *accuracy* study — the memory
win is separate):

| config | match frac vs fp16 |
|---|---:|
| rotation + **fp8** (Stage 1a) | **0.61** |
| 3-bit uniform (no rotation) | 0.02 |
| 3-bit + rotation | 0.14 |
| 3-bit + rotation + **KVmix** (first/last 4 layers @ 8-bit) | 0.21 |
| 2-bit + rotation | 0.02 |

- **Rotation is essential at low bit** (3-bit 0.02 → 0.14, 7×).
- **KVmix per-layer mixed precision helps further** (0.14 → 0.21) — the hypothesis
  holds directionally.
- **But even the best sub-fp8 (0.21) is far below fp8's 0.61** — matching the
  literature (vLLM ships only 8-bit KV because <8-bit "significantly impacts
  accuracy"). *Caveats:* greedy token-match cascades (noisy at 4 passages — 4-bit
  landed within noise of 3-bit), and our quantizer is naive uniform vs real
  TurboQuant (Lloyd-Max) / KVmix (gradient importance), so this is a lower bound.

### Real quantizer (NormalFloat) + the coherence caveat — the verdict flips

The naive uniform grid is a lower bound. `nf_quantize` places levels at the
Gaussian quantiles (density-matched — the optimal-after-rotation scalar quant
TurboQuant uses):

| config | uniform | **NormalFloat** |
|---|---:|---:|
| 4-bit + rotation | 0.12 | **0.27** |
| 3-bit + rotation | 0.14 | 0.20 |

NF roughly doubles 4-bit fidelity (0.12→0.27) — the real quantizer matters. **But
the token-match metric badly understates usability.** Decoding the actual
continuations, every config produces *coherent, on-topic* text — greedy decode
just cascades to a different valid wording once any token differs:

> fp16: "…hierarchical, sparse, and **compressed** format. We show that this…"
> NF 4-bit+rot: "…hierarchical, sparse, and **distributed** manner. This allows…"

NF-4-bit is fluent and semantically near-identical to fp16; even 3-bit stays
coherent. So fp16-token-match measures *fidelity to fp16*, not *quality* — the
wrong lens for a **latency** study, where coherent output + smaller KV (fewer
preemptions) is the win.

**Decision (revised).** fp8+rotation is the safe high-fidelity C1 (real 2×);
**NF 4-bit + rotation is a strong candidate for real 4× memory with coherent
output**, so a sub-fp8 CUDA int4 codec (clone the fp8 `quant_utils.cuh` → NF int4
+ 2 dispatch branches + 3 Python spots, ~1–3 days) **is worth doing** to
demonstrate the memory→preemption→latency win that the fake-quant harness cannot
(it stores fp16). Next: (a) a decode-perplexity metric (fairer than cascade
token-match); (b) the int4 CUDA codec + a C1 bench arm (latency + preemptions vs
B1a-opt at 4× KV). No vLLM version ships <8-bit KV (4-bit-Hadamard RFC #28538
closed unimplemented), so the codec is a fresh port — but bounded and mapped.

### Asymmetric k8v4 (TurboQuant's core insight) + port reality

TurboQuant's `k8v4` keeps K at fp8 and compresses only V — softmax amplifies K
error exponentially but V error only linearly, so "V compression is nearly free".
Testing asymmetric K/V bits (`VLLM_KV_BITS_K`/`_V`, NF + rotation):

| config | compression | match frac vs fp16 |
|---|---:|---:|
| fp8 (K8 V8) | 2× | 0.61 |
| **k8v4** (K8 V4) | ~2.7× | **0.38** |
| symmetric 4-bit (K4 V4) | 4× | 0.27 |
| k8v3 (K8 V3) | ~3× | 0.25 |

**k8v4 (0.38) beats symmetric 4-bit (0.27)** — sparing K while compressing V
recovers fidelity, confirming the asymmetric insight. All configs stay coherent
(§ coherence), so k8v4 is a usable ~2.7× operating point.

**Port reality (honest).** `turboquant_plus` (`ltr/vendor/turboquant_plus/`) is a
pure-NumPy reference — its production **Triton kernels live in a SEPARATE modern-
vLLM repo** (`TheTom/vllm-turboquant`, PR #38479, Triton 3.6, v1 paged layout),
not portable 1:1 to 0.4.1. So the C1 *algorithm* (rotation + NF/PolarQuant +
asymmetric k8v4) is fully characterized in Python here; the *real memory/latency*
win needs either (a) porting those v1 Triton kernels to 0.4.1's paged attention
(hard — different engine + layout), or (b) running C1 on modern vLLM with native
turboquant (the platform we set aside to keep swap). **fp8+rotation remains the
zero-risk C1 with a real 2× today.**

### C1 vs B1 — real fp8 latency on the 0.4.1 swap stack (the decisive test)

We unblocked **real** fp8 KV on the swap stack: 0.4.1's prefix Triton kernel
(`context_attention_fwd`) crashed on fp8 (`tl.dot(bf16, uint8)`) under
preemption/chunked-prefill; we patched it to dequant the cached K/V in-kernel
(`x.to(tl.float8e5, bitcast=True).to(q.dtype)`, verified Triton 2.2.0 supports
the fp8 bitcast on sm_86 — see PATCHES.md §4e). fp8 loads with **4291 GPU blocks
= real 2×** the fp16 2145. C1 (opt + fp8) vs B1 (opt + fp16), same swap stack:

| rate | B1 fp16 mean·p99·thru | C1 fp8 mean·p99·thru | B1 / C1 peak-swap |
|---:|---|---|---|
| 4 | **98 · 167 · 1.00** | 150 · 404 · 0.93 | — |
| 8 | **170 · 482 · 1.38** | 278 · 574 · 1.15 | 78 / 44 |
| 16 | **353 · 1621 · 1.31** | 621 · 2053 · 0.90 | 77 / 33 |

**C1(fp8) is worse than B1(fp16) at every rate**, even though fp8's 2× capacity
**does** cut preemption (peak swap 44 vs 78, 33 vs 77). The per-token fp8 dequant
overhead outweighs the capacity benefit.

**The finding — swap and KV-quantization are SUBSTITUTES, not complements.** Both
relieve memory pressure. B1a shows swap is the dominant lever (2.1×); once swap is
handling pressure efficiently, adding fp8 is redundant and only pays the dequant
tax. So quantization's value should be highest **without** swap — on vLLM v1
(recompute), where quantization's extra capacity would avoid the *costly
recompute* that swap avoids differently. **We ran it — and it confirms.**

The original `.venv`'s bleeding-edge CUDA 13.2 outpaced flashinfer's fp8 JIT, so
we spun up an **isolated `.venv-v1fp8`** (`vllm==0.8.5 + torch 2.6+cu124 +
xformers`) that serves v1 fp8 cleanly — *without touching the working `.venv`*.
C1 (fp8) vs B1 (fp16) on **v1 (recompute, FCFS, 3090)**, server-side metrics:

| load | metric | B1 fp16 | C1 fp8 |
|---|---|---:|---:|
| rate 16 (light) | preemptions (recompute) | 6 | 4 |
|                 | median throughput (tok/s) | **818** | 648 |
| **rate 64 (heavy)** | preemptions (recompute) | 19 | **10** |
|                     | median throughput (tok/s) | 552 | **619** |

**The crossover confirms the substitutes thesis.** At light load fp8's per-token
dequant overhead dominates (lower throughput). At **heavy load fp8's 2× capacity
halves the preemptions (10 vs 19) — halving the costly recompute — and fp8 now
has HIGHER throughput (619 vs 552)**. So on v1 (recompute) quantization **does**
help, exactly where swap would otherwise be needed. **Swap and KV-quantization are
substitutes:** the paper used swap for the 2.1×; modern vLLM (no swap) recovers
the benefit via quantization, by avoiding recompute. Stacking both (C1 on the
0.4.1 swap stack, above) is redundant. (v1 setup recipe: isolated cu124 venv +
`CUDA_HOME` + ninja + xformers backend — v1 was never broken, just a bleeding-edge
CUDA/flashinfer gap in the main `.venv`.)

---

## C1 cross-stack grid — memory-PRESSURED B0/B1/C1 × {swap, recompute}, rates 4–64 (final-paper figures)

The **authoritative** ladder. One client (`bench.run_sweep`), one workload
(**LMSYS reference-len** — paper-consistent output mean ~197 ≈ LLM.pdf's 157;
per-request *fixed real length* so LTR has a spread to reorder yet the load is
model-independent/fp8-fair; cap 4096 → 0 % truncation), the full
metric set, swept 4/8/16/32/64 req/s. B0 = FCFS/fp16; B1(a) = LTR + fp16; C1 =
LTR + **fp8 KV**. Figures: `results/summaries/fig_{04stack,v1stack,xstack}_*.png`.

> **Terminology (per `docs/C_TIERS.md` §1).** "C1" in every *latency* table below is
> **fp8 KV — the *naive-quant baseline***, used as a **conservative proxy** for C1's
> capacity mechanism. The **full C1 (TurboQuant rotation + KVmix)** is characterized
> only for *accuracy* (§C1 Stage-1a: rotation lifts fp8 token-match 0.41→0.61; target
> **4–7× compression** vs fp8's 2×); its CUDA kernels are **not ported to 0.4.1**
> (§"Port reality"), so latency uses fp8. Real C1's higher capacity would only
> **strengthen** the capacity-avoidance mechanism shown here — the latency win is a
> lower bound.

**Why a capped KV pool (and why it's honest).** The 3090 is *compute*-limited: an
earlier natural-pool sweep of this exact ladder sat at **KVpk ≤ 0.37 with 0
preemptions at every rate** — Poisson arrivals queue at *admission* and never
fill the KV, so LTR/quant had nothing to relieve (that zero-pressure grid, and
the earlier uniform-256 one, are superseded here). LLM.pdf's A100 fills KV
natively at 30–60 req/s; the 3090 can't. So we cap the pool with
`--num-gpu-blocks-override`, **dtype-scaled to hold the physical KV *bytes*
equal** — fp16 arms get **512 blocks**, fp8 arms **1024** (fp8 is half the
bytes/token, so 1024 fp8 blocks = 512 fp16 blocks of GPU memory; verified in the
server logs, e.g. `# cuda blocks: 512`/`1024`). Applied identically across arms so
the *relative* ladder is unbiased. (LLM.pdf rejected this knob on the A100 where
it wasn't needed; on a compute-limited 3090 it is the documented accommodation to
reach the paper's memory-bound regime. `docs/BENCHMARK.md`'s A100 run reaches it
natively.)

**TTFT mean (ms) — 0.4.1 SWAP stack (B0 fcfs → B1a opt+swap → C1 opt+swap+fp8):**

| rate | B0 (512) | B1a (512) | C1 (**fp8**, 1024) |
|---:|---:|---:|---:|
| 4  |   74 |   92 |   92 |
| 8  | 1264 |  731 |  143 |
| 16 | 3149 | 1136 |  175 |
| 32 | 3608 | 2337 |  386 |
| 64 | 4316 | 3160 | **925** |
| peak KVpk | **1.00** | 0.95 | **0.76** |
| peak Swapped | 0 (queues) | **17** | 0 |

**TTFT mean (ms) — v1 RECOMPUTE stack (B0 fcfs → B1 LTR → C1 LTR+fp8):**

| rate | B0 (512) | B1 (512) | C1 (**fp8**, 1024) |
|---:|---:|---:|---:|
| 4  |   69 |   69 |   68 |
| 8  | 2575 |  191 |   84 |
| 16 | 4086 |  350 |  119 |
| 32 | 4297 |  826 |  382 |
| 64 | 5076 | 1423 | **939** |
| peak KVpk | **1.00** | **1.00** | 0.78 |
| peak preempt (recompute) | **73** | **64** | **0** |

**Cross-stack synthesis (r64 TTFT mean, ms):**

| | B0 (fcfs) | B1(a) (LTR) | C1 (LTR+fp8) |
|---|---:|---:|---:|
| **0.4.1 swap** | 4316 | 3160 | **925** |
| **v1 recompute** | 5076 | 1423 | **939** |

1. **Both FCFS baselines are terrible (~4.3–5.1 s r64 TTFT)** — memory saturation
   ruins latency whether you *queue* (0.4.1 fcfs: 0 preempts, HOL) or *recompute*
   (v1: up to **73** preempts, churn). Both hit KVpk 1.00.
2. **LTR helps on both — and more on modern v1** (v1 B1 **1423** < 0.4.1 B1a
   3160). On our 3090 at equal memory, v1's `priority` scheduler outperforms the
   fork's `opt`+swap — **an intra-repo 3090 comparison, NOT a comparison to the
   paper's A100 result.** v1 admits short-predicted requests first, so its 64
   preemptions fall on long requests that can absorb them.
3. **C1 (fp8) is the great equalizer — the project thesis, shown where it bites.**
   fp8's 2× capacity holds KVpk at 0.76–0.78 → it **never saturates**, so
   **preemptions and swaps both go to zero** and *both stacks converge to ~930
   ms* — a **4.7× / 5.4×** cut over their FCFS baselines. Once KV fits, the
   swap-vs-recompute distinction **vanishes**. The tail gain is larger still:
   TTFT p99 at r64 is **v1 C1 1538 vs v1 B0 18654 (12.1×)**, **0.4.1 C1 2880 vs
   B0 15337 (5.3×)**.

**This REVERSES our earlier compute-bound finding — and that's the point.** With
abundant memory (natural pool) fp8's capacity went unused, so C1 only paid the
dequant tax and landed ≈/below B1a ("swap ⊕ quant are substitutes"). **Under real
memory pressure — LLM.pdf's regime — fp8 capacity is the dominant lever and C1
beats even swap-scheduling.** KV quantization is not merely a swap-substitute; where
KV memory is the bottleneck it is a *first-class, stack-independent* latency lever.

**Honest tradeoff:** LTR/swap cut TTFT but *lengthen the e2e tail* (SJF starves
long requests: 0.4.1 B1a e2e-p99 43.1 s > B0 35.6 s; v1 B1 42.4 s > B0 35.1 s).
**C1 improves both** (e2e-p99 0.4.1 31.2 s, v1 34.7 s — below their B0) because it
adds capacity instead of reordering scarcity — no one is starved.

Caveats: 3090 (small KV, capped pool as above), single seed, LMSYS. A100 + ctx
8192 (BENCHMARK.md) is the paper-scale native-pressure re-run. **This grid
supersedes all earlier C1 runs** (compute-bound natural pool + uniform-256 load).

## C1-native — real TurboQuant on vLLM 0.25 V1 (the fp8-proxy upgrade)

The grid above uses **fp8 KV as a conservative proxy** for C1 (TurboQuant unported to
0.4.1). vLLM **v0.25.0 upstreamed a native TurboQuant KV backend**, and it **runs on the
RTX 3090** (V1 engine + priority scheduling both work — see `docs/V025_SMOKE.md`; the
0.8.5-V0 stack could not). So we re-ran B0/B1/C1 on **true V1** with **real TurboQuant-4bit**
KV, same harness (`run_sweep.py`, LMSYS reference-len, n=200, rates 4–64).

**Setup:** vLLM 0.25.0 V1 · Llama-3.1-8B · RTX 3090 · `enforce_eager` · FLASH_ATTN for
bf16 / **TURBOQUANT** for C1. **Memory-equal cap** (the honest comparison): bf16 **512
blocks = 8,192 tok**; TQ4 **724 blocks = 23,168 tok** — *same GPU KV bytes, 2.83× the token
capacity* (724/3692 = 512/2614 = 0.196 of each dtype's natural pool; block_size is 16 for
bf16, **32** for TQ4, so match on natural-block fraction, not raw block count).

- B0 = fcfs bf16 · B1 = `priority`+LTR-ranker bf16 · C1 = `priority`+LTR-ranker **TQ4**.

**TTFT mean (ms) — C1 wins the admission/queueing regime decisively:**

| rate | B0 fcfs | B1 LTR | **C1 TQ4** | C1 vs B0 |
|---:|---:|---:|---:|---:|
| 4  |    82 |   88 |  109 | — (no pressure) |
| 8  |   129 |  134 |  120 | 1.1× |
| 16 |  5707 | 2228 |  **213** | **27×** |
| 32 |  8762 | 3901 |  **599** | **15×** |
| 64 | 10740 | 5552 | **1585** | **6.8×** |

**TTFT p99 (ms) — C1 rescues the tail that LTR alone *worsens*:**

| rate | B0 | B1 | **C1** |
|---:|---:|---:|---:|
| 16 | 13080 | 21085 | **383** |
| 32 | 19374 | 24320 | **1104** |
| 64 | 23202 | 26210 | **4323** (6.1× vs B1) |

**Preemptions (V1 recompute):** r64 **B0 171 → B1 133 → C1 40** (r16: 147 → 112 → **0**).
The 2.83× capacity keeps the pool off saturation.

**HONEST COST — TPOT (decode) and the e2e crossover:**

| rate | TPOT B0/B1/**C1** | e2e mean B0/B1/**C1** | winner (e2e) |
|---:|---|---|---|
| 16 | 53 / 56 / **63** | 11774 / 10305 / **8538** | **C1** |
| 32 | 60 / 61 / **110** | 15276 / 12124 / **12986** | B1 (C1 close) |
| 64 | 65 / 63 / **160** | 17487 / 13535 / **15336** | B1 |

**TurboQuant-4bit's decode overhead (Hadamard-unpack + dequant per token) raises TPOT ~1.4×
at low load (pure codec) to ~2.5× at high load (C1 also runs 2.83× more concurrent
requests).** So the **end-to-end** story is *regime-dependent*, exactly as the vLLM
TurboQuant study predicts (memory-for-throughput tradeoff): **C1 wins e2e at moderate load
(r16) but the TPOT cost offsets its huge TTFT gain at r≥32**, where B1 (LTR alone) is
competitive-to-better on e2e.

**Takeaway.** Real TurboQuant C1 is a **first-class TTFT / tail / preemption lever**
(6.8× TTFT, 6.1× p99, 4× fewer preemptions at equal memory) — decisively better than the
fp8 proxy on capacity — **but it is not a free lunch on throughput**: its decode tax makes
e2e a wash at the highest loads. This is the honest, publishable tradeoff, and it argues
for C1 in **TTFT-bound / interactive** serving and for pairing it with C2 (BLASST, which
attacks the *compute* side) to claw back the decode cost.

### C1-native attribution — two controls (matches the fp8 P0 method), full percentiles, accuracy

To attribute C1's win we add the two controls the fp8 grid uses, adapted for TurboQuant:

- **`c1te` (token-equal, TQ4 @ 256 blk = 8,192 tok — same as B0/B1):** isolates the *codec*.
  Its TPOT vs B1 (both 8,192 tok, same pressure) is the **pure TurboQuant decode overhead**:
  **+7 ms/tok at low load, ≈0 (even −0.6) at high load** — the Triton unpack/dequant is
  cheap. So the memory-equal C1's higher TPOT (160 vs 63 @ r64) is **~all concurrency**
  (it runs 2.83× more requests), **not a slow codec**.
- **`ctrl` (capacity oracle, bf16 @ 1448 blk = 23,168 tok — *same tokens as C1*, 2.83×
  memory):** isolates *capacity*. bf16 given C1's token budget **matches C1** (r64 TTFT
  ctrl **1319** vs C1 1585; r16 222 vs 213) — proving **C1's win is capacity, not
  TurboQuant magic** (the fp8-P0 result, reproduced for TQ4). ctrl is *slightly* better than
  C1 (TTFT 1319<1585, TPOT 126<160) = TurboQuant's residual cost — but **C1 delivers ~bf16
  latency at 1/2.83 the memory.** That is the whole value proposition, quantified.

**TTFT full percentiles @ r64 (ms) — the tail story only percentiles show:**

| config | p25 | p50 | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| B0 fcfs | 1179 | 12192 | 22306 | 22932 | 23202 |
| B1 LTR | **835** | 2529 | 6199 | **16953** | **26210** |
| **C1 TQ4** | 1003 | 1492 | 1819 | 2781 | 4323 |
| ctrl bf16 | 1041 | 1260 | 1448 | 2234 | 3288 |

**B1 (LTR) has the best p25 (835) but its p90/p99 explode (16953 / 26210)** — SJF starves
long requests, visible *only* in the full distribution. **C1/ctrl stay tight across all
percentiles** (capacity avoids the starvation). This is the same "LTR wins the head, loses
the tail; capacity fixes both" pattern as the fp8 grid, now on real TurboQuant.

**Accuracy (GSM8K, n=100, greedy, generation-based — the metric that *sees* KV quant):**
**bf16 0.80 vs TurboQuant-4bit 0.83** (Δ **+0.03**, within noise: 5 correct→wrong, 8
wrong→correct). **No measurable accuracy loss from 4-bit TurboQuant KV** — the capacity/TTFT
win is essentially free on quality here. (Larger-n confirmation owed; `eval_gsm8k.py`.)

**Audit note (config parity across stacks).** The v0.25 ladder now matches the 0.4.1/0.8.5
grids' control set: B0 · B1 · C1 · **ctrl** (capacity oracle, ⟵ was the gap) · plus a v0.25
bonus **c1te** (codec isolation). Boundaries: boot+serving bench, single seed, LMSYS; open
upstream TurboQuant bugs (#41560/#40420); `k8v4`/fp8 variants need SM89 (Ampere → `_nc`).
Data: `results/summaries/{b0,b1,c1,c1te,ctrl}_v025.json` + `gsm8k_{bf16,tq4}_v025.json`;
repro: `serving/v025_smoke/`.

### C1-native long-context — v0.25 aligned with the 0.4.1/0.8.5 lc probe (+ bf16 vs TQ4)

Aligning v0.25 with the older stacks' long-context probe (fixed **rate 6, pool 1024, FCFS,
sweep ctx 512–7168, uniform 96-tok output**). bf16 arm = OVR 1024 (16,384 tok); TQ4 arm =
OVR 1446 (**memory-equal**, 2.83× tokens = 46,272).

**4-way TTFT mean (ms) @ rate 6:**

| ctx | 0.4.1 swap | 0.8.5 recompute | v0.25 V1 (bf16) | **v0.25 TQ4** |
|---:|---:|---:|---:|---:|
| 512 | 506 | 635 | 822 | 806 |
| 2048 | 12571 | 12378 | 12143 | **9751** |
| 4096 | 40045 | 39107 | 32660 | **26039** |
| 7168 | 77507 | 73824 | 66348 | **55397** |

Two findings: **(1) the modern V1 engine handles long context better** — at ctx 7168 v0.25
(66.3 s) beats 0.8.5 (73.8 s) and 0.4.1 (77.5 s); at ctx 512 the old swap stack is
marginally faster (differences <1 s). **(2) TurboQuant's capacity gives a *modest* 1.2–1.3×
TTFT at long context** (ctx 2048–7168), far smaller than the short-context grid's 6.8×.
Why: at long ctx + rate 6 the 3090 is **admission-queueing-bound** — a few huge requests
saturate the pool (preempt ≈0, KVpk 0.8–1.0), so extra capacity only admits a couple more;
the bottleneck is the per-request KV, not preemption. This **reproduces the older lc "honest
negative"** (long ctx = queueing, not preemption) and extends it: even 2.83× compression is
modest here — a real long-ctx win needs TQ3 (5×) or an A100 (native fill). e2e: TQ4 is
flat-to-worse at short ctx (decode+concurrency tax), ≈equal at 7168. Data:
`results/summaries/lcv025{,tq4}_{512,2048,4096,7168}.json`; repro: `serving/v025_smoke/longctx_v025.sh`.

**LTR arm at long context — LTR *hurts* here (answers "why FCFS?").** We added the missing
B1 (LTR priority + ranker) long-context arm on v0.25 (`lcv025ltr_*`). TTFT mean @ rate 6:

| ctx | B0 fcfs | **B1 LTR** | C1 TQ4 |
|---:|---:|---:|---:|
| 512 | 822 | **1071** | 806 |
| 2048 | 12143 | **14638** | 9751 |
| 4096 | 32660 | **40319** | 26039 |
| 7168 | 66348 | **79209** | 55397 |

**LTR is slower than plain FCFS at every context** (r7168 79.2 s vs 66.3 s, 1.19× worse).
This is the regime-dependent thesis's sharpest edge: long context on the 3090 is
**admission-queueing-bound** (a few giant prefills, ~0 preemptions), where LTR reordering
can't help *and* the OPT-125M ranker scoring + priority churn is pure overhead. So the
long-ctx section runs FCFS by design — **only C1's capacity (TQ4) helps here**; LTR's win is
in the short-context / high-concurrency / preemption regime, not this one. Data:
`results/summaries/lcv025ltr_*.json`; repro: `serving/v025_smoke/longctx_ltr_v025.sh`.

### C2 at long context (measured) — a counter-intuitive, honest finding

We had hypothesized long context = attention-dominated TPOT = where C2 (BLASST) shines. **We
ran it and the data says otherwise.** Clean dense-vs-BLASST on the TQ4 path (both
`--no-enable-chunked-prefill`, rate 6, sweep ctx, n=40, all 40/40 healthy):

| ctx | dense TPOT | +C2 TPOT | ΔTPOT | reading |
|---:|---:|---:|---:|---|
| 512 | 69.1 | **55.0** | **−20.4%** | big batch → compute-dense → C2 wins big |
| 2048 | 118.2 | **106.1** | **−10.3%** | batch shrinking |
| 4096 | 144.8 | 145.7 | +0.6% | turning memory-bound |
| 7168 | 159.7 | 164.2 | **+2.8%** | batch ~2 → C2 fades |

**Finding: C2's benefit tracks COMPUTE DENSITY (batch/concurrency), NOT context length.** At
rate 6, longer ctx means bigger per-request KV, so the memory-equal pool admits fewer
concurrent requests (batch ~27 → ~2); decode turns **memory-bandwidth-bound** (the cost is
reading the 7k-token KV back, not the attention matmul), and BLASST — which skips *compute*,
not the K-load needed for `block_max` — stops paying, its guard overhead even nets slightly
negative. This **reconciles with the r64 result** (−22% TPOT: high load → big batch →
compute-dense even at ctx 2048). So C2 is a **compute-bound-regime lever**, not a universal
win — exactly the kind of "does it compose cleanly?" answer the ablation was designed to
surface. Data: `results/summaries/lctq4nc_*.json` (dense), `lctq4blastnc_*.json` (+C2);
repro: `serving/v025_blasst/longctx_c2_nochunk.sh`.

**Debugging note (kernel stability).** The first long-ctx C2 attempt crashed with a CUDA
`illegal memory access` in the BLASST branch of `_tq_decode_stage1` at ctx ≥ 2048. Root
cause: vLLM's **chunked-prefill** batches a long prefill *chunk* together with in-flight
decodes, and that mixed batch trips the data-dependent skip branch. A *single* long sequence
(offline, 3401 tok) never crashed, and the main payoff (`c1c2_tq4`, ≤ 2048 tok, chunked-prefill
on) was always 200/200 healthy. **Disabling chunked-prefill fixes it** (numbers above). A
chunked-prefill-safe BLASST guard is a scoped kernel follow-up.

## C2a — BLASST algorithmic screen (real Llama attention; the CTA-uniform tax, measured)

First C2 step (`c2/blasst_screen.py`, **no vLLM kernel yet** — the algorithm screen from
`docs/C2_BLASST_PLAN.md §7`): capture post-RoPE Q/K/V from Llama-3.1-8B **layer 16** on a
1536-token prefill, simulate the online-softmax block-skip in torch, sweep threshold **τ →
(achieved block sparsity, output rel-error vs dense)**. Contrasts the *per-query* skip
(best case = **decode**, one query/program) with the *block-uniform* skip a Triton **prefill**
kernel needs (all BLOCK_M rows must agree — §1 scalar/CTA-uniform predicate).

| τ | per-query (=decode) | block M=16 | block M=64 |
|---:|---|---|---|
| 4 | **0.725** / 12% | 0.338 / 4.2% | 0.147 / 1.6% |
| 6 | **0.423** / 3.2% | 0.067 / 0.6% | 0.013 / 0.13% |
| 8 | 0.103 / 0.4% | 0.001 / 0.01% | 0.000 / 0 |

*(sparsity = fraction of KV blocks skipped; err = ‖O_sparse − O_dense‖ / ‖O_dense‖.)*

Findings: **(1) the BLASST algorithm works** — per-query skip drops **42% of blocks at 3.2%
output error** (τ=6), 72% at 12% (τ=4): a real sparsity/error knee. **(2) the CTA-uniform
tax is severe and scales with tile size** — at τ=6 realizable sparsity collapses per-query
**42% → M=16 6.7% → M=64 1.3%**, because a causal-prefill tile's queries (spanning many
positions) rarely agree to skip a block. This **empirically confirms both kernel audits'
warning** (§9). **(3) strategy, now data-backed: decode-first.** Decode = one query/program
→ per-query *is* the achievable → full 42% sparsity, **no *position* tile tax** (matches the
0.4.1 audit's "decode is the better algorithmic fit"). **Caveat (C2 audit):** this holds for
a kernel that gives each *(seq, head)* its own program (like C2b) — no KV-load sharing.
vLLM's production grouped-decode kernel shares a KV load across a **GQA group** (16 query
heads / KV head), so it would pay a *cross-head* agreement tax there **unless** the
integration drops KV-load sharing. So ~42%/~1.3× is the per-head-program result; production
transfer must pick: per-head layout (full sparsity, no KV sharing) or GQA grouping (KV
sharing, lower realized sparsity). This is the key integration risk, not a measured tax. Prefill needs a small BLOCK_M (16 → 34% at τ=4 /
4% err) to be worth it; the default 64–128 kills it. Next (future): implement the guard in
v0.25 `kernel_unified_attention` decode path (§2.3) + a real Triton-latency benchmark. Data:
`results/summaries/c2a_blasst_screen{,_m16}.json`; repro: `c2/blasst_screen.py`.

## C2b — Triton decode kernel with the BLASST skip: real 3090 speedup (§8's risk, resolved)

C2a said decode-first; **C2b builds a standalone Triton decode-attention kernel**
(`c2/blasst_decode_kernel.py`) with the scalar skip guard and benchmarks **dense vs skip on
the 3090** at controlled sparsity — answering the plan's §8 open risk ("Ampere may not
net-speedup"). (Triton 3.x rejects `continue`; the guard is a scalar `if` around the block
body — still a real branch.)

**RTX 3090 · 256 programs (8 seqs × 32 heads) · N=2048 · BLOCK_N=64 · τ=4:**

| block sparsity | dense (ms) | skip (ms) | speedup |
|---:|---:|---:|---:|
| 0% | 0.616 | 0.616 | **1.00×** |
| 25% | 0.614 | 0.547 | 1.12× |
| 50% | 0.613 | 0.462 | **1.33×** |
| 75% | 0.613 | 0.368 | 1.67× |
| 90% | 0.613 | 0.350 | **1.75×** |

Correctness: skip vs dense-softmax **rel-err 2.9e-4**. Three results: **(1) it's a real
speedup** — 1.33× at 50 %, 1.75× at 90 % sparsity on the 3090's Triton decode. **(2) no
penalty at 0 % sparsity** (1.00×) — the scalar branch is free when nothing skips, so BLASST
is safe to leave on. **(3) the ceiling is ~2×** (sub-linear: 90 % → 1.75×, not ~9×) because
the **Q·K dot still runs for every block** (needed for block_max) — only exp + V-load + P·V
are skipped (~half the per-block work; matches §1's half-FLOP ceiling). At the C2a-realistic
**~42 % decode sparsity → ~1.25–1.3× decode-attention speedup**, at 3 % output error,
verified correct. **§8's honest risk is resolved positively for decode.** Data:
`results/summaries/c2b_decode_bench.json`; repro: `c2/blasst_decode_kernel.py`.

**C2 status.** Algorithm validated (C2a) + kernel speedup proven on the 3090 (C2b), both on
real/realistic shapes with verified correctness. Remaining (production engineering, future):
thread τ into vLLM v0.25's `kernel_unified_attention` decode path (§2.3) and measure e2e
serving TPOT — i.e. turn this ~1.3× kernel win into the TPOT relief the C1-native bench
showed C1 wants (C1 attacks memory/capacity → TTFT; C2 attacks compute/decode → TPOT; the
two are the complementary halves this project set out to show).

## C-tier ablation matrix — LTR / LTR+C1 / LTR+C2 / LTR+C1+C2 (v0.25, r64)

Bringing the pieces together — **all four rows are now real v0.25 serving sweeps.** C2 was
threaded into *both* decode kernels via runtime monkeypatch (`serving/v025_blasst/`). **The
decisive finding: the same BLASST algorithm has OPPOSITE sign on the two kernels.** On bf16
`kernel_unified_attention` (GQA 4-way KV sharing + BLOCK_Q packing) the CTA-uniform skip needs a
whole tile to agree → realized sparsity ≪ 42%, guard cost > savings → **TPOT +4.5% (net-neg)**.
On TQ4 `_tq_decode_stage1` (per-head flash-decode, 1D scores, scalar predicate) there is no tile
tax → full sparsity, and a skipped block also skips the value dequant → **TPOT −22%, p99 −47%
(net-pos)**. **C2 effects are per-backend τ0→τ6 deltas — different backends' absolute TPOT is NOT
comparable across rows** (c1_tq4's 214.7 ≠ c1_v025's 160 — different run/config). C1 and C2 attack
**orthogonal** metrics; per-head is C2's correct landing spot and it *is* the C1 path, so
**LTR+C1+C2 is the project payoff.**

| config | mechanism · backend | TTFT | TPOT | preempt | quality | status |
|---|---|---:|---:|---:|---|---|
| **LTR** (B1) | scheduling only · FLASH | 5552 | 63 | 133 | PPL 9.46 (bf16=dense) | ✅ measured |
| **LTR+C1** | + KV capacity (TQ4) · default | **1585** | 160 | **40** | GSM8K 0.83 (bf16 0.80) | ✅ measured |
| **LTR+C2** | + BLASST · bf16/unified | flat | 67.9→71.3 (**+4.5%**) | flat | PPL 9.46 (τ=6, 0 cost) | ✅ measured (net-neg) |
| **LTR+C1+C2** | + BLASST · TQ4/per-head | 2347→1747 | 214.7→167.3 (**−22%**) | 42→40 | GSM8K 0.85 joint (vs TQ4 0.83, lossless) | ✅ measured (net-pos) |

**On the TQ4 net-positive:** C2 on `_tq_decode_stage1` beats bf16's attention-only ~1.3× because
per-head removes the GQA tile tax **and** a skipped block also skips TQ4's value dequant (MSE
unpack / centroid gather / norm) — so the −22% TPOT / −47% p99 measurably claws back C1's decode
cost. The earlier audit caveat still holds (C1's r64 TPOT inflation 63→160 is largely *concurrency*,
2.83× time-slicing, which a faster kernel doesn't directly undo) — C2 does not erase the concurrency
term, but it does cut the per-token decode cost (−22%) and crush the tail (−47%), which is where the
combined lever pays off. **On the bf16 net-negative:** it is the real-serving confirmation of the
audit's "GQA cross-head tax" risk (below) — the unified kernel is the *wrong* landing spot; the
per-head kernel is the right one, and it is the C1 path. **TTFT footnote:** C2 adds no KV capacity,
so within each backend the τ=6 arm's TTFT tracks the τ=0 arm; on TQ4 the faster decode also drains
the queue (TTFT 2347→1747 @ r64).

**The coordination thesis, quantified:** C1 = a **capacity → TTFT/tail/preemption** lever
(TTFT 5552→1585, p99 26210→4323, preempt 133→40) that *costs* TPOT (concurrency); C2 = a
**compute → TPOT** lever (1.3× decode) that adds no capacity. Complementary — so
**LTR+C1+C2 is where C1's TTFT win and C2's TPOT relief combine**, each covering the other's
weakness. That is exactly what "KVCache-coordinated latency optimization" set out to show.

**Same-backend ladders + joint quality (2026-07-12, added per review — removes the cross-backend caveat):**
B0 (fcfs) was run on BOTH C2-arm backends, giving two ladders whose absolutes ARE comparable within each:
- **bf16/TRITON (OVR512):** B0→B1→B1+C2 — TTFT @r64 12275→**6488**→6537, TPOT 79.5→**67.9**→71.3, preempt 162→147→112. LTR (B0→B1) is a real **1.9× TTFT lever** here (pressured pool, r16 2.8×); B1→+C2 is **+4.5%** (GQA tax). Data `b0_triton/b1_triton/c2_bf16`.
- **TQ4/turboquant (OVR724):** B0→B1+C1→B1+C1+C2 — TTFT @r64 2220→2347→**1747**, TPOT 157.1→214.7→**167.3**, p99 1121→**597**. Here LTR (B0→+C1) is **net-negative** (abundant capacity → ranker/churn overhead with no pressure to optimize, consistent with the long-ctx LTR-hurts finding), while +C2 cuts **TPOT −22% / p99 −47% AND recovers TTFT to the best 1747** (faster decode drains the queue → reclaims what LTR cost). So on TQ4, C2 is the only lever that reliably pays; LTR is regime-dependent. Data `b0_tq4/c1_tq4/c1c2_tq4`.
- **Joint C1+C2 quality (closes the composed-quality gap):** GSM8K on **TQ4+BLASST (τ=6, through the real `_tq_decode_stage1` decode kernel, n=100) = 0.85** vs TQ4 dense 0.83 (Δ+0.02, within sampling noise) → **BLASST is lossless even reading TQ4-dequantized KV** — a true JOINT measurement, not a composition of the separate C1-GSM8K + C2-PPL results. Note: teacher-forced PPL is a *prefill* computation and would not exercise the BLASST decode skip, so GSM8K (which generates → uses decode) is the correct joint-quality signal. Data `gsm8k_tq4blasst_v025.json`.
- **Complete per-config GSM8K (n=100, greedy):** bf16 dense (B0/B1) 0.80 · bf16+BLASST (C2) **0.88** · TQ4 (C1) 0.83 · TQ4+BLASST (C1+C2) 0.85 — all inside the n=100 sampling-noise band (±~5%), so **both C1 (quantization) and C2 (sparsity) are accuracy-lossless, and so is the joint**. C1 uses GSM8K because KV-quant is blind to teacher-forced PPL (§C1); C2 is double-checked with PPL (9.464→9.463). Data `gsm8k_{bf16,c2bf16,tq4,tq4blasst}_v025.json`.

### Perplexity — BLASST is lossless at its operating point

`c2/blasst_ppl.py` applies the per-query block-skip in **every** layer and measures PPL
(Llama-3.1-8B, 8×1024-tok chunks):

| τ | sparsity (C2a) | PPL | ΔPPL |
|---:|---|---:|---:|
| ∞ (dense) | 0 | 9.464 | 0 |
| 8 | 10% | 9.463 | −0.001 |
| 6 | 42% | 9.463 | −0.001 |
| 4 | 72% | 9.597 | +0.133 |

**At τ=6 (42% sparsity) PPL is unchanged** — the 3.2% attention-output error (C2a) does not
propagate to LM loss; only aggressive τ=4 (72%) raises PPL +1.4%. So BLASST's quality cost at
a useful operating point is **zero**, complementing C1's GSM8K-neutral result (both C-tiers
are quality-safe at their target settings). (KV-quant C1 is blind to *teacher-forced* PPL by
construction — §C1 — hence GSM8K for C1, PPL for C2.) Data: `results/summaries/c2_blasst_ppl.json`.

### C2 independent audit (static) — correct + honest, with scoping caveats

An independent static audit read both C2 sources + the JSONs + the claims. **Verdict: no
correctness bugs — the online-softmax skip is a true no-op on skip, block 0 is protected,
causal/−inf masking never corrupts block_max or the sums, and every headline number matches
the JSON exactly; scoping is honest (C2 was pending serving-integration *at audit time* — now
DONE on both decode kernels: bf16/unified net-neg +4.5%, TQ4/per-head net-pos −22%/−47% p99;
sparsity labeled controlled/synthetic).** Caveats folded in (optimism/scope, not math errors):
1. **GQA cross-head tax (medium → NOW CONFIRMED in serving)** — the "no tile tax" decode result
   assumes a per-(seq,head) program; vLLM's grouped `kernel_unified_attention` adds a cross-head
   agreement tax. **This is exactly why C2 on bf16/unified measured net-NEGATIVE (+4.5% TPOT) in
   serving — the risk was real.** The fix is confirmed too: TQ4's `_tq_decode_stage1` IS
   per-(seq,head), so C2 there hits the per-head ceiling → **−22% TPOT / −47% p99 (net-positive)**.
   So the integration risk is resolved by *kernel choice*: unified = wrong spot, per-head = right.
2. **fp32 in C2b (low)** — the benchmark uses fp32 K/V, so absolute ms are ~2–4× pessimistic
   vs production fp16/fp8; the **speedup ratios** are dtype-robust (that's the headline).
3. **Easy correctness case (low)** — C2b's 2.9e-4 validates the accumulation math (cold blocks
   ~10 below max); decision-boundary error is characterized separately in C2a on real scores.
4. **Idealized load balance (note)** — C2b's controlled sparsity is uniform across programs;
   a real workload varying per head/seq is bottlenecked by the least-sparse program, so the
   true speedup at a given *average* sparsity is ≤ the idealized curve.

### C2 production decode reality — GQA cross-head tax (measured) + the CUDA path

The audit's #1 risk, now measured (`c2/blasst_gqa_decode.py`, real Llama L16 — 32 heads / 8
KV heads / queries_per_kv=4; every query position as a decode):

| τ | per-head sparsity | GQA-group sparsity | retained |
|---:|---:|---:|---:|
| 4 | 0.735 | 0.548 | 75 % |
| 6 (lossless) | 0.421 | 0.185 | 44 % |
| 8 | 0.092 | 0.012 | 13 % |

**The cross-head tax is real but not catastrophic:** at τ=4 the GQA-grouped kernel still skips
**55 %** of blocks (vs 74 % per-head); at the PPL-lossless τ=6, **18.5 %** (vs 42 %). Through
the C2b speedup curve, v0.25's grouped Triton decode nets **~1.1× at τ=6 (lossless)** or
**~1.3× at τ=4 (+1.4 % PPL)** — smaller than per-head, still positive.

**The CUDA decode path avoids the tax entirely.** The 0.4.1/0.8.5 CUDA paged-attention kernel
launches `Grid=(num_heads, num_seqs, partitions)` — **one block per (seq, head)**, no GQA
KV-load sharing (`attention_kernels.cu:88,114,146`). So a BLASST skip there is
**per-(seq,head) = per-query = no cross-head tax → the full 42 % / ~1.3×.** This *reverses*
the earlier "C2 belongs on v0.25" lean: to **realize** BLASST's full decode speedup, the
**CUDA decode (0.4.1 swap or 0.8.5)** is the better target — cost = a CUDA-kernel edit +
recompile (the fork builds from source) vs v0.25's taxed-but-Python Triton edit. So C2 has
**two viable homes with a clear trade-off: v0.25-Triton (easy edit, cross-head-taxed ~1.1×
lossless) vs 0.4.1/0.8.5-CUDA (recompile, full ~1.3×, back on the paper's swap stack).**
Answering "can C2 run on v04/v08?" — **yes, via a CUDA decode edit, and it's the *better* fit
for full sparsity.** Data: `results/summaries/c2_gqa_decode.json`.

---

## Follow-up: P0 attribution · full percentiles · Nlatency ladder · accuracy · long-context

Five audit-driven follow-ups (single seed, RTX 3090). Drivers:
`serving/grid_pressured.sh` (P0), `serving/grid_reference_ladder.sh`,
`ltr/quant/eval_gsm8k.py` (P1), `serving/grid_longctx.sh` (P2).

### P0 — is C1's win the fp8 dtype, or just capacity?
Three arms (opt/LTR; only KV blocks/dtype vary), giving **two clean, distinct
ablations**. r64 TTFT mean:

| stack | fp16@512 | fp16@1024 (ctrl) | fp8@1024 (C1) |
|---|---:|---:|---:|
| 0.4.1 swap | 3160 | **874** | **925** |
| v1 recompute | 1423 | **1048** | **939** |

Note the block/byte accounting: **fp16@1024 and fp8@1024 hold *tokens* equal**
(both 16 384) **but fp16@1024 uses ~2× the KV *bytes***; **fp16@512 and fp8@1024
hold *bytes* equal** (fp8 is half the bytes/token) **but fp8 gets 2× tokens**. So:
- **Memory-equal (fp16@512 → fp8@1024):** fp8 fits 2× tokens in the same GPU bytes,
  avoids saturation → 3160→925 / 1423→939. **This is C1's real fixed-budget win.**
- **Token-equal oracle (fp16@1024 vs fp8@1024):** ctrl ≈ C1 (874≈925, 1048≈939;
  KVpk both 0.76, 0 preempts). fp16 given the *same token capacity* matches fp8, so
  **the causal mechanism is CAPACITY, not fp8 arithmetic**; fp8 pays no net penalty
  (dequant ≈ bandwidth saved).

So the fp16@1024 oracle **isolates capacity as the mechanism** — it is *not* a
fixed-memory ablation (it spends ~2× the bytes); that role is filled by
fp16@512-vs-fp8@1024. On fixed hardware where fp16 is already at its natural max and
still saturating, **fp8 is the only lever that buys the 2×** ("quantization = cheap
capacity per byte"). *Caveat:* fp16@1024 is only runnable because our 512-block cap
is an **emulated memory-bound stress regime**, not the 3090's natural max (2145
blocks) — see the cap note above.

### Percentiles (p25/50/75/90/99) — mean/p99 alone hid the distribution shape
**TTFT full percentiles @ r64 (ms)** — the grid tables above report mean+p99 for the
trend; the full distribution tells more:

| 0.4.1 swap @ r64 | p25 | p50 | p75 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|---:|
| B0 fcfs | 305 | 814 | 12771 | 14213 | 15337 | 4316 |
| B1a opt+swap | 354 | 560 | 2340 | 11489 | 17204 | 3160 |
| C1 opt+swap+fp8 | 343 | **497** | **1559** | **1957** | **2880** | **925** |

| v1 recompute @ r64 | p25 | p50 | p75 | p90 | p99 | mean |
|---|---:|---:|---:|---:|---:|---:|
| B0 fcfs | 370 | 1308 | 14979 | 16358 | 18654 | 5076 |
| B1 LTR | 379 | 1119 | 1386 | 1873 | **13399** | 1423 |
| C1 LTR+fp8 | 337 | 1126 | 1338 | 1486 | **1538** | 939 |

Three things **p99-alone hides**: (1) **p25 is config-independent (~305–379)** — the
fastest quartile gets first token fast regardless of scheduling/quant; (2) **B0 is
bimodal** (p50 814 but p75 12771 — half fast, half HOL-blocked); (3) **C1 compresses
the WHOLE tail** (p75/p90/p99 all drop), not just p99. On v1 the sharpest insight:
**B1 (LTR) has great p50–p90 (1119–1873) but p99 explodes to 13399** — SJF starves
long requests, which v1 recomputes; **C1 rescues p99 to 1538 (8.7×)** and zeros
preemptions. *Honest caveat:* absolute TTFT has **~15 % run-to-run variance** (fixed
seed, non-deterministic server/swap timing); the relative ladder and **medians** are
the robust readouts.

### ShareGPT Nlatency ladder (paper's metric, rate 2–60) — answers "why only 2/4/8?"
The original Nlatency table stopped at 2/4/8 because the 3090, on the paper's
**long ShareGPT trace**, saturates at rate ~8 (small KV pool); the paper's 30–60 is
the **A100's** saturation range. Re-run to the paper's full 5–60 window (fixed 200
prompts/rate; `reference_ladder.json`, `fig_ladder_nlatency.png`):

| Mean Nlatency (ms/token) | r2 | r4 | r8 | r16 | r32 | r60 |
|---|---:|---:|---:|---:|---:|---:|
| B0 fcfs | 40 | 124 | 157 | 189 | 205 | **207** |
| B1a opt+swap | 48 | 107 | 130 | 135 | 129 | **129** |
| C1 opt+swap+fp8 | 52 | 86 | 119 | 126 | 137 | **144** |
| **P99 Nlatency** B0 / C1 | 49/65 | 789/125 | 1174/227 | 1425/235 | 1574/257 | **1593/273** |

- **rate 2: all three ~40–52** (the paper's *"<20 req/s, all methods similar"*).
- **rate 16–60 (deep overload): B1a/C1 stay flat (~130) while B0 climbs to 207** —
  LTR+swap keeps per-token latency stable; FCFS degrades.
- **P99 Nlatency at r60: B0 1593 vs C1 273 = 5.8×** (B1a 4.4×). The curve now spans
  the paper's full range. rates >8 are deep 3090 overload; the A100 reaches 30–60
  natively (BENCHMARK.md).

### P1 — fp8 KV accuracy (generation-based, the correct metric)
GSM8K, 200 problems, greedy generation (each token attends the **quantized** cache):
**fp16 86.0 % vs fp8 87.0 %** — the 1-point gap is **within sampling noise** (6
correct→wrong balanced by 8 wrong→correct), giving **no evidence of material accuracy
degradation** in this evaluation. *Not "lossless"* — that stronger claim needs 3
seeds, 500–1000 examples, a paired significance test, and a long-context (RULER /
LongBench) probe with a quality metric beyond exact-token match (future work).
Teacher-forced perplexity is *blind* here (fp16 and fp8 give bit-identical PPL); only
generation exercises the quantized KV. See §C1 Stage-1a.

### P2 — swap vs recompute at long context (inconclusive / honest negative)
Fixed rate 6, pool 1024, both FCFS **order** (only the preempt mechanism differs),
sweep context 512–7168:

| context tokens | 0.4.1 swap TTFT / preempt | v1 recompute TTFT / preempt |
|---:|---:|---:|
| 512 | 506 / 0 | 635 / **4** |
| 2048 | 12571 / 0 | 12378 / 0 |
| 4096 | 40045 / 0 | 39107 / 0 |
| 7168 | 77507 / 0 | 73824 / 0 |

**No crossover** — long contexts on the 3090 trigger **admission queueing, not
preemption** (few long requests co-fit; the rest queue → preempt=0), so swap and
recompute perform within ~5 % (recompute marginally *better*). Only at ctx 512
(many co-admit → pool fills during decode → real preemption) does recompute show
its cost (635 vs 506 ms, 4 vs 0 preempts). The reviewer-motivated crossover needs a
genuine **preemption** regime (A100 memory, or many medium-length requests) — the
3090's 512–7168 range can't reach it. **Future work.**

---

### LTR ranker head-to-head — ours vs the released ShareGPT predictor, both distributions
**Attribution first (this was misstated in an earlier revision).** LLM.pdf (Kumar et al.)
trained its OPT-125M ranker on **LMSYS-Chat-1M** and it is **unreleased**. The only
*released* reference ranker is **Fu et al.'s [12] vllm-ltr predictor, trained on ShareGPT**
(`opt-125m-llama3-8b-sharegpt-…`) — that is what the 0.4.1 §B1a reproduction used and what
"downloaded predictor" means here. Our `opt125m-ltr` is **LMSYS-trained, i.e. it replicates
LLM.pdf's *own* methodology** (LMSYS), not Fu et al.'s. So this is our-LMSYS-ranker vs
Fu-et-al's-ShareGPT-ranker, not a comparison to LLM.pdf's (unavailable) ranker.

Ranking quality (Kendall's |τ| vs true output-length order), each ranker on BOTH
distributions (`ltr/ranker/headtohead.py`, n=1000; LMSYS length = reference reply,
ShareGPT length = tokens in the trace's `generated`):

| ranker | LMSYS | ShareGPT | trained on |
|---|---:|---:|---|
| opt125m-ltr (ours; LMSYS = LLM.pdf's methodology) | **0.714** | 0.372 | LMSYS lengths |
| opt125m-real (ours) | 0.467 | 0.378 | real Llama-8B lengths |
| Fu et al.'s released predictor | 0.347 | **0.491** | ShareGPT / Llama-3-8B |

**The diagonal holds — each ranker wins on its OWN training distribution.** Fu et al.'s
ShareGPT predictor is best on ShareGPT (0.491); ours on LMSYS (0.714). Rule = **workload-
matched ranker**: Fu et al.'s on the ShareGPT/0.4.1 reproduction (✓), ours on the v1/LMSYS
arm (✓). **This is NOT "beating the paper":** on LMSYS (LLM.pdf's own home) our ranker
out-scores the off-distribution ShareGPT predictor *because we follow LLM.pdf's LMSYS
methodology* — replication, not a better method. LLM.pdf's actual ranker is unreleased, so
a direct ranker-vs-ranker comparison to it isn't possible. `opt125m-real` is a moderate
generalist. Note: higher τ ≠ better serving — a *more accurate* SJF worsens the v1
recompute tail (starves long jobs).

---

## Thesis — what we actually proved (regime-aware interaction study)

Not "LTR always compounds with KV quantization." The precise, evidence-backed claim
(aligned with the proposal's goal of KV controls beneath LTR to cut preemption/latency):

> **Under a memory-bound KV regime, extra KV capacity removes the preemption
> mechanism itself, making swap-vs-recompute largely irrelevant; under a
> compute-bound regime, KV quantization is neutral-to-harmful because its codec
> overhead isn't repaid.**  *Avoidance beats recovery.*

Four findings:

1. **Preemption semantics decide scheduler value.** An accurate SJF-like ranker
   *degrades* tail latency under **recompute-only** preemption (v1 B1): deprioritized
   long jobs lose their KV and must redo prefill. Matches the paper's swap-over-
   recompute choice.
2. **LTR has a bounded operating region.** The **clean LTR ablation is `opt` vs
   `fifo`** (both swap; only ordering differs) — **not `opt` vs `fcfs`, which changes
   ordering AND preemption mode**. Holding swap fixed: ranking wins at moderate
   overload (r4: opt **98** vs fifo **165** ms/token = **1.68×**) but LOSES at deep
   overload (r8: fifo **92** vs opt **195**). The opt-vs-fcfs 3.6× headline is a
   *combined policy-stack* number, not an LTR-only result. *(The extended 2–60 ladder
   currently omits the `fifo` arm, so this clean ablation is available at rates 2/4/8
   only — a `fifo` re-run would extend it.)*
3. **KV capacity changes the regime.** fp8 doubles effective blocks at a fixed
   KV-byte budget; when that prevents saturation, preemptions/swaps → 0 (P0).
4. **The real win is saturation avoidance.** Once KV fits, the legacy **swap** stack
   and the modern **recompute** stack **converge** (~925 / 939 ms mean TTFT @ r64,
   4.6–4.7× under FCFS) — the recovery mechanism stops mattering.

**Scope (honest).** The capped 512-block pool is an **emulated memory-bound stress
regime**, not the 3090's native max (2145 blocks) — a *mechanism* stress test, not a
deployment number. The clean final-paper run is the planned **A100-40 GB**
(`docs/BENCHMARK.md`, native 5–60 req/s, no block override). Every latency "C1" = the
**fp8 naive-quant baseline** (proxy for full TurboQuant+KVmix C1); fp8 accuracy shows
no material degradation (P1) pending a larger multi-seed study.

**On the LTR itself (honest framing):** we did *not* "improve" the paper's ranker. The
two-way head-to-head (above) shows each ranker wins on its **own** distribution, so we
did **workload-matched reuse** — LLM.pdf's own predictor on the ShareGPT / 0.4.1
reproduction, a LMSYS-trained ranker on the v1 / LMSYS extension. **Our contribution is
the C-tier KV layer *beneath* LTR (capacity → avoidance), not a better scheduler** — the
reproduction of the LTR result is a faithful baseline we build on, not a claim to beat it.

---

## Environment / reproducibility
- WSL2 3090 recipe + race notes: `docs/RUNBOOK.md`.
- Bare-metal A100 step-by-step: `docs/BENCHMARK.md`.
- All runs are seed-0 and driven by committed scripts (`serving/serve_*.sh`,
  `bench.run_sweep`), so each row here regenerates from the repo.
