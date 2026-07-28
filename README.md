# KVCache-Coordinated Latency Optimization for LLM Inference Serving

Capstone (CSCI 6806, FDU). A **KV-cache optimization layer that sits _beneath_ a
Learning-to-Rank (LTR) request scheduler** on vLLM. We **build on** the prior LTR scheduling
work (Fu et al., NeurIPS 2024 [2]; Kumar et al. [3]) — we do **not** replace it — and ask a
focused question: *once LTR is ordering requests, do KV-cache levers cut latency further,
without hurting accuracy, and where does each lever actually pay off?*

Two KV levers, single-GPU:

- **C1 — KV quantization (TurboQuant 4-bit).** Shrinks the KV cache → more concurrent
  requests at equal GPU memory → fewer preemptions, lower TTFT and tail under memory pressure.
- **C2 — attention sparsity (BLASST).** Online-softmax block-skipping inside the decode
  kernel → less attention compute per token, in the compute-bound regime.

C3 (head-wise offloading), C4 (a Rust control plane unifying the tiers), and C5 (speculative
decoding) are **future work** and are not in this repo.

## Platform

- **GPU:** 1× RTX 3090 24 GB (SM86), WSL2 (kernel 6.6), model `meta-llama/Llama-3.1-8B-Instruct`.
- **Three engines** — the same lever behaves differently under each preemption model, which is
  half the story:
  - vLLM **0.4.1** fork (`hao-ai-lab/vllm-ltr`) — **swap** preemption; the reference stack that
    reproduces the paper's LTR result.
  - vLLM **0.8.5** V0 — **recompute** preemption; the modern baseline.
  - vLLM **0.25.0** V1 — native **TurboQuant** KV backend; the headline C1 stack (V1 engine +
    priority scheduling both run on the 3090, where the 0.8.5 V0 stack could not).
- **Workload:** LMSYS-Chat-1M, Poisson arrivals, request-rate sweep. **Metrics:** TTFT / TPOT /
  e2e (mean + p50/p90/p99), throughput, peak KV, preemptions, and GSM8K accuracy.

> **On scale.** The 3090 is compute-limited, so to reach the paper's *memory-bound* regime we
> cap the KV pool (dtype-scaled to hold GPU KV **bytes** equal across arms — fp16 512 blocks vs
> TQ4 724). An A100-40 GB would reach the same pressure natively at 30–60 req/s; that native
> re-run is future work, not a dependency. **Every number below is measured on the 3090.**

## What we found

Honest and **regime-dependent** — we deliberately never quote a single speedup multiplier, and
a lever that *doesn't* help in a regime is reported as a finding, not hidden. Every number below
is the one printed in the paper (`latex_source/main.tex`), measured on the `*_triton` / `*_tq4`
sweep in `results/summaries/`.

- **The LTR baseline reproduces — and it needs swap.** On the 0.4.1 swap stack the prior LTR
  benefit is real: at the moderate-overload sweet spot, `opt` vs `fcfs` is **1.66× mean / 3.6×
  P99** normalized latency, bracketing the reported 2.1×. On recompute-only engines an
  *accurate* SJF ranker can make things **worse** — it starves long requests into full-prefill
  recompute — which is the load-dependence a single "2.1×" hides.
- **C1 (TurboQuant) is a TTFT / tail / preemption lever, and it costs decode time.** On v0.25
  V1 at equal GPU memory (**2.83×** the token capacity, 512/8,192 → 724/23,168 blk/tok), at
  64 req/s C1 cuts mean TTFT **12,275 → 2,347 ms** (5.2× over FCFS, 2.8× over LTR alone) and
  preemptions **162 → 42** per run. It is *capacity*, not codec magic. In the control
  arms (the `*_v025` sweep, where `ctrl` and `c1te` live): `ctrl` — bf16 handed the same
  23,168-token budget — gets there and slightly past it (mean TTFT **1,319 vs 1,585 ms**),
  because it pays no dequantization tax; `c1te` — TQ4 held to the baseline 8,192-token budget —
  does not (**6,234 ms**, no better than LTR alone at 5,552 ms). **The cost is not a wash.**
  Mean TPOT rises **67.9 → 214.7 ms**, so mean E2E gets **37.5 % worse** than LTR alone
  (15,063 → 20,708 ms). C1 alone is for **TTFT-bound / interactive** serving, not for E2E.
- **C2 (BLASST) — same algorithm, opposite sign on the two decode kernels.** At the
  perplexity-lossless τ=6 the screen identifies **42.1 %** of blocks as skippable. In serving,
  on the bf16 GQA-shared `kernel_unified_attention` the CTA-uniform tile tax makes it
  **+4.9 % mean TPOT / +15.0 % P99 (net-negative)**; on the per-head TurboQuant
  `_tq_decode_stage1` the same code lands **−22.1 % mean TPOT / −46.8 % P99 (net-positive)**
  and also skips the value dequant. The cause is measured, not inferred: four query heads share
  a tile, so **42.1 % identified sparsity becomes only 18.5 % realized** on the GQA kernel,
  which is below the break-even for the screen's own cost.
- **C2's payoff tracks compute density (batch size), not context length.** At a fixed 6 req/s:
  512-token context / batch 40 → **−20.4 %** TPOT; 2,048 / batch 21 → −10.3 %; 4,096 / batch 10
  → +0.6 %; 7,168 / batch 6 → +2.8 %. The screen is a fixed per-block cost, and a thin batch
  cannot amortize it.
- **Together they are complementary — but sparsity refunds the tax, it does not erase it.**
  B1+C1+C2 at 64 req/s: TTFT **1,747 ms**, mean TPOT **167.3 ms**, 40 preemptions per run, and
  mean E2E back down from 20,708 to **17,308 ms**. That is still above LTR alone (15,063 ms):
  the full stack wins decisively on TTFT and preemptions, not on mean end-to-end.
- **No measured accuracy loss.** GSM8K, n=100, greedy: bf16 **0.80**, bf16+C2 **0.88**, TQ4
  **0.83**, TQ4+C2 **0.85**. Nothing falls below baseline, but at n=100 a ±0.05 swing is inside
  sampling noise — the defensible claim is *no measured degradation*, not higher accuracy. τ=6
  was fixed by a perplexity sweep before these runs.

The paper is **`latex_source/main.tex`** (`make -C latex_source` to build it). Full tables,
percentiles, ablations and the honest negatives live in **`results/RESULTS.md`**.

## Repository layout

```
bench/     benchmark harness — LMSYS loader, Poisson loadgen, metrics/percentiles, sweep runner, plots
ltr/       LTR baseline — OPT-125M ranker (ListMLE) + eval, vLLM priority mapping, GSM8K accuracy harness
c2/        C2 BLASST — block-skip screen, standalone Triton decode kernel, GQA decode tax, perplexity
serving/   launch + benchmark drivers
           ├── serve_b0.sh / serve_b1_ltr.sh   B0 vanilla / B1 LTR
           ├── bench_reference.sh              0.4.1 swap-stack 2.1× reproduction
           ├── v025_smoke/                     C1 native TurboQuant on vLLM 0.25 (V1)
           └── v025_blasst/                    C2 runtime kernel patch (sitecustomize) + serving drivers
results/   summaries/ (per-config JSON) · RESULTS.md · figures/ (paper + slide renders) · ranker_meta
latex_source/  the paper — main.tex, references.bib, Makefile, figures/, and scripts/ that
           regenerate every figure straight from results/summaries/ (plus the deck builders)
tests/     CPU unit tests (loadgen, datasets, metrics, ranker, scheduler)
docs/      RUNBOOK (3090/WSL2 run guide) · BENCHMARK (optional A100 recipe) · C2_BLASST_PLAN · C_TIERS · REFERENCES
```

## Reproduce

Full steps in **`docs/RUNBOOK.md`** (3090 / WSL2). Short version — the headline C1 + C2 path on
vLLM 0.25:

```bash
# 1. build the pinned env (vLLM 0.25.0)          → docs/V025_SMOKE.md
# 2. C1 (TurboQuant) vs C1+C2 (BLASST) on the per-head decode kernel:
serving/v025_blasst/c2_serving_tq4.sh
# 3. baselines (B0/B1, both backends) + GSM8K quality:
serving/v025_blasst/b0_and_quality.sh
# summaries land in results/summaries/ ; full results in results/RESULTS.md
```

External (not committed — clone/build locally): the vendored vLLM forks (`ltr/vendor/`, see
`ltr/vendor/PATCHES.md`) and the `.venv-v025` environment. The trained OPT-125M LTR ranker is
published on Hugging Face — **[nvmmonkey/opt125m-ltr-ranker](https://huggingface.co/nvmmonkey/opt125m-ltr-ranker)**
— so B1 runs without retraining (`ranker_meta.json` is committed for reference). The target-sampled output-length labels are on Hugging Face too — **[nvmmonkey/llama31-8b-output-lengths](https://huggingface.co/datasets/nvmmonkey/llama31-8b-output-lengths)** (LMSYS prompts withheld per license).

## Contributors

| Member | Area |
|---|---|
| **Guoliang Liu** | LTR baseline — B0/B1, the 0.4.1 swap-stack 2.1× reproduction, the OPT-125M ranker |
| **Wenhui Kang** | KV layer — C1 native TurboQuant serving, C2 BLASST kernels + serving integration |
| **Junpeng Huang** | Benchmark harness, the request-rate / long-context runs, results + report |

## References

Numbering matches the reference list in `latex_source/main.tex`.

**Systems / serving**
- [1] W. Kwon et al., "Efficient memory management for LLM serving with PagedAttention," in
  *SOSP*, 2023. — vLLM: `github.com/vllm-project/vllm`
- [9] Y. Zhong et al., "DistServe: Disaggregating prefill and decoding for goodput-optimized
  LLM serving," in *OSDI*, 2024.
- [10] Y. Liu et al., "LMCache: An efficient KV cache layer for enterprise-scale LLM inference,"
  arXiv:2510.09665, 2025. — code: `github.com/LMCache/LMCache`
- [11] Y. Li et al., "EAGLE-3: Scaling up inference acceleration of large language models,"
  arXiv:2503.01840, 2025.
- [12] DeepSeek-AI, "DeepSeek-V3 technical report," arXiv:2412.19437, 2024.

**Prior work we build on**
- [2] Y. Fu, S. Zhu, R. Su, A. Qiao, I. Stoica, and H. Zhang, "Efficient LLM scheduling by
  learning to rank," in *NeurIPS*, vol. 37, 2024, pp. 59006–59029. — code:
  `github.com/hao-ai-lab/vllm-ltr`
- [3] A. Saravana Kumar, V. Janarthanan, S. Sharma, and K. Palani, "An empirical study on
  latency reduction techniques for large language models," Olsen Coll. Eng. Sci., Fairleigh
  Dickinson Univ., 2026.

**KV-cache techniques**
- [4] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online vector quantization
  with near-optimal distortion rate," arXiv:2504.19874, 2025.
- [5] F. Li et al., "KVmix: Gradient-based layer importance-aware mixed-precision quantization
  for KV cache," in *AAAI*, 2026. — code: `github.com/LfLab-AI/KVmix`
- [6] J. Yuan et al., "BLASST: Dynamic blocked attention sparsity via softmax thresholding," in
  *MLSys*, 2026.
- [7] C. Luo et al., "HeadInfer: Memory-efficient LLM inference by head-wise offloading,"
  arXiv:2502.12574, 2025.
- [8] Q. Liu et al., "MELL: Memory-efficient LLM serving via multi-GPU KV cache management,"
  arXiv:2501.06709, 2025.

**Data**
- LMSYS-Chat-1M: `huggingface.co/datasets/lmsys/lmsys-chat-1m`
