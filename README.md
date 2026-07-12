# KVCache-Coordinated Latency Optimization for LLM Inference Serving

Capstone (CSCI 6806, FDU). A **KV-cache optimization layer that sits _beneath_ a
Learning-to-Rank (LTR) request scheduler** on vLLM. We **build on** the prior LTR scheduling
work (Fu et al., NeurIPS 2024 [12]; Kumar et al. [11]) — we do **not** replace it — and ask a
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
a lever that *doesn't* help in a regime is reported as a finding, not hidden.

- **The LTR baseline reproduces — and it needs swap.** On the 0.4.1 swap stack the paper's
  LTR+swap benefit is real: at the moderate-overload sweet spot, `opt` vs `fcfs` is **1.66×
  mean / 3.6× P99** per-token latency, bracketing the reported 2.1×. On recompute-only engines
  an *accurate* SJF ranker can make things **worse** (it starves long requests into full-prefill
  recompute) — the load-dependence the single "2.1×" hides.
- **C1 (TurboQuant) is a first-class TTFT / tail / preemption lever.** On v0.25 V1 at equal GPU
  memory (2.83× the token capacity), C1 cuts **r64 TTFT 6.8× over FCFS**, **p99 6.1× over
  LTR-alone**, and takes **4× fewer preemptions** — and it's *capacity*, not codec magic (a bf16
  capacity-oracle at the same token budget matches it). **Honest cost:** TurboQuant's decode tax
  raises TPOT ~1.4×→2.5×, so end-to-end is a wash at the highest loads. C1 is for **TTFT-bound /
  interactive** serving.
- **C2 (BLASST) — same algorithm, opposite sign on the two decode kernels.** 42% block sparsity
  at perplexity-lossless τ=6. In serving: on the bf16 GQA-shared `kernel_unified_attention` the
  CTA-uniform tile tax makes it **+4.5% TPOT (net-negative)**; on the per-head TurboQuant
  `_tq_decode_stage1` it lands **−22% TPOT / −47% p99 (net-positive)** and also skips the value
  dequant. C2's payoff tracks **compute density (batch size), not context length** — it fades
  once low batch / long context turns decode memory-bound.
- **Accuracy is preserved.** GSM8K: bf16 0.80 vs TurboQuant-4bit 0.83 (Δ within noise); BLASST
  at τ=6 is perplexity-lossless. The C1+C2 latency wins are essentially free on quality.

Full tables, percentiles, ablations, and the honest negatives live in **`results/RESULTS.md`**
and the standalone report **`results/report.html`**.

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
results/   summaries/ (per-config JSON) · RESULTS.md · report.html · ranker_meta
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
# summaries land in results/summaries/ ; the report is results/report.html
```

External (not committed — clone/build locally): the vendored vLLM forks (`ltr/vendor/`, see
`ltr/vendor/PATCHES.md`), the `.venv-v025` environment, and the ranker weights under
`results/ranker/` (`ranker_meta.json` is committed).

## Contributors

| Member | Area |
|---|---|
| **Guoliang Liu** | LTR baseline — B0/B1, the 0.4.1 swap-stack 2.1× reproduction, the OPT-125M ranker |
| **Wenhui Kang** | KV layer — C1 native TurboQuant serving, C2 BLASST kernels + serving integration |
| **Junpeng Huang** | Benchmark harness, the request-rate / long-context runs, results + report |

## References

**Prior work we build on**
- [11] A. Saravana Kumar, V. Janarthanan, S. Sharma, and K. Palani, "An empirical study on
  latency reduction techniques for large language models," Olsen Coll. Eng. Sci., Fairleigh
  Dickinson Univ., 2026.
- [12] Y. Fu, S. Zhu, R. Su, A. Qiao, I. Stoica, and H. Zhang, "Efficient LLM scheduling by
  learning to rank," in *NeurIPS*, vol. 37, 2024, pp. 59006–59029. — code:
  `github.com/hao-ai-lab/vllm-ltr`

**KV-cache techniques**
- [1] A. Zandieh et al., "TurboQuant: Online vector quantization with near-optimal distortion
  rate," arXiv:2504.19874, 2025.
- [2] F. Li et al., "KVmix: Gradient-based layer importance-aware mixed-precision quantization
  for KV cache," in *AAAI*, 2026. — code: `github.com/LfLab-AI/KVmix`
- [4] J. Yuan et al., "BLASST: Dynamic blocked attention sparsity via softmax thresholding," in
  *MLSys*, 2026.
- [5] C. Luo et al., "HeadInfer: Memory-efficient LLM inference by head-wise offloading,"
  arXiv:2502.12574, 2025.

**Systems / serving**
- [7] W. Kwon et al., "Efficient memory management for LLM serving with PagedAttention," in
  *SOSP*, 2023. — vLLM: `github.com/vllm-project/vllm`
- [8] Y. Li et al., "EAGLE-3: Scaling up inference acceleration...," arXiv:2503.01840, 2025.
- [10] Y. Liu et al., "LMCache: An efficient KV cache layer for enterprise-scale LLM inference,"
  arXiv:2510.09665, 2025. — code: `github.com/LMCache/LMCache`

**Data**
- LMSYS-Chat-1M: `huggingface.co/datasets/lmsys/lmsys-chat-1m`
