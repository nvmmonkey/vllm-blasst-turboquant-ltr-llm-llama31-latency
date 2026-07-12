# BENCHMARK.md — optional bare-metal A100 re-run (future work)

> **Every result in this repo was measured on the RTX 3090 (WSL2) — see
> `docs/RUNBOOK.md`.** This file is the *optional* recipe for re-running the same
> harness at the prior paper's native scale on a bare-metal **1× NVIDIA A100 40 GB**,
> where the KV pool fills at 30–60 req/s without the dtype-scaled cap the 3090 needs.
> It is future work, not a dependency of any number we report.

The A100 path has **no WSL2 workarounds** — on bare metal with a normal CUDA stack,
vLLM runs with its default multiprocessing + `torch.compile` + CUDA graphs, and the
Llama-8B ZMQ startup race we hit under WSL2 does not occur. The serve scripts still
export the WSL2 env vars, but they are harmless no-ops here. The headline C1+C2 stack is
pinned to vLLM 0.25.0 (`docs/V025_SMOKE.md`). Results → `results/summaries/<config>.json`
(one file per config) + `<config>_sweep.csv`.

---

## 0. System prerequisites

```bash
# NVIDIA driver must already be installed; confirm the GPU is visible:
nvidia-smi                         # expect: A100-SXM4-40GB (or 80GB)

# Build tools (Triton JITs kernels at runtime) + venv + git:
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv git
```
- No separate CUDA toolkit needed — vLLM's wheels bundle the CUDA runtime.
- A100-40GB comfortably holds Llama-3.1-8B FP16 (~16 GB weights) + a large KV
  cache at `--max-model-len 8192`.

---

## 1. Python environment

```bash
git clone <this repo> capstone && cd capstone
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install vllm                       # the serving engine (v1 by default)
pip install -e ".[dev,ltr]"            # harness (editable) + dev + LTR extras
python -c "import torch; print(torch.cuda.get_device_name(0))"   # sanity: A100
```

---

## 2. Hugging Face (gated model + dataset)

Accept the licenses once in a browser (logged in):
- Model: <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
- Dataset: <https://huggingface.co/datasets/lmsys/lmsys-chat-1m>

Then authenticate + pre-download:
```bash
hf auth login                          # paste a READ token (or: export HF_TOKEN=hf_...)
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
hf download facebook/opt-125m          # the LTR ranker backbone
```

---

## 3. Config files (what to check / edit)

| File | Role | A100 value |
|---|---|---|
| `configs/model.yaml` | canonical model + sampling (source of truth) | Llama-3.1-8B, fp16, ctx 8192, greedy, seed 0 |
| `env/b0_vanilla.env` | B0 serve knobs | `GPU_MEM_UTIL=0.90`, `MAX_MODEL_LEN=8192`, `PORT=8000` |
| `env/b1_ltr.env` | B1 serve knobs + `RANKER_PATH` | same model; `RANKER_PATH=results/ranker/opt125m-ltr` |

**One A100 override vs the defaults:** the serve scripts default to eager mode
for the toolchain-limited dev box. On bare metal, turn compilation + CUDA
graphs ON for representative latency:
```bash
export ENFORCE_EAGER=0 NO_TORCH_COMPILE=0     # prepend to the serve commands below
```

---

## 4. B0 — vanilla vLLM v1 (FCFS, no KV optimization)

```bash
# Terminal 1 — serve (compilation on):
ENFORCE_EAGER=0 NO_TORCH_COMPILE=0 serving/serve_b0.sh
#   waits on: curl -sf http://127.0.0.1:8000/v1/models

# Terminal 2 — the full request-rate sweep (spec rates 5–60):
python -m bench.run_sweep --config b0 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --source lmsys -n 500 --warmup 16 \
  --rates 5,10,20,30,40,50,60 \
  --max-tokens 256 --gpu A100-40GB
```
→ `results/summaries/b0.json` (+ `b0_sweep.csv`). This is the FCFS floor.

---

## 5. B1 — reproduce the LTR baseline (~2.1× over FCFS)

```bash
# 5.1 Train the OPT-125M ranker (ListMLE) — PAPER SPEC: 23,800 samples, 10 epochs
#     (LLM.pdf §III-B/§IV: expanded from 10k; >10 epochs overfits). The ranker
#     (OPT-125M) is a DIFFERENT model from the served target (Llama-3.1-8B).
python -m ltr.ranker.train --out results/ranker/opt125m-ltr \
  --base facebook/opt-125m \
  --target-model meta-llama/Llama-3.1-8B-Instruct \
  --source lmsys --n 23800 --list-size 16 --epochs 10

# 5.2 Report ranking quality (Kendall tau) on a held-out split:
python -m ltr.ranker.eval --ranker results/ranker/opt125m-ltr \
  --target-model meta-llama/Llama-3.1-8B-Instruct

# 5.3 Serve with priority scheduling (LTR ordering; no engine patching):
ENFORCE_EAGER=0 NO_TORCH_COMPILE=0 serving/serve_b1_ltr.sh   # --scheduling-policy priority

# 5.4 Same sweep, but stamp each request's priority from the ranker:
python -m bench.run_sweep --config b1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --ltr-ranker results/ranker/opt125m-ltr \
  --source lmsys -n 500 --warmup 16 \
  --rates 5,10,20,30,40,50,60 \
  --max-tokens 256 --gpu A100-40GB
```
→ `results/summaries/b1.json`. B1 vs B0 should show the latency reduction at
high load (30–60 req/s), where KV pressure and preemptions dominate.

> **Two models, two roles:** the ranker is **OPT-125M** (predicts output-length
> order); the served target is **Llama-3.1-8B**. Give the ranker the SAME
> `--target-model` for train/eval/serve so its length labels match the served
> model.
>
> **Sample count (LLM.pdf spec): 23,800**, expanded from an original 10,000, at
> 10 epochs (the original Fu-LTR used 10k / 5 epochs). *Honest boundary:* the
> paper **synthesized** those 23,800 with a private `synthesize_dataset.py`; we
> use 23,800 **real** LMSYS conversations' reference-reply lengths as labels —
> a close approximation, not the identical synthetic set.

---

## 6. Compare + figures

```bash
python -m bench.plots --configs b0,b1 --metric e2e_mean_ms --bar-rate 60
python -m bench.plots --configs b0,b1 --metric preemptions --bar-rate 60
```
→ latency-vs-rate line charts + B0/B1 comparison bars in `results/summaries/`.

Key columns to read in `<config>_sweep.csv`: `e2e_p99_ms`, `tpot_mean_ms`,
`preemptions`, `kv_peak_gb`, `peak_batch` — the mechanism claim is **fewer
preemptions at the same load** for B1 vs B0.

---

## 7. After B0/B1 — the C-tiers

With B0 (floor) and B1 (LTR bar) recorded, C1/C2/C3 each add a KV-cache
technique *on top of B1* (same harness, same rates), and are compared back to
B1:
- **C1** B1 + KV quantization · **C2** B1 + attention sparsity. (C3 head-wise offloading and C4 the unified control plane are future work.)
Add each as a new `--config c1` (etc.) run → `results/summaries/c1.json`, and
extend `bench.plots --configs b0,b1,c1,...`.

---

## Appendix — bare-metal vs the WSL2 dev box

| | bare-metal A100 (this guide) | WSL2 RTX 3090 (RUNBOOK.md) |
|---|---|---|
| Compilation | ON (`ENFORCE_EAGER=0 NO_TORCH_COMPILE=0`) | OFF by default (or race-tuned) |
| `max_model_len` | 8192 (spec) | 4096 (memory) |
| `gpu_memory_utilization` | 0.90 | 0.88 |
| WSL2 env vars | harmless no-ops | required (UVA pin, IPv4 loopback, flashinfer-off) |
| Llama-8B ZMQ startup race | not present | present (probabilistic; see RUNBOOK) |
