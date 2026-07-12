# RUNBOOK — run B0 and B1 end to end

Recorded environment: WSL2 + RTX 3090 24 GB, Python 3.10. The B0/B1 pipeline runs
against whatever vLLM the serve script launches; the **headline C1+C2 stack is pinned
to vLLM 0.25.0** (see `docs/V025_SMOKE.md`). Results land in
`results/summaries/<config>.json` (+ `<config>_sweep.csv`), one file per config.

---

## 0. One-time environment setup

```bash
# from the repo root
python3 -m venv .venv --without-pip          # WSL2/Debian: ensurepip is often absent
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python   # bootstrap pip
.venv/bin/pip install -U pip
pip install -U uv                            # optional but faster (or use .venv/bin/pip)

# the serving engine (GPU host) + the harness (editable) + dev/ltr extras
.venv/bin/pip install vllm
uv pip install --python .venv/bin/python -e ".[dev,ltr]"

# gated models/datasets — accept the licenses in a browser first:
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct   (Meta form)
#   https://huggingface.co/datasets/lmsys/lmsys-chat-1m       (click-through)
.venv/bin/hf auth login          # paste a READ token; or set HF_TOKEN
```

### WSL2 / minimal-toolchain hosts (local RTX 3090 dev)
vLLM v1 JIT-compiles Triton kernels at runtime, so a **C compiler is required**:
```bash
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
```
The serve scripts export these automatically (ignored on non-WSL hosts):
- `VLLM_WSL2_ENABLE_PIN_MEMORY=1` — else the v1 engine aborts *"UVA is not
  available"* (WSL2 disables pinned memory by default; our kernel supports it).
- `VLLM_USE_FLASHINFER_SAMPLER=0` — flashinfer's sampler JIT needs `nvcc`
  (absent here); the torch-native sampler avoids it.
- `ENFORCE_EAGER` / `NO_TORCH_COMPILE` — see the hardware notes below.

Sanity check the GPU is visible from torch:
```bash
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"   # -> NVIDIA GeForce RTX 3090
```

---

## 1. B0 — vanilla vLLM v1 (FCFS, no KV optimization)

```bash
# terminal 1: launch the server (defaults: Llama-3.1-8B, FP16, ctx 8192)
serving/serve_b0.sh
# wait until it responds:
curl -sf http://127.0.0.1:8000/v1/models

# terminal 2: drive the request-rate sweep
.venv/bin/python -m bench.run_sweep --config b0 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --source lmsys -n 500 --rates 5,10,20,30,40,50,60 \
  --gpu RTX3090-24GB
```
→ writes `results/summaries/b0.json` + `b0_sweep.csv`.

---

## 2. B1 — reproduce the LTR baseline (~2.1× over FCFS)

See `ltr/README.md` for the design. Steps:
```bash
# 1. fine-tune the OPT-125M ranker (ListMLE) on LMSYS output lengths
#    LLM.pdf spec: --n 23800 --epochs 10 (expanded from 10k; >10 epochs overfits).
#    Use a smaller --n (e.g. 2000) only for a quick local smoke test.
.venv/bin/python -m ltr.ranker.train --out results/ranker/opt125m-ltr --n 23800 --epochs 10

# 2. check ranking quality (Kendall tau) on a held-out split
.venv/bin/python -m ltr.ranker.eval --ranker results/ranker/opt125m-ltr

# 3. serve with priority scheduling (LTR ordering, no engine patching)
serving/serve_b1_ltr.sh

# 4. same sweep, config b1
.venv/bin/python -m bench.run_sweep --config b1 \
  --model meta-llama/Llama-3.1-8B-Instruct --source lmsys -n 500 \
  --rates 5,10,20,30,40,50,60 --gpu RTX3090-24GB
```
Compare: `b1.json` vs `b0.json` should show the latency reduction at high load.

---

## 3. Hardware notes

**RTX 3090-24 GB (WSL2) is the platform for every number in this repo.** An A100-40 GB —
the prior paper's platform — would reach the same memory pressure natively at 30–60 req/s;
that native re-run is optional future work (`docs/BENCHMARK.md`). If you do run it, serve
with compilation on: `ENFORCE_EAGER=0 NO_TORCH_COMPILE=0 serving/serve_b0.sh`.

**RTX 3090-24GB (WSL2, local dev).** Llama-3.1-8B FP16 **does run here**, but
only past a vLLM v1 ZMQ startup-handshake race that hangs slow-starting models
under WSL2 loopback. Validated working recipe (~51 tok/s, produced the real
`b0.json`):
```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=4096 GPU_MEM_UTIL=0.88 \
  ENFORCE_EAGER=0 NO_TORCH_COMPILE=0 serving/serve_b0.sh
.venv/bin/python -m bench.run_sweep --config b0 --model meta-llama/Llama-3.1-8B-Instruct \
  --source lmsys -n 100 --rates 5,20,40 --max-tokens 128 --gpu RTX3090-24GB
```
Key knobs (baked into the serve scripts): `VLLM_LOOPBACK_IP=127.0.0.1`,
`MAX_MODEL_LEN=4096` + `GPU_MEM_UTIL=0.88` (faster startup, ~1 GB headroom),
compilation on. **The race is probabilistic:** if the log says `Application
startup complete` but `/v1/models` and even `/ping` then time out with the GPU
idle, it lost the race — kill the server and relaunch.

If it wedges repeatedly, switch WSL networking mirrored→NAT (more-tested
loopback): edit `C:\Users\<you>\.wslconfig` → `networkingMode=nat` (keep
`localhostforwarding=true`), then `wsl --shutdown` from Windows and reopen WSL.

The proposal's open 3090 model runs with no race and lots of headroom:
```bash
MODEL=Qwen/Qwen2.5-3B-Instruct MAX_MODEL_LEN=8192 GPU_MEM_UTIL=0.55 serving/serve_b0.sh
```
> Every headline number in this repo comes from the RTX 3090, which runs the full
> pipeline and both the spec model and the fallback. The A100 re-run at the paper's
> native scale is optional future work (`docs/BENCHMARK.md`).

---

## 4. Figures
```bash
.venv/bin/python -m bench.plots --configs b0,b1 --metric e2e_mean_ms --bar-rate 60
```
→ `results/summaries/latency_vs_rate_e2e_mean_ms.png` and a comparison bar chart.

## 5. Tests / CI
```bash
.venv/bin/python -m pytest -q -m "not gpu"    # CPU-only suite
.venv/bin/ruff check .
```
