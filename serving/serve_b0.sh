#!/usr/bin/env bash
# B0 — vanilla vLLM v1: FCFS scheduling, no KV optimization.
#
# This IS the B0 baseline: default engine, FCFS scheduler, prefix caching
# disabled so the KV cache carries no optimization (a clean floor matching the
# prior paper's FCFS baseline). Every knob comes from env/b0_vanilla.env and is
# overridable from the environment.
#
# Usage:
#   serving/serve_b0.sh                     # Llama-3.1-8B on defaults
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct MAX_MODEL_LEN=4096 serving/serve_b0.sh   # smoke
#   PORT=8001 serving/serve_b0.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# shellcheck source=/dev/null
source "env/b0_vanilla.env"

# --- WSL2 / minimal-toolchain local-dev workarounds (ignored on the A100 host) ---
# 1. vLLM disables pinned memory (hence UVA) on WSL2 by default; our kernel
#    supports it, so enable it or the v1 engine aborts "UVA is not available".
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
# 2. Use the pure-torch sampler; flashinfer's sampler JIT-compiles with nvcc.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# 2b. Pin the engine<->API-server ZMQ loopback to IPv4. vLLM v1's startup
#     handshake over loopback can race under WSL2 (esp. for slow-starting big
#     models like Llama-8B), hanging every endpoint after "startup complete".
export VLLM_LOOPBACK_IP="${VLLM_LOOPBACK_IP:-127.0.0.1}"
# 3. Skip torch.compile + CUDA graphs for fast startup. NB: vLLM v1 STILL needs
#    a C compiler for Triton's CUDA-driver init — install build-essential +
#    python3-dev once (see docs/RUNBOOK.md). On the A100 (full toolchain) set
#    ENFORCE_EAGER=0 and NO_TORCH_COMPILE=0 for representative performance.
: "${ENFORCE_EAGER:=1}"
: "${NO_TORCH_COMPILE:=1}"
EXTRA_ARGS=()
if [ "${ENFORCE_EAGER}" = "1" ]; then EXTRA_ARGS+=(--enforce-eager); fi
if [ "${NO_TORCH_COMPILE}" = "1" ]; then EXTRA_ARGS+=(--compilation-config '{"mode":0}'); fi
# Optional: cap the KV pool to force preemption under load (B0/B1 comparison).
if [ -n "${NUM_GPU_BLOCKS:-}" ]; then EXTRA_ARGS+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS}"); fi

# Prefer the repo venv's vllm, fall back to PATH.
VLLM_BIN="${VLLM_BIN:-$HERE/.venv/bin/vllm}"
[ -x "$VLLM_BIN" ] || VLLM_BIN="vllm"

echo "[serve_b0] model=${MODEL} dtype=${DTYPE} max_model_len=${MAX_MODEL_LEN} \
gpu_mem_util=${GPU_MEM_UTIL} port=${PORT}"

set -x
exec "$VLLM_BIN" serve "${MODEL:-meta-llama/Llama-3.1-8B-Instruct}" \
  --dtype "${DTYPE:-float16}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --scheduling-policy fcfs \
  --no-enable-prefix-caching \
  "${EXTRA_ARGS[@]}" \
  --seed "${SEED:-0}" \
  --no-enable-log-requests \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}"
# NB: --scheduling-policy fcfs is vLLM's default; stated explicitly so B0 is
# self-documenting. Prometheus /metrics stays enabled (log-stats on by default)
# so the harness can read preemptions + KV usage.
