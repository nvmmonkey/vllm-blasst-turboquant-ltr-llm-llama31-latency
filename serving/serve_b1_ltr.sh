#!/usr/bin/env bash
# B1 — LTR-scheduled vLLM v1 (reproduce the prior paper's ~2.1x over FCFS).
#
# Same model/dtype/ctx as B0, but the scheduler is driven by the OPT-125M
# output-length ranker. We reach this on modern vLLM v1 WITHOUT patching the
# scheduler or block manager (the crash source in the prior attempt, paper
# SS IV-D): the server runs --scheduling-policy priority, and the harness
# attaches each request's priority from the ranker (ltr/scheduler/priority.py).
#
# Usage:
#   serving/serve_b1_ltr.sh
#   MODEL=Qwen/Qwen2.5-3B-Instruct GPU_MEM_UTIL=0.55 serving/serve_b1_ltr.sh   # 3090
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# shellcheck source=/dev/null
source "env/b1_ltr.env"

# Same WSL2 / minimal-toolchain workarounds as B0 (ignored on the A100 host).
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_LOOPBACK_IP="${VLLM_LOOPBACK_IP:-127.0.0.1}"  # dodge the WSL2 v1 ZMQ loopback race
: "${ENFORCE_EAGER:=1}"
: "${NO_TORCH_COMPILE:=1}"
EXTRA_ARGS=()
if [ "${ENFORCE_EAGER}" = "1" ]; then EXTRA_ARGS+=(--enforce-eager); fi
if [ "${NO_TORCH_COMPILE}" = "1" ]; then EXTRA_ARGS+=(--compilation-config '{"mode":0}'); fi
# Optional: cap the KV pool to force preemption under load (B0/B1 comparison).
if [ -n "${NUM_GPU_BLOCKS:-}" ]; then EXTRA_ARGS+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS}"); fi
# Optional: C1 KV quantization (fp8). On v1 (recompute) this tests whether
# quantization's extra capacity avoids the costly recompute — where swap can't.
if [ -n "${KV_CACHE_DTYPE:-}" ]; then EXTRA_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}"); fi

VLLM_BIN="${VLLM_BIN:-$HERE/.venv/bin/vllm}"
[ -x "$VLLM_BIN" ] || VLLM_BIN="vllm"

echo "[serve_b1] model=${MODEL} policy=priority (LTR ranker) port=${PORT}"

set -x
exec "$VLLM_BIN" serve "${MODEL:-meta-llama/Llama-3.1-8B-Instruct}" \
  --dtype "${DTYPE:-float16}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --scheduling-policy priority \
  --no-enable-prefix-caching \
  "${EXTRA_ARGS[@]}" \
  --seed "${SEED:-0}" \
  --no-enable-log-requests \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}"
# The ONLY engine difference vs B0 is --scheduling-policy priority. The ranker
# runs client-side in the harness, which sends each request's `priority`.
