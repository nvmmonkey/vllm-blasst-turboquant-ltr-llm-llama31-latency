#!/usr/bin/env bash
# Parametrized vLLM 0.25 V1 server for the C1-native bench (RTX 3090 / WSL2).
# Bakes in the WSL2 UVA/pin-memory fix. torchcodec must already be uninstalled.
#   KV=auto|turboquant_4bit_nc  POL=fcfs|priority  OVR=<blocks>  PORT=8000  bash serve_v025.sh
set -u
cd /home/mking/capstone || exit 2
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
KV="${KV:-auto}"; POL="${POL:-fcfs}"; OVR="${OVR:-512}"; PORT="${PORT:-8000}"; MAXLEN="${MAXLEN:-2048}"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO
ARGS=(--dtype bfloat16 --max-model-len "$MAXLEN" --gpu-memory-utilization 0.85
      --enforce-eager --no-enable-prefix-caching
      --num-gpu-blocks-override "$OVR" --scheduling-policy "$POL"
      --port "$PORT")
[ "$KV" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV")
echo "[serve_v025] KV=$KV POL=$POL OVR=$OVR PORT=$PORT"
exec .venv-v025/bin/vllm serve "$MODEL" "${ARGS[@]}"
