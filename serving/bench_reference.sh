#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reference-stack (hao-ai-lab/vllm-ltr, old vLLM 0.4.1 + SWAP) bench driver.
#
# This is the PARITY path for the paper's ~2.1x latency reduction, which needs
# swap preemption (vLLM v1 removed swap — see results/RESULTS.md). It drives the
# fork's own OpenAI server + benchmark_serving_real.py client for ONE arm at ONE
# request-rate, starting the server, waiting for health, running the client, and
# shutting the server down cleanly (by PID + GPU PID — never by name-match).
#
# THREE ARMS decompose the benefit (set SCHED):
#   fcfs     vanilla vLLM engine, FCFS order, RECOMPUTE preemption (paper baseline)
#   fifo     general/ranked engine, arrival order, SWAP preemption  (isolates swap)
#   opt-xxx  general/ranked engine, LTR order (OPT ranker), SWAP    (paper's method)
# fcfs->fifo = the swap contribution; fifo->opt = the LTR-ordering contribution.
#
# Prereqs (see ltr/vendor/PATCHES.md): built vllm-ltr micromamba env + downloaded
#   benchmarks/llama3-8b-sharegpt-test-t1-s0-8192.jsonl (LLM-ltr/Llama3-Trace)
#   benchmarks/MODEL/results/opt-125m-...-score-trainbucket10-b32/ (LLM-ltr/OPT-Predictors)
#
# Example 3-arm sweep at rate 8 (RTX 3090; use MAXLEN 8192 UTIL 0.9 on A100):
#   for S in fcfs fifo; do SCHED=$S RATE=8 bash serving/bench_reference.sh; done
#   SCHED=opt-xxx RATE=8 PREDICTOR=MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json \
#     bash serving/bench_reference.sh
# ---------------------------------------------------------------------------
set -u
MM="${MICROMAMBA:-$HOME/.local/bin/micromamba}"
ENV="${VLLM_LTR_ENV:-vllm-ltr}"
# nvcc lives in the micromamba env, not /usr/local/cuda; fp8 KV's CUDA-version
# check (config.py) needs CUDA_HOME to find it (else it crashes on None).
ENVP="${VLLM_LTR_ENVP:-$HOME/.local/share/mamba/envs/$ENV}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$REPO/ltr/vendor/vllm-ltr/benchmarks"

SCHED="${SCHED:-fcfs}"
RATE="${RATE:-8}"
REQTIME="${REQTIME:-60}"          # paper uses 60; num_prompts = REQTIME*RATE
NUMPROMPTS="${NUMPROMPTS:-}"       # if set (>0), FIXED prompt count (bypasses REQTIME*RATE;
                                   # keeps high-rate runs tractable instead of 60*rate=3840 @ r64)
SWAP="${SWAP:-8}"                  # CPU swap-space GiB (3090/22GB RAM: 8; A100: 16)
PREDICTOR="${PREDICTOR:-}"        # usage_config.json (LTR/opt arm only)
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAXLEN="${MAXLEN:-4096}"          # 3090: 4096; paper/A100: 8192
UTIL="${UTIL:-0.85}"
KVDTYPE="${KVDTYPE:-auto}"        # C1: fp8 halves KV/token (0.4.1 native, no patch)
PORT="${PORT:-8000}"
DATASET="${DATASET:-llama3-8b-sharegpt-test-t1-s0-8192.jsonl}"
RESDIR="${RESDIR:-$BENCH/RESULTS}"
SRVLOG="${SRVLOG:-$RESDIR/srv_${SCHED}_r${RATE}.log}"
CLILOG="${CLILOG:-$RESDIR/cli_${SCHED}_r${RATE}.log}"

mkdir -p "$RESDIR"; cd "$BENCH" || exit 2
# free any prior GPU procs by GPU PID (never name-match "vllm serve")
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 2

PRED_ARG=""; [ -n "$PREDICTOR" ] && PRED_ARG="--prefill-predictor-model-config $PREDICTOR"
echo "=== SERVER sched=$SCHED swap=$SWAP maxlen=$MAXLEN util=$UTIL pred=${PREDICTOR:-none} $(date) ===" > "$SRVLOG"
VLLM_WSL2_ENABLE_PIN_MEMORY=1 CUDA_VISIBLE_DEVICES=0 CUDA_HOME="$ENVP" \
  PYTHONPATH="$REPO:${PYTHONPATH:-}" VLLM_KV_ROTATE="${VLLM_KV_ROTATE:-0}" \
  "$MM" run -n "$ENV" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --swap-space "$SWAP" --disable-log-requests \
  --schedule-type "$SCHED" --enable-chunked-prefill --enforce-eager \
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --port "$PORT" \
  --kv-cache-dtype "$KVDTYPE" \
  $PRED_ARG >> "$SRVLOG" 2>&1 &
SRV_PID=$!

ready=0
for i in $(seq 1 120); do
  kill -0 "$SRV_PID" 2>/dev/null || { echo "SERVER DIED during load"; break; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] \
    && { ready=1; echo "SERVER READY after ${i}0s"; break; }
  sleep 10
done

if [ "$ready" = "1" ]; then
  NP_ARG="--num-prompts -1 --request-time $REQTIME"
  [ -n "$NUMPROMPTS" ] && NP_ARG="--num-prompts $NUMPROMPTS"
  echo "=== CLIENT sched=$SCHED rate=$RATE $NP_ARG $(date) ===" > "$CLILOG"
  "$MM" run -n "$ENV" python benchmark_serving_real.py --backend vllm \
    --model "$MODEL" --tokenizer "$MODEL" --dataset "$DATASET" \
    $NP_ARG --schedule-type "$SCHED" \
    --output-len -1 --request-rate "$RATE" --result-dir "$RESDIR" >> "$CLILOG" 2>&1
  echo "CLIENT rc=$? $(date)" >> "$CLILOG"
fi

kill "$SRV_PID" 2>/dev/null; sleep 5
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
echo "=== DONE sched=$SCHED ready=$ready $(date) ==="
