#!/usr/bin/env bash
# B0 baselines on the C2-arm backends + joint C1+C2 quality (GSM8K on TQ4+BLASST).
# Completes two clean SAME-BACKEND ladders (absolute values comparable within each):
#   bf16 / TRITON:    b0_triton (B0 fcfs) -> b1_triton (B1 LTR) -> c2_bf16 (B1+C2, tau6)
#   TQ4 / turboquant: b0_tq4 (B0 fcfs)    -> c1_tq4 (B1+C1)     -> c1c2_tq4 (B1+C1+C2, tau6)
# And answers "is BLASST lossless on TQ4-quantized KV?" via GSM8K through the REAL
# _tq_decode_stage1 decode kernel at tau=6, vs TQ4 dense (gsm8k_tq4_v025 = 0.83, n=100).
cd /home/mking/capstone || exit 2
L="${L:-/tmp/kvbench}"
MODEL=meta-llama/Llama-3.1-8B-Instruct
kill_engine(){ ps -eo pid,cmd|grep -E "EngineCore|vllm serve|resource_tracker"|grep -v grep|awk '{print $1}'|xargs -r kill -9 2>/dev/null; sleep 4; }
wait_ready(){ for _ in $(seq 1 90); do grep -q "Application startup complete" "$1" 2>/dev/null&&return 0; sleep 6; done; echo TIMEOUT;return 1;}

# B0 serving arm: FCFS, no LTR ranker, tau=0 (dense). $1=KV arg, $2=OVR, $3=backend extra, $4=name, $5=gpu label
run_b0(){ local KVARG=$1 OVR=$2 BEXTRA=$3 NM=$4 GPU=$5; kill_engine
  PYTHONPATH=serving/v025_blasst VLLM_BLASST_TAU=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid .venv-v025/bin/vllm serve "$MODEL" --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.85 \
    --enforce-eager --no-enable-prefix-caching $KVARG --num-gpu-blocks-override "$OVR" \
    --scheduling-policy fcfs $BEXTRA --port 8000 > "$L/srv_${NM}.log" 2>&1 &
  wait_ready "$L/srv_${NM}.log" || { echo "SERVER FAIL $NM"; return 1; }
  .venv/bin/python -m bench.run_sweep --config "$NM" --base-url http://127.0.0.1:8000 \
    --model "$MODEL" --rates 4,8,16,32,64 -n 200 --warmup 16 --use-reference-len --source lmsys \
    --endpoint chat --out-dir results/summaries --gpu "$GPU" > "$L/sweep_${NM}.log" 2>&1
  echo "done $NM exit=$?"
}
echo "=== B0 + JOINT QUALITY $(date) ==="
run_b0 "" 512 "--attention-backend TRITON_ATTN" b0_triton RTX3090-v025-triton
run_b0 "--kv-cache-dtype turboquant_4bit_nc" 724 "" b0_tq4 RTX3090-v025-turboquant

# Joint quality: GSM8K on TQ4 + BLASST (tau=6) — exercises the real _tq_decode_stage1 decode path
kill_engine
PYTHONPATH=serving/v025_blasst VLLM_BLASST_TAU=6 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  setsid .venv-v025/bin/vllm serve "$MODEL" --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --enforce-eager --no-enable-prefix-caching --kv-cache-dtype turboquant_4bit_nc --num-gpu-blocks-override 724 \
  --scheduling-policy fcfs --port 8000 > "$L/srv_gsm8k_tq4blasst.log" 2>&1 &
wait_ready "$L/srv_gsm8k_tq4blasst.log" || { echo "SERVER FAIL gsm8k_tq4blasst"; kill_engine; echo B0_QUALITY_DONE; exit 0; }
echo "tq_patch=$(grep -c 'installed patched _tq_decode_stage1' "$L/srv_gsm8k_tq4blasst.log")"
.venv/bin/python -m ltr.quant.eval_gsm8k --base-url http://127.0.0.1:8000 --model "$MODEL" \
  --n 100 --dtype-note "tq4+blasst" --out results/summaries/gsm8k_tq4blasst_v025.json > "$L/gsm8k_tq4blasst.log" 2>&1
echo "gsm8k done exit=$?"
kill_engine
echo "B0_QUALITY_DONE"
