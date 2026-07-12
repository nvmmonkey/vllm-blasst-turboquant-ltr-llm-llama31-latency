#!/usr/bin/env bash
# Clean long-context C2: BOTH dense TQ4 and TQ4+BLASST with --no-enable-chunked-prefill
# (the chunked-prefill + long-decode mixed batch is what crashed the BLASST kernel; disabling
# it avoids the crash). Same config both arms => pure C2 delta at long context.
# rate 6, sweep ctx 512-7168, OVR 1446, MAXLEN 8192, fcfs, n=40, max-tokens 96.
cd /home/mking/capstone || exit 2
L="${L:-/tmp/kvbench}"
MODEL=meta-llama/Llama-3.1-8B-Instruct
CTXS="512 2048 4096 7168"
kill_engine(){ ps -eo pid,cmd|grep -E "EngineCore|vllm serve|resource_tracker"|grep -v grep|awk '{print $1}'|xargs -r kill -9 2>/dev/null; sleep 4; }
wait_ready(){ for _ in $(seq 1 90); do grep -q "Application startup complete" "$1" 2>/dev/null&&return 0; sleep 6; done; echo TIMEOUT;return 1;}
run_arm(){ local TAU=$1 NM=$2; kill_engine
  PYTHONPATH=serving/v025_blasst VLLM_BLASST_TAU=$TAU VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid .venv-v025/bin/vllm serve "$MODEL" --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill --kv-cache-dtype turboquant_4bit_nc \
    --num-gpu-blocks-override 1446 --scheduling-policy fcfs --port 8000 > "$L/srv_${NM}.log" 2>&1 &
  wait_ready "$L/srv_${NM}.log" || { echo "SERVER FAIL $NM"; return 1; }
  for P in $CTXS; do
    LONGCTX_TOKENS="$P" .venv/bin/python -m bench.run_sweep --config "${NM}_${P}" --base-url http://127.0.0.1:8000 \
      --model "$MODEL" --rates 6 -n 40 --warmup 4 --ignore-eos --max-tokens 96 --source longctx --endpoint completions \
      --out-dir results/summaries --gpu RTX3090-v025-turboquant > "$L/${NM}_${P}.log" 2>&1
    echo "done ${NM}_${P} exit=$?"
  done
}
echo "=== LONGCTX C2 CLEAN (no-chunked-prefill) $(date) ==="
run_arm 0 lctq4nc       # dense TQ4, no-chunk (baseline)
run_arm 6 lctq4blastnc  # TQ4 + BLASST, no-chunk
kill_engine
echo "LONGCTX_NC_DONE"
