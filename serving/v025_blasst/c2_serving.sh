#!/usr/bin/env bash
cd /home/mking/capstone || exit 2
L="${L:-/tmp/kvbench}"
MODEL=meta-llama/Llama-3.1-8B-Instruct
kill_engine(){ ps -eo pid,cmd|grep -E "EngineCore|vllm serve|resource_tracker"|grep -v grep|awk '{print $1}'|xargs -r kill -9 2>/dev/null; sleep 4; }
# vLLM prints benign WARNING tracebacks at startup (deep_gemm/cutedsl/flashinfer
# optional deps on WSL2) — grep-for-failure is unreliable. Use ONLY the definitive
# ready signal + timeout. A truly-crashed server never prints it -> TIMEOUT.
wait_ready(){ for _ in $(seq 1 90); do grep -q "Application startup complete" "$1" 2>/dev/null&&return 0; sleep 6; done; echo TIMEOUT;return 1;}
run_arm(){ local TAU=$1 NM=$2; kill_engine
  PYTHONPATH=serving/v025_blasst VLLM_BLASST_TAU=$TAU VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid .venv-v025/bin/vllm serve "$MODEL" --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.85 \
    --enforce-eager --no-enable-prefix-caching --num-gpu-blocks-override 512 --scheduling-policy priority \
    --attention-backend TRITON_ATTN --port 8000 > "$L/srv_${NM}.log" 2>&1 &
  wait_ready "$L/srv_${NM}.log" || { echo "SERVER FAIL $NM"; return 1; }
  grep -c "BLASST\] installed" "$L/srv_${NM}.log"
  .venv/bin/python -m bench.run_sweep --config "$NM" --base-url http://127.0.0.1:8000 \
    --model "$MODEL" --rates 4,8,16,32,64 -n 200 --warmup 16 --use-reference-len --source lmsys \
    --endpoint chat --ltr-ranker results/ranker/opt125m-ltr --out-dir results/summaries --gpu RTX3090-v025-triton > "$L/sweep_${NM}.log" 2>&1
  echo "done $NM exit=$?"
}
echo "=== C2 SERVING $(date) ==="
run_arm 0 b1_triton
run_arm 6 c2_bf16
kill_engine
echo "C2_SERVING_DONE"
