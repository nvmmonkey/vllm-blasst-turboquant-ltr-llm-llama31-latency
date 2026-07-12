#!/usr/bin/env bash
# Round 3: isolate C2 on the TQ4 (TurboQuant per-head decode) path.
#   arm c1_tq4   = LTR+C1     (dense TurboQuant, tau=0 -> BLASST off, patch not installed)
#   arm c1c2_tq4 = LTR+C1+C2  (TurboQuant + BLASST, tau=6 -> _tq_decode_stage1 patched)
# Both arms: same TQ4 KV (kv-cache-dtype turboquant_4bit_nc), same OVR=724 (memory-equal
# with bf16 512blk), priority + LTR ranker. Only difference is BLASST tau => isolates C2.
cd /home/mking/capstone || exit 2
L="${L:-/tmp/kvbench}"
MODEL=meta-llama/Llama-3.1-8B-Instruct
kill_engine(){ ps -eo pid,cmd|grep -E "EngineCore|vllm serve|resource_tracker"|grep -v grep|awk '{print $1}'|xargs -r kill -9 2>/dev/null; sleep 4; }
# vLLM prints benign WARNING tracebacks at startup (deep_gemm/cutedsl on WSL2) — only the
# definitive ready signal + timeout are reliable. A crashed server never prints it -> TIMEOUT.
wait_ready(){ for _ in $(seq 1 90); do grep -q "Application startup complete" "$1" 2>/dev/null&&return 0; sleep 6; done; echo TIMEOUT;return 1;}
run_arm(){ local TAU=$1 NM=$2; kill_engine
  PYTHONPATH=serving/v025_blasst VLLM_BLASST_TAU=$TAU VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid .venv-v025/bin/vllm serve "$MODEL" --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.85 \
    --enforce-eager --no-enable-prefix-caching --kv-cache-dtype turboquant_4bit_nc --num-gpu-blocks-override 724 \
    --scheduling-policy priority --port 8000 > "$L/srv_${NM}.log" 2>&1 &
  wait_ready "$L/srv_${NM}.log" || { echo "SERVER FAIL $NM"; return 1; }
  echo "tq_patch_installed=$(grep -c 'installed patched _tq_decode_stage1' "$L/srv_${NM}.log")"
  .venv/bin/python -m bench.run_sweep --config "$NM" --base-url http://127.0.0.1:8000 \
    --model "$MODEL" --rates 4,8,16,32,64 -n 200 --warmup 16 --use-reference-len --source lmsys \
    --endpoint chat --ltr-ranker results/ranker/opt125m-ltr --out-dir results/summaries --gpu RTX3090-v025-turboquant > "$L/sweep_${NM}.log" 2>&1
  echo "done $NM exit=$?"
}
echo "=== C1+C2 SERVING $(date) ==="
run_arm 0 c1_tq4      # LTR+C1 (dense TurboQuant baseline)
run_arm 6 c1c2_tq4    # LTR+C1+C2 (TurboQuant + BLASST)
kill_engine
echo "C1C2_SERVING_DONE"
