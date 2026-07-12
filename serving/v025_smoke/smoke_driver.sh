#!/usr/bin/env bash
# Run each smoke config in its own process; grep the log for the decisive evidence.
cd /home/mking/capstone || exit 2
PY=.venv-v025/bin/python
S=serving/v025_smoke
D=$S/smoke_logs; mkdir -p "$D"
export VLLM_LOGGING_LEVEL=INFO HF_HUB_OFFLINE=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1     # WSL2: makes pin memory (=> UVA) available for the V1 worker
export VLLM_USE_FLASHINFER_SAMPLER=0     # avoid sampler JIT-compile issues on this box

run() { # name  args...
  local nm=$1; shift
  echo "######## CONFIG $nm ########"
  timeout 420 $PY "$S/smoke_one.py" "$@" > "$D/$nm.log" 2>&1
  echo "exit=$?"
  grep -iE "Initializing a V[01] LLM engine|Using .* backend|Automatically detected|GPU KV cache|# GPU blocks|num_gpu_blocks|available_kv_cache|RESULT |Cannot use|not supported|requires SM|raise|Error" "$D/$nm.log" \
    | grep -viE "logger|traceback most recent|self\.|^\s+File \"" | tail -8
  echo
}

run g0_auto        auto
run g1_priority    auto priority
run g2_fp8         fp8
run g3_tq4bit      turboquant_4bit_nc
run g4_tqk8v4      turboquant_k8v4
echo "SMOKE_DONE"
