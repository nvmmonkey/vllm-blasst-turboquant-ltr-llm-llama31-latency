#!/usr/bin/env bash
# v0.25 long-context probe — aligns with the 0.4.1/0.8.5 lc runs (fixed rate 6,
# pool cap, sweep ctx 512-7168) AND adds the C1 angle: bf16 vs TurboQuant-4bit at
# equal memory (does TQ4 capacity help more at long context?).
#   bf16 arm: OVR=1024 (16,384 tok) ; TQ4 arm: OVR=1446 (memory-equal, 2.83x tokens)
cd /home/mking/capstone || exit 2
L="${L:-/tmp/kvbench}"
CTXS="512 2048 4096 7168"

kill_engine() { ps -eo pid,cmd | grep -E "EngineCore|vllm serve|resource_tracker" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null; sleep 4; }

wait_ready() {  # $1 = server logfile
  for _ in $(seq 1 50); do
    grep -q "Application startup complete" "$1" 2>/dev/null && return 0
    grep -qE "ValueError|Free memory|Engine core init.*failed" "$1" 2>/dev/null && { echo "SERVER_FAIL $1"; return 1; }
    sleep 6
  done
  echo "READY_TIMEOUT $1"; return 1
}

run_arm() {  # $1=KV  $2=OVR  $3=prefix
  local KV=$1 OVR=$2 PFX=$3
  kill_engine
  KV="$KV" POL=fcfs OVR="$OVR" MAXLEN=8192 PORT=8000 setsid bash serving/v025_smoke/serve_v025.sh > "$L/srv_${PFX}.log" 2>&1 &
  wait_ready "$L/srv_${PFX}.log" || return 1
  for P in $CTXS; do
    LONGCTX_TOKENS="$P" .venv/bin/python -m bench.run_sweep --config "${PFX}_${P}" --base-url http://127.0.0.1:8000 \
      --model meta-llama/Llama-3.1-8B-Instruct --rates 6 -n 40 --warmup 4 --ignore-eos --max-tokens 96 \
      --source longctx --endpoint completions --out-dir results/summaries --gpu RTX3090-v025 > "$L/${PFX}_${P}.log" 2>&1
    echo "done ${PFX}_${P} exit=$?"
  done
}

echo "=== LONGCTX v0.25 $(date) ==="
run_arm auto              1024 lcv025
run_arm turboquant_4bit_nc 1446 lcv025tq4
kill_engine
echo "LONGCTX_V025_DONE"
