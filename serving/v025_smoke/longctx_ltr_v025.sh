#!/usr/bin/env bash
# Missing LTR arm for the v0.25 long-context alignment: bf16 + priority + LTR ranker,
# rate 6, sweep ctx 512-7168 (matches lcv025_* fcfs and lcv025tq4_*).
cd /home/mking/capstone || exit 2
L="${L:-/tmp/kvbench}"
kill_engine() { ps -eo pid,cmd | grep -E "EngineCore|vllm serve|resource_tracker" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null; sleep 4; }
wait_ready() { for _ in $(seq 1 55); do grep -q "Application startup complete" "$1" 2>/dev/null && return 0; grep -qE "ValueError|Free memory|Engine core init.*failed" "$1" 2>/dev/null && { echo FAIL; return 1; }; sleep 6; done; echo TIMEOUT; return 1; }
kill_engine
KV=auto POL=priority OVR=1024 MAXLEN=8192 PORT=8000 setsid bash serving/v025_smoke/serve_v025.sh > "$L/srv_lcv025ltr.log" 2>&1 &
wait_ready "$L/srv_lcv025ltr.log" || exit 3
for P in 512 2048 4096 7168; do
  LONGCTX_TOKENS="$P" .venv/bin/python -m bench.run_sweep --config "lcv025ltr_${P}" --base-url http://127.0.0.1:8000 \
    --model meta-llama/Llama-3.1-8B-Instruct --rates 6 -n 40 --warmup 4 --ignore-eos --max-tokens 96 \
    --source longctx --endpoint completions --ltr-ranker results/ranker/opt125m-ltr \
    --out-dir results/summaries --gpu RTX3090-v025 > "$L/lcv025ltr_${P}.log" 2>&1
  echo "done lcv025ltr_${P} exit=$?"
done
kill_engine
echo "LONGCTX_LTR_DONE"
