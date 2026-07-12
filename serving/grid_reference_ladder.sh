#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reference-stack (ShareGPT / paper-parity) FULL rate ladder — makes the
# Nlatency result consistent with the LMSYS grid's rate axis (4..64).
#
# WHY the original Nlatency table was only 2/4/8: the 3090 with the paper's
# long ShareGPT trace (mean 416, tail 6334 tokens) SATURATES at rate ~4-8 (a
# small-KV-pool effect; the paper's 30-60 is the A100's saturation range). Also
# bench_reference.sh sizes num_prompts = request_time * rate, so rate 64 * 60s =
# 3840 long requests — impractical. Here we FIX num_prompts ~= 240 (REQTIME =
# 240/rate) so every rate is tractable, and add the C1 (fp8) arm across all rates.
#
# ARMS (paper's own workload + metric, Nlatency = e2e / output_len):
#   b0  = fcfs    (FCFS, recompute)         KVDTYPE=auto
#   b1a = opt-xxx (LTR order + swap)         KVDTYPE=auto  + predictor
#   c1  = opt-xxx (LTR + swap + fp8 KV)      KVDTYPE=fp8   + predictor
# HONEST: rates >8 are DEEP OVERLOAD on the 3090 (past its ShareGPT saturation);
# they complete the curve but the paper's true 30-60 regime needs the A100.
# ---------------------------------------------------------------------------
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$REPO/ltr/vendor/vllm-ltr/benchmarks"
PRED="MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json"
RATES="${RATES:-2 4 8 16 32 60}"  # cap at 60 = the paper's max (5-60 req/s); 60 is deep
NP="${NUMPROMPTS:-200}"            # FIXED prompt count per rate — the fork client computes
                                   # num=request_time*rate when num-prompts=-1, which blows up
                                   # to 3840 @ r64; a fixed count keeps every rate tractable.
LADDER="$BENCH/RESULTS_ladder"; mkdir -p "$LADDER"

echo "=== REFERENCE LADDER (ShareGPT, Nlatency, $NP prompts/rate) $(date) ==="
for rate in $RATES; do
  echo "--- rate $rate ($NP prompts) ---"
  # B0: fcfs / recompute / fp16
  SCHED=fcfs RATE=$rate NUMPROMPTS=$NP KVDTYPE=auto RESDIR="$LADDER/b0" \
    SRVLOG="$LADDER/b0/srv_r${rate}.log" CLILOG="$LADDER/b0/cli_r${rate}.log" \
    bash "$REPO/serving/bench_reference.sh" >/dev/null 2>&1; echo "  b0  r$rate rc=$?"
  # B1a: opt / LTR+swap / fp16
  SCHED=opt-xxx PREDICTOR="$PRED" RATE=$rate NUMPROMPTS=$NP KVDTYPE=auto RESDIR="$LADDER/b1a" \
    SRVLOG="$LADDER/b1a/srv_r${rate}.log" CLILOG="$LADDER/b1a/cli_r${rate}.log" \
    bash "$REPO/serving/bench_reference.sh" >/dev/null 2>&1; echo "  b1a r$rate rc=$?"
  # C1: opt / LTR+swap / fp8 KV
  SCHED=opt-xxx PREDICTOR="$PRED" RATE=$rate NUMPROMPTS=$NP KVDTYPE=fp8 RESDIR="$LADDER/c1" \
    SRVLOG="$LADDER/c1/srv_r${rate}.log" CLILOG="$LADDER/c1/cli_r${rate}.log" \
    bash "$REPO/serving/bench_reference.sh" >/dev/null 2>&1; echo "  c1  r$rate rc=$?"
done
echo "=== LADDER DONE $(date) — parse Nlatency from $LADDER/{b0,b1a,c1}/*.json ==="
