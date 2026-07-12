# LTR baseline (B1)

B1 reproduces the Learning-to-Rank scheduler of the prior work [11][12] to
re-establish the **~2.1× latency reduction over FCFS** at high load. It is the
system every later C-tier is measured against — we improve it, we don't replace
it.

## Idea (from the paper)
A fine-tuned **OPT-125M** predicts each request's **output-length rank**; the
scheduler runs shorter-predicted-output requests first (SJF-like), so fewer
requests are preempted when the KV cache fills. The ranker is trained with
**ListMLE** on LMSYS-Chat-1M output lengths.

## Components (this directory)
| File | Role |
|---|---|
| `ranker/losses.py` | ListMLE listwise ranking loss (training objective) |
| `ranker/ranking_metrics.py` | Kendall's τ, pairwise accuracy (eval) |
| `ranker/model.py` | OPT-125M backbone + scalar scoring head |
| `ranker/dataset.py` | LMSYS → (prompt, output_length) training lists |
| `ranker/train.py` | fine-tune the ranker (ListMLE), deterministic |
| `ranker/eval.py` | rank quality (τ) on a held-out split |
| `scheduler/priority.py` | ranker scores → vLLM request priorities |
| `../serving/serve_b1_ltr.sh` | launch the LTR-scheduled server |

## Two-step reproduction

**B1a — reproduce on the reference stack (get the number).**
Vendor `hao-ai-lab/vllm-ltr` (Fu et al., NeurIPS 2024) under `ltr/vendor/`
(pin the commit), obtain/train the OPT-125M ranker on LMSYS output lengths, and
run the same `bench.run_sweep --config b1`. Target: ~2.1× vs B0 at high load.
The reference stack pins an **old vLLM**, so B1a runs in its **own venv** —
it will not share this repo's modern-vLLM environment.

**B1b — port to vLLM v1 (crash-safe).**
The prior attempt crashed by patching `scheduler.py` + `block_manager_v1.py`
(paper §IV-D). We avoid that entirely: modern vLLM v1 exposes
**`--scheduling-policy priority`**, so we drive LTR ordering by attaching a
per-request `priority` (from `scheduler/priority.py`) — **never touching block
allocation**. `serve_b1_ltr.sh` is exactly B0 + `--scheduling-policy priority`.

```
        requests ──► OPT-125M ranker ──► scores ──► priorities ──► vLLM (priority policy)
                     (client side, in the harness)                 (no engine patching)
```

## Honest boundaries / v1 migration notes
- **Preemption mode:** the paper used `--preemption-mode swap` (better than
  recompute on vLLM 0.4.1). vLLM v1 **removed** that flag and is
  **recompute-only**, so a v1 B1 cannot use swap — this is a documented
  difference, not a regression we introduced.
- **Ranker:** OPT-125M + ListMLE is the prior work's; we keep it deterministic
  (fixed seed, no online learning on the hot path) to avoid the overfitting the
  paper reports on held-out prompts.
- **What we claim later** is the KV-cache layer *beneath* this scheduler, not a
  better scheduler.

## Run it
```bash
# 1. train the ranker (needs the LMSYS dataset + a GPU)
# LLM.pdf spec: 23,800 samples (expanded from 10k), 10 epochs, OPT-125M ranker
python -m ltr.ranker.train --out results/ranker/opt125m-ltr --n 23800 --epochs 10

# 2. evaluate ranking quality
python -m ltr.ranker.eval --ranker results/ranker/opt125m-ltr

# 3. serve B1 and sweep (same harness as B0)
serving/serve_b1_ltr.sh                 # --scheduling-policy priority
python -m bench.run_sweep --config b1 --model meta-llama/Llama-3.1-8B-Instruct ...
```
