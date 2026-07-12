# Reference stack (vllm-ltr) — reproducible build + patches

The parity path (the B1a reference stack): `hao-ai-lab/vllm-ltr` (Fu et al., NeurIPS
2024) — a **vLLM 0.4.1 fork** with the Learning-to-Rank scheduler + **swap
preemption**. This is what can reproduce the paper's ~2.1×, because vLLM v1
removed swap (see `results/RESULTS.md` — the swap-vs-recompute finding).

The fork itself is git-ignored (`ltr/vendor/vllm-ltr/`, it's a full vLLM tree).
This file documents how to rebuild it + the source patches we applied. Verified
building + serving **Llama-3.1-8B on an RTX 3090 (WSL2, no sudo)**.

## 0. Clone
```bash
git clone --depth 1 https://github.com/hao-ai-lab/vllm-ltr ltr/vendor/vllm-ltr
```

## 1. Environment (micromamba — conda without sudo)
The build compiles 21 CUDA kernels → needs a **consistent** CUDA toolkit
matching torch 2.2.1's CUDA (12.1). micromamba's solver drifts newer, so pin
nvcc explicitly.
```bash
# micromamba (standalone, no sudo)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~/.local bin/micromamba

~/.local/bin/micromamba create -y -n vllm-ltr -c pytorch -c nvidia -c conda-forge \
  python=3.10 pytorch==2.2.1 pytorch-cuda=12.1 cuda-version=12.1 cuda-toolkit \
  ninja setuptools wheel packaging

# CRITICAL: the solver pulls nvcc 12.4 while headers/torch are 12.1 -> torch's
# cuda.cmake aborts on the mismatch. Pin nvcc DOWN to 12.1 (+ numpy for torch):
~/.local/bin/micromamba install -y -n vllm-ltr -c nvidia -c conda-forge \
  "numpy=1.26" "cuda-nvcc=12.1.105" "cuda-cudart-dev=12.1.105" "cuda-compiler=12.1"
```

## 2. Build (from source — CPU compile, ~8 min with MAX_JOBS=4)
```bash
ENVP="$HOME/.local/share/mamba/envs/vllm-ltr"
cd ltr/vendor/vllm-ltr
CUDA_HOME="$ENVP" MAX_JOBS=4 ~/.local/bin/micromamba run -n vllm-ltr \
  pip install -e . --no-build-isolation
```

## 3. Fix deps (pip pulls too-new transformers)
vLLM 0.4.1's `transformers>=4.40` lets pip grab transformers 5.x, which needs
torch>=2.4 and disables PyTorch. Pin a version that supports Llama-3.1
(`rope_type: llama3`, added in 4.43) AND works with torch 2.2.1:
```bash
~/.local/bin/micromamba run -n vllm-ltr pip install "transformers==4.44.2" "tokenizers<0.20"
```

## 4. Source patches (for Llama-3.1's `llama3` rope on this old vLLM)
transformers >=4.43 renamed rope_scaling `type` → `rope_type` and added the
`llama3` method; vLLM 0.4.1 only knew `type` ∈ {linear, dynamic, yarn}.

**`vllm/config.py`** (`_get_and_verify_max_len`, ~line 1002): read either key,
and don't multiply the (already-extended) max_len for `llama3`:
```python
rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
if rope_type != "llama3":
    assert "factor" in rope_scaling
    scaling_factor = rope_scaling["factor"]
    if rope_type == "yarn":
        derived_max_model_len = rope_scaling["original_max_position_embeddings"]
    derived_max_model_len *= scaling_factor
```

**`vllm/model_executor/layers/rotary_embedding.py`** (`get_rope`, ~line 360):
read either key; for `llama3` use a plain rope (its frequency-smoothing mainly
affects long context, so for our short-context ≤4096 scheduling benchmark this
is sufficient and keeps output-length labels self-consistent):
```python
scaling_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
if scaling_type == "llama3":
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style)
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
scaling_factor = rope_scaling["factor"]
...
```

### 4b. `benchmarks/benchmark_serving_real.py` — accept `fifo` (swap baseline)
The client asserts on unknown `--schedule-type`. We add `fifo` to the no-op
branch (~line 267) so the client sends requests in arrival order while the
**server** (started with `--schedule-type fifo`) does arrival-order scheduling
with **SWAP** preemption. This gives the decomposition baseline: `fcfs`
(recompute) → `fifo` (swap) isolates the swap contribution; `fifo` → `opt`
(LTR) isolates the ranking contribution.
```python
if schedule_type == "fcfs" or ... or schedule_type == "fifo" or ...:
    pass
```
Also `pip install aiohttp tqdm scipy` into the env for the client.

### 4c. `vllm/attention/backends/xformers.py` — C1 Stage 1a rotation hook
Gated by env `VLLM_KV_ROTATE=1` (off by default). Imports the committed,
tested `ltr.quant.rotation` (needs the capstone repo on `PYTHONPATH`, which
`serving/bench_reference.sh` sets) and, in `XFormersImpl.forward`, rotates
Q/K/V by a fixed orthogonal `R` (per module, seed 0) right after the
`view(..., head_size)` reshape, then inverse-rotates the output before the final
reshape. Dot-product-invariant ⇒ attention unchanged; the (fp8-quantized) KV
cache then stores near-uniform coordinates (TurboQuant idea). `C1 = VLLM_KV_ROTATE=1
+ --kv-cache-dtype fp8`. See docs/C_TIERS.md §3 (Stage 1a). Full TurboQuant scalar
quant (Stage 1b) will add a `"turboquant"` codec header + CUDA dispatch branches.

### 4d. `xformers.py` — C1 Stage 1b/2 fake-quant hook
Gated by `VLLM_KV_BITS=N`: fake-quantizes (quantize→dequantize) the STORED K/V to
N bits (`ltr.quant.scalar_quant.fake_quantize`) so decode reads N-bit KV while
prefill uses raw K/V (matches real fp8). Measures sub-fp8 *accuracy* without CUDA
storage kernels (the memory win is the CUDA productionization; fp8 already gives
real 2×). `VLLM_KV_BITS_SCHEDULE="idx:bits,…"` overrides bits per layer (KVmix
mixed precision, Stage 2) — each `XFormersImpl` gets a construction-order
`_layer_idx`. Composes with `VLLM_KV_ROTATE` (rotate first, then quantize).
`VLLM_KV_NF=1` uses the density-matched NormalFloat quantizer (real TurboQuant
scalar quant) instead of the naive uniform grid. `VLLM_KV_BITS_K`/`VLLM_KV_BITS_V`
set K and V bits separately (TurboQuant's asymmetric `k8v4` = fp8 K + 4-bit V;
"V compression is nearly free" since softmax amplifies K error but not V).

### 4e. `vllm/attention/ops/prefix_prefill.py` — fp8 KV in the prefix kernel (C1)
0.4.1's `context_attention_fwd` Triton kernel (`_fwd_kernel`) does
`tl.dot(q_bf16, k_uint8)` on the cached context and **crashes on fp8 KV**
("First input (bf16) and second input (uint8) must have the same dtype") — hit on
any preemption-resume or chunked-prefill, i.e. exactly the high-load regime C1
needs. Patch (mirrors modern vLLM's prefix fp8 handling; verified Triton 2.2.0
supports the fp8 bitcast on sm_86): add a `USE_FP8_KV: tl.constexpr` to
`_fwd_kernel`, dequant the **cached** K and V after each `tl.load(K_cache…)` /
`tl.load(V_cache…)` with `x = x.to(tl.float8e5, bitcast=True).to(q.dtype)`, and
pass `USE_FP8_KV=(k_cache.dtype == torch.uint8)` at the `_fwd_kernel[grid]`
launch in `context_attention_fwd`. (Only the cached-context loads need it; the
current-query k/v are fresh bf16.) This unblocks real fp8 KV (2× capacity) on the
0.4.1 swap stack — the C1 latency comparison vs B1.

## 5. Verify
```bash
~/.local/bin/micromamba run -n vllm-ltr python -c "
from vllm import LLM, SamplingParams
llm = LLM('meta-llama/Llama-3.1-8B-Instruct', dtype='float16', max_model_len=4096,
          gpu_memory_utilization=0.85, enforce_eager=True, swap_space=4)
print(llm.generate(['Hello'], SamplingParams(max_tokens=8))[0].outputs[0].text)"
```
✅ Serves Llama-3.1-8B with **swap** preemption (`swap_space=4`).

## 6. Next: the LTR benchmark (B0 FCFS vs B1 LTR, with swap)
Use the fork's `benchmarks/` (dataset generation via its `synthesize`, the
allRank/ListMLE ranker in `train/allrank`, and its LTR scheduler) to run
FCFS vs LTR under swap — where the 2.1× lives. On the A100 this is the paper's
exact platform; on the 3090 it's memory-tight (cap the KV pool as in B0/B1).
