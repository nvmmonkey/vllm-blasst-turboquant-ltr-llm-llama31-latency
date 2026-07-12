#!/usr/bin/env bash
# Fresh, isolated .venv-v025 for vLLM 0.25.0 smoke test — does NOT touch
# .venv / .venv-v1fp8 / the 0.4.1 fork.
cd /home/mking/capstone || exit 2
export UV_HTTP_TIMEOUT=600
echo "=== [1/4] create venv ==="
uv venv .venv-v025 --python 3.10 2>&1 || { echo "VENV_FAIL"; exit 3; }
echo "=== [2/4] install vllm==0.25.0 (pulls matching torch) ==="
uv pip install --python .venv-v025 "vllm==0.25.0" 2>&1 || { echo "VLLM_INSTALL_FAIL"; exit 4; }
echo "=== [3/4] import check ==="
.venv-v025/bin/python -c "import vllm; print('VLLM_VERSION', vllm.__version__)" 2>&1 || { echo "VLLM_IMPORT_FAIL"; exit 5; }
echo "=== [4/4] flashinfer (optional — needed for fp8 KV on Ampere) ==="
uv pip install --python .venv-v025 flashinfer-python 2>&1 && \
  .venv-v025/bin/python -c "import flashinfer; print('FLASHINFER_OK', getattr(flashinfer,'__version__','?'))" 2>&1 || \
  echo "FLASHINFER_OPTIONAL_SKIPPED"
echo "INSTALL_DONE"
