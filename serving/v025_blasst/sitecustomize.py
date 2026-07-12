# On PYTHONPATH, runs at interpreter startup in every process (incl. spawned
# vLLM EngineCore workers). Applies the BLASST monkeypatch when VLLM_BLASST_TAU>0.
try:
    import blasst_patch
    blasst_patch.apply()
except Exception as _e:
    import sys
    print(f"[BLASST] sitecustomize skipped: {_e}", file=sys.stderr, flush=True)
