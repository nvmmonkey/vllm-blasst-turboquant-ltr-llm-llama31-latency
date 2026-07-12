"""Offline verify the BLASST-patched TurboQuant decode kernel (_tq_decode_stage1) actually
COMPILES and produces coherent output under TQ4 KV at tau=6 (C1+C2). sitecustomize on
PYTHONPATH installs both patched kernels; kv_cache_dtype=turboquant_4bit_nc routes decode
through _tq_decode_stage1, so this exercises the C1+C2 path end to end offline."""
import os
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    tau = os.getenv("VLLM_BLASST_TAU", "?")
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        kv_cache_dtype="turboquant_4bit_nc",
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    prompts = [
        "The capital of France is",
        "List three primary colors:",
        "Q: What is 2+2? A:",
    ]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=24))
    print(f"\n===== BLASST tau={tau} + TQ4 (C1+C2) generation =====")
    for p, o in zip(prompts, outs):
        print(f"[{p!r}] -> {o.outputs[0].text!r}")
    print("===== VERIFY DONE =====")
