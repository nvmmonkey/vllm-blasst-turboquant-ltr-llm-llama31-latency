#!/usr/bin/env python3
# One config per process (isolates GPU mem + failures). V1 engine spawns the
# EngineCore, so the entry point MUST be guarded by if __name__ == '__main__'.
import sys


def main():
    dtype = sys.argv[1]                               # auto | fp8 | turboquant_4bit_nc | turboquant_k8v4
    test_priority = len(sys.argv) > 2 and sys.argv[2] == "priority"
    from vllm import LLM, SamplingParams

    kw = dict(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=2048,
              gpu_memory_utilization=0.85, enforce_eager=True)
    if dtype != "auto":
        kw["kv_cache_dtype"] = dtype
    if test_priority:
        kw["scheduling_policy"] = "priority"

    tag = f"{dtype}{'+prio' if test_priority else ''}"
    try:
        llm = LLM(**kw)
    except Exception as e:
        print(f"RESULT INIT_FAIL {tag} :: {repr(e)[:500]}")
        return

    try:
        kvb = llm.llm_engine.cache_config.num_gpu_blocks
        print(f"RESULT KVBLOCKS {tag} :: {kvb}")
    except Exception:
        pass

    sp = SamplingParams(temperature=0, max_tokens=24)
    try:
        if test_priority:
            out = llm.generate(["Count to five:", "Name one color:"], sp, priority=[1, 0])
        else:
            out = llm.generate(["The capital of France is", "2+2="], sp)
        txt = [o.outputs[0].text.strip().replace(chr(10), " ")[:50] for o in out]
        ok = all(len(t) > 0 for t in txt)
        print(f"RESULT GEN_{'OK' if ok else 'EMPTY'} {tag} :: {txt}")
    except Exception as e:
        print(f"RESULT GEN_FAIL {tag} :: {repr(e)[:500]}")


if __name__ == "__main__":
    main()
