# References (IEEE)

Prior work we build on and the techniques the KV levers (C1, C2) draw from.

**Prior work we build on**
- [11] A. Saravana Kumar, V. Janarthanan, S. Sharma, and K. Palani, "An empirical
  study on latency reduction techniques for large language models," Olsen Coll.
  Eng. Sci., Fairleigh Dickinson Univ., Vancouver, BC, Canada, 2026. — the prior
  FDU paper; B1 reproduces its LTR result.
- [12] Y. Fu, S. Zhu, R. Su, A. Qiao, I. Stoica, and H. Zhang, "Efficient LLM
  scheduling by learning to rank," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*,
  vol. 37, 2024, pp. 59006–59029. — code: `github.com/hao-ai-lab/vllm-ltr`.

**KV-cache techniques (later C-tiers)**
- [1] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni, "TurboQuant: Online
  vector quantization with near-optimal distortion rate," arXiv:2504.19874, 2025.
- [2] F. Li, S. Liu, W. Wu, S. Nie, and J. Wang, "KVmix: Gradient-based layer
  importance-aware mixed-precision quantization for KV cache," in *AAAI*, 2026.
- [3] Q. Liu, Z. Hong, P. Li, F. Chen, and S. Guo, "MELL: Memory-efficient large
  language model serving via multi-GPU KV cache management," arXiv:2501.06709, 2025.
- [4] J. Yuan et al., "BLASST: Dynamic blocked attention sparsity via softmax
  thresholding," in *Proc. MLSys*, 2026.
- [5] C. Luo et al., "HeadInfer: Memory-efficient LLM inference by head-wise
  offloading," arXiv:2502.12574, 2025.

**Systems / serving**
- [6] Y. Zhong et al., "DistServe: Disaggregating prefill and decoding for
  goodput-optimized large language model serving," in *Proc. OSDI*, 2024.
- [7] W. Kwon et al., "Efficient memory management for large language model
  serving with PagedAttention," in *Proc. SOSP*, 2023. — vLLM.
- [8] Y. Li, F. Wei, C. Zhang, and H. Zhang, "EAGLE-3: Scaling up inference
  acceleration of large language models via training-time test," arXiv:2503.01840, 2025.
- [9] DeepSeek-AI, "DeepSeek-V3 technical report," arXiv:2412.19437, 2024.
- [10] Y. Liu et al., "LMCache: An efficient KV cache layer for enterprise-scale
  LLM inference," arXiv:2510.09665, 2025. — code: `github.com/LMCache/LMCache`.

**Data & tooling**
- LMSYS-Chat-1M: `huggingface.co/datasets/lmsys/lmsys-chat-1m`
- vLLM serving benchmark reference: `benchmarks/benchmark_serving.py` (studied,
  then reimplemented in `bench/` so we control metrics + preemption/KV scraping).
