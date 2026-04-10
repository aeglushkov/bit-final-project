# Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation

`CVPR 2025` · 🏛️ Shanghai Jiao Tong University · 🏛️ Coop. Medianet Innovation Center

[📄 Paper](https://arxiv.org/abs/2412.01694) · [💻 Code](https://github.com/zhengrongz/AoTD) · [🚀 Project](https://zhengrongz.github.io/AoTD/) · [📊 Model](https://huggingface.co/Zhengrongzz/AoTD-7B)

🏷️ **SUBJECT:** Instruction-tuning Video-LLMs with distilled agent reasoning traces for VideoQA.

❓ **PROBLEM:**
- Video-LLMs trained on raw (Q, A) pairs lack explainability and struggle with spatial-temporal grounding.
- Agent-based VideoQA systems offer interpretable step-by-step reasoning but are too slow and memory-heavy for practical use.
- No established way to transfer the reasoning skill of an agent pipeline back into a single fast model.

💡 **IDEA:** Distill multi-step reasoning traces from a slow **agent-of-thoughts** pipeline (specialist vision tools orchestrated by an LLM) into a single Video-LLM via Chain-of-Thought instruction tuning.

🛠️ **SOLUTION:**
- **Specialist Selection:** benchmark off-the-shelf models on atomic sub-tasks (detection, temporal grounding, QA) and pick the best per slot.
- **Program Generation:** DeepSeek-Coder decomposes each VideoQA question into a Python program that invokes the specialists and records an execution trace.
- **CoT Conversion & Filtering:** LLaMA-3.1-8B rewrites traces into natural-language CoTs; traces with wrong answers or incoherent reasoning are dropped.
- **Distillation:** fine-tune LLaVA-NeXT-Video 7B with joint loss `L = L_label + λ·L_rationale` on QA + verified CoTs.

🏆 **RESULTS:** AoTD-tuned LLaVA-NeXT-Video beats baselines on STAR (74.3), NExT-QA (77.6), MVBench (55.6) and VSIBench (28.8), while running ~5× faster and using ~3.5× less memory than the agent pipeline it was distilled from.
