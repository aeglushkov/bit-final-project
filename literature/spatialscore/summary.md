# SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence

`Arxiv 2025` · 🏛️ Shanghai Jiao Tong University · 🏛️ Shanghai AI Laboratory

[📄 Paper](https://arxiv.org/abs/2505.17012) · [💻 Code](https://github.com/haoningwu3639/SpatialScore/) · [📊 Dataset](https://huggingface.co/datasets/haoningwu/SpatialScore) · [🚀 Project](https://haoningwu3639.github.io/SpatialScore/)

🏷️ **SUBJECT:** Holistic benchmark, training corpus, and agent system for evaluating and improving spatial intelligence in MLLMs.

❓ **PROBLEM:**
- Existing spatial benchmarks cover narrow slices (single modality, single task type) and under-report where MLLMs actually fail.
- No training corpus targets the specific sub-skills that spatial reasoning requires.
- No standard tool-augmented baseline exists to measure how far agent wrappers can close the gap without retraining.

💡 **IDEA:** Ship all three artifacts together — a broad **SpatialScore** benchmark, a matching **SpatialCorpus** for SFT, and a plug-in **SpatialAgent** — so data-centric and agent-centric improvements can be compared head-to-head.

🛠️ **SOLUTION:**
- **SpatialScore benchmark:** 5K samples · 30 tasks · 10 categories spanning real-world, simulated, and AIGC imagery across single-image, multi-frame, and video inputs.
- **SpatialCorpus:** 331K supervised QA pairs for spatial SFT, covering perception, relations, and measurement.
- **SpatialAgent:** multi-agent system with 12 specialized perception tools, driven by Plan-Execute and ReAct paradigms, zero training required.
- **Broad evaluation:** 40 representative MLLMs benchmarked to expose systematic failure modes.

🏆 **RESULTS:** Best MLLM still trails human performance (86.60) by **26.48 points** on SpatialScore; SFT on SpatialCorpus gives Qwen3-VL-4B a **+10.47** gain, and SpatialAgent improves reasoning without any training.
