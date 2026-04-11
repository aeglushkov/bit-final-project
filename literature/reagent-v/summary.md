# ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding

`NeurIPS 2025` · 🏛️ UNC-Chapel Hill · 🏛️ University of Washington

[📄 Paper](https://arxiv.org/abs/2506.01300) · [💻 Code](https://github.com/aiming-lab/ReAgent-V)

🏷️ **SUBJECT:** Agentic video-QA framework that couples entropy-guided frame selection, tool-augmented inference, and multi-perspective reflective refinement driven by inference-time reward signals.

❓ **PROBLEM:**
- Single-pass LVLM video QA has no mechanism to self-correct or integrate dynamic feedback.
- Offline reward models / template rewards cannot capture real-time reasoning state during inference.
- Prior multi-agent / tool-agent frameworks are slow, lack reward signals, and overprocess frames.

💡 **IDEA:** Bolt a reward-generating critic and a **multi-perspective reflection** loop (conservative / neutral / aggressive) onto tool-augmented VLM inference, so the same reward trace both refines the current answer and curates high-value samples for SFT / DPO / GRPO.

🛠️ **SOLUTION:**
- **Entropy-Calibrated Relevance Scoring (ECRS):** jointly scores frames by CLIP query-similarity × per-channel RGB histogram entropy; iterative threshold `k·α^m·τ` keeps only frames that are both relevant and information-rich.
- **Tool-augmented reasoning:** target agent selects a subset of a tool factory (OCR, ASR, Grounding-DINO, scene graph, CLIP, SharedGPT4Video, caption model) per query to build the input context.
- **Critic agent:** rejects unsatisfactory answers, generates critic sub-questions, re-invokes tools, and emits an **evaluation report** with scalar reward + five scored dimensions (visual alignment, temporal accuracy, option disambiguation, reasoning specificity, linguistic precision).
- **Multi-perspective reflection:** three persona prompts (conservative / neutral / aggressive) regenerate answers with confidence scores; if `min` confidence > 0.6 a meta-agent fuses them, otherwise the highest-confidence revision wins.
- **Reward-aware data curation:** the same critic report drives SFT filtering, DPO preference pairs, and GRPO sample selection (keep samples with importance score < 5/10 as "reflection-worthy").

🏆 **RESULTS:** Up to **+6.9%** on video understanding (LongBench / NextQA / EgoSchema / LVBench / MLVU / VideoMME with LLaVA-Video-72B and Qwen2.5-VL-72B), **+2.1%** on video reasoning (VSI-Bench / VideoMMMU / MMVU / MVBench / TempCompass / VideoMME via GRPO data curation on Qwen2.5-VL-7B, using only 52k vs. 260k samples), and **+9.8%** on VLA alignment (OpenVLA + TPO on SIMPLER) over GRAPE.

💭 **THOUGHTS:**
- VSI-Bench is evaluated but the framework contains **no spatial primitives** — spatial tools are scene-graph text tokens, no ego-allo transforms, no geometry, no cognitive maps. The gain on VSI-Bench (27.7 → 33.1) comes from data curation, not a spatial reasoning mechanism.
- The five-dimension critic rubric is a natural hook for adding a spatial-reasoning axis (ego-allo consistency, metric plausibility).
- Reflection runs on every sample: no early-exit, and Table 4 shows the aggressive persona alone actually *reduces* accuracy — the ensemble is doing real work.
