# TRACE: Unleashing Spatial Reasoning in MLLMs via Textual Representation Guided Reasoning

`Arxiv 2026` · 🏛️ Tsinghua University · 🏛️ Shanghai AI Lab · 🏛️ University of Tokyo

[📄 Paper](https://arxiv.org/abs/2603.23404)

🏷️ **SUBJECT:** Prompting method that builds an allocentric text scaffold for spatial QA in MLLMs.

❓ **PROBLEM:**
- MLLMs reason egocentrically from raw frames and lose the global 3D layout needed for spatial QA.
- Existing prompting baselines (CoT, ToT, LtM, Cognitive Map) give small or inconsistent gains on spatial tasks.
- Fine-tuning or bolting on geometric modules is expensive and does not transfer across backbones.

💡 **IDEA:** Have the MLLM first generate a **Textual Representation of Allocentric Context from Egocentric video (TRACE)** — room topology, coordinate frame, camera trajectory, and an entity registry — and answer the question in the *same* forward pass.

🛠️ **SOLUTION:**
- **Meta-context block:** room topology and a global coordinate system the model commits to before listing entities.
- **Camera trajectory log:** per-frame ego pose so the model grounds observations allocentrically.
- **Entity registry:** object list with estimated 3D positions and pairwise spatial relations.
- **One-stage inference:** the scaffold and the answer are produced in a single pass (two-stage variants are worse — the generation *process* itself is the reasoning).

🏆 **RESULTS:** TRACE consistently beats CoT, ToT, LtM, and Cognitive Map prompting on **VSI-Bench** and **OST-Bench** across Gemini 3 Pro, Qwen2.5-VL-72B, and MiMo-VL-7B — gains hold across backbone scale.

💭 **THOUGHTS:**
- **One-stage > two-stage is surprising:** suggests that "externalizing" spatial reasoning only helps when the representation is entangled with the answer generation — a caveat for our own agent-layer design.
- **No perception module needed:** TRACE is pure prompting, so it's a strong zero-training baseline that any agent framework must clearly beat to justify the complexity.
