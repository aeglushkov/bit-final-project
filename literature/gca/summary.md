# Geometrically-Constrained Agent for Spatial Reasoning

`Arxiv 2025` · 🏛️ Beihang University · 🏛️ Shanghai AI Lab · 🏛️ SJTU · 🏛️ ZIP Lab (ZJU)

[📄 Paper](https://arxiv.org/abs/2511.22659) · [💻 Code](https://github.com/Zx55) · [🚀 Project](https://gca-spatial-reasoning.github.io)

🏷️ **SUBJECT:** Agent framework that constrains VLM spatial reasoning with an explicit task formalization before tool use.

❓ **PROBLEM:**
- **Semantic-to-geometric gap:** VLMs reason in a lossy semantic space misaligned with high-fidelity geometry.
- **Oracle paradox:** training-based methods inherit flawed spatial logic from imperfect GPT-4o-generated data.
- **Unconstrained planning:** tool-integrated methods (SpatialAgent, TIGeR) route only the final computation through tools — the VLM still hallucinates "what to solve" in its head.

💡 **IDEA:** Force the VLM to first emit an explicit **formal task constraint** `C_task = (C_R, C_O)` — a reference frame and an objective — then solve it as a ReAct agent strictly governed by that constraint.

🛠️ **SOLUTION:**
- **Task Formalization (Stage 1):** VLM as semantic analyst generates `C_R` (object/camera/direction-based 3D frame) and `C_O` (what to measure) in parallel from the query.
- **Constrained Geometric Computation (Stage 2):** VLM as task solver issues tool calls governed by `C_task` with closed-loop ambiguity resolution.
- **8-tool geometric toolbox:** VGGT reconstruction, open-vocab detection, 6-DoF pose (Orient Anything), metric-scale estimation, OCR, optical flow, Python sandbox.
- **Knowledge-Augmented Code Generation (KACG):** inject verified geometric formulas via RAG so the VLM composes code instead of hallucinating math.

🏆 **RESULTS:** **64.8% average across 5 spatial benchmarks (new SOTA)** — +12% over Gemini-2.5-Pro, +27% over SpatialLadder, +38% over TIGeR; generalizes across GPT-4o, GLM-4.5V, Qwen3-VL, Gemini (+37% avg when the framework is applied).

💭 **THOUGHTS:**
- **Video / temporal extension:** toolbox is image-only today; adding temporal reasoning is the obvious next step and aligns with our own agent-layer direction.
- **Abstract reference entities:** rooms and regions are handled via proxies with cumulative error — is there a principled way to formalize non-rigid reference frames?
- **Formalization ceiling:** the VLM only gets ~70% on formalization alone, so fixing `C_task` quality likely buys more than scaling tools.
