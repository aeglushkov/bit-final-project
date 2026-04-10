# VADAR: Visual Agentic AI for Spatial Reasoning with a Dynamic API

`Arxiv 2025` · 🏛️ California Institute of Technology

[📄 Paper](https://arxiv.org/abs/2502.06787) · [🚀 Project](https://glab-caltech.github.io/vadar/)

🏷️ **SUBJECT:** Training-free agentic program synthesis for 3D visual-spatial reasoning.

❓ **PROBLEM:**
- Prior visual program synthesis (ViperGPT, VisProg) relies on a fixed, human-authored DSL that caps the reachable reasoning.
- End-to-end VLMs struggle on compositional 3D queries (distance, size, relative position).
- Training a neuro-symbolic spatial reasoner requires large supervision (e.g., LEFT needs 10K+ samples).

💡 **IDEA:** Let LLM agents **dynamically grow a Pythonic API** — proposing, implementing, and testing new reusable spatial functions on the fly — then synthesize programs against that evolving API.

🛠️ **SOLUTION:**
- **API Generation phase:** a Signature Agent proposes reusable method signatures from a batch of 15 queries; an Implementation Agent writes each one via depth-first dependency resolution; a Test Agent validates and re-tries (up to 5×).
- **Program Synthesis phase:** a Program Agent writes a CoT-planned Python program against the grown API; an Execution Agent runs it line-by-line and feeds errors back for revision.
- **Vision specialists:** Molmo + GroundingDINO, SAM2, UniDepth, and GPT-4o VQA expose `loc`, `depth`, `vqa`, `same_object`, `get_2D_object_size` as the API seed.
- **Prompting scaffolds:** weak-ICL usage hints + pseudo-ICL implementation tips — ablation shows each contributes independently.

🏆 **RESULTS:** On **Omni3D-Bench** (new real-world benchmark), VADAR is within ~2 pts of GPT-4o and ahead of all other VLMs; on **CLEVR**, 53.6% vs. ViperGPT 42.6% and VisProg 39.9%. Oracle-vision runs hit 83% / 94% — the remaining gap is perception, not program logic. Matches or beats LEFT with **zero training data**.
