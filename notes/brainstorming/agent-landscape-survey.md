# Agent-Based Approaches for Video & Spatial Understanding: Landscape Overview

## 1. Foundational Agent Architectures

General-purpose LLM agent patterns that underpin most applied work:

| Architecture | Core idea |
|---|---|
| **ReAct** | Interleaved reasoning traces + actions; the model thinks, acts, observes in a loop ([arxiv:2210.03629](https://arxiv.org/abs/2210.03629)) |
| **Reflexion** | Single-agent self-reflection via linguistic feedback after task attempts; uses persistent memory of past failures ([arxiv:2303.11366](https://arxiv.org/abs/2303.11366)) |
| **Plan-and-Execute** | Separate planner (generates full plan upfront) + executor (carries out steps); re-plans on failure |
| **Tree-of-Thoughts** | Tree-structured deliberation with branching and backtracking for complex reasoning ([arxiv:2305.10601](https://arxiv.org/abs/2305.10601)) |

Survey of the full landscape: [The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling](https://arxiv.org/abs/2404.11584)

---

## 2. Visual Programming / Modular Approaches (perception-reasoning split via code generation)

These pioneered the idea of using LLMs as planners that call vision tools:

- **VisProg** (CVPR 2023 Best Paper) — LLM generates a program calling vision modules (object detectors, depth estimators, VQA models). Training-free, interpretable. ([github](https://github.com/allenai/visprog))
- **ViperGPT** — LLM generates Python code composing vision-language models as subroutines; applied to images and video. ([arxiv:2303.08128](https://arxiv.org/abs/2303.08128))
- **MoReVQA** (CVPR 2024) — Three-stage modular reasoning for VideoQA: event parsing -> temporal grounding -> reasoning. All stages are training-free via few-shot prompting. ([arxiv:2404.06511](https://arxiv.org/abs/2404.06511))

**Perception/reasoning boundary:** LLM handles planning and reasoning; specialized vision models handle perception. The LLM never "sees" images directly — it orchestrates tools.

---

## 3. Agent Systems for Video Understanding

- **VideoAgent** (ECCV 2024) — LLM as agent, VLM + CLIP as tools. Iteratively selects keyframes to inspect. Achieves strong zero-shot VideoQA with only ~8 frames on average. ([arxiv:2403.10517](https://arxiv.org/abs/2403.10517))
- **VideoAgent2** (2025) — Adds uncertainty-aware chain-of-thought; +13.1% over VideoAgent. Addresses error propagation from external tools. ([arxiv:2504.04471](https://arxiv.org/abs/2504.04471))
- **LongVideoAgent** (2025) — Multi-agent: Master LLM coordinates a GroundingAgent (temporal localization) + VisionAgent (visual perception). Master trained with GRPO reinforcement learning. ([arxiv:2512.20618](https://arxiv.org/abs/2512.20618))
- **VideoThinker** (2026) — Agentic VideoLLM with two tool types: Temporal Retrieval (clip/subtitle search) and Temporal Zoom (fine-grained inspection). Trained on synthetic tool-interaction trajectories. ([arxiv:2601.15724](https://arxiv.org/abs/2601.15724))
- **VideoSeek** (CVPR 2026) — Think-act-observe loop agent with a 3-tool toolkit: *overview* (coarse storyline), *skim* (probe candidate intervals), *focus* (inspect short clips for details). Mimics human-like seeking behavior — uses far fewer frames while improving accuracy. Outperforms GPT-5 by +10.2 on LVBench while using 93% fewer frames. ([arxiv:2603.20185](https://arxiv.org/abs/2603.20185), [github](https://github.com/jylins/videoseek))
- **Agent-of-Thoughts Distillation** (CVPR 2025) — Distills agentic reasoning traces into a student Video-LLM, removing the need for tool calls at inference. ([CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_Enhancing_Video-LLM_Reasoning_via_Agent-of-Thoughts_Distillation_CVPR_2025_paper.pdf))

**Perception/reasoning boundary:** VLMs/CLIP handle frame-level perception; the LLM agent handles temporal reasoning, frame selection, and answer synthesis.

---

## 4. Agent Systems for Spatial Reasoning (most relevant to our project)

- **GCA — Geometrically-Constrained Agent** (Nov 2025) — Training-free. Decouples VLM into two roles: (1) semantic analyst that formalizes the query into a geometric constraint, (2) task solver that executes tool calls within those constraints. +27% over prior SOTA on spatial benchmarks. ([arxiv:2511.22659](https://arxiv.org/abs/2511.22659))
- **RieMind** (Mar 2026) — Builds a 3D scene graph (3DSG) from video via a dedicated perception module; then an LLM (not VLM!) queries the graph with geometric tools (distances, volumes, poses). +16% on VSI-Bench over fine-tuned spatial models. **Directly decouples perception from reasoning.** ([arxiv:2603.15386](https://arxiv.org/abs/2603.15386))
- **World2Mind** (Mar 2026) — Constructs an Allocentric Spatial Tree (AST) from 3D reconstruction + instance segmentation; converts it to text so even text-only LLMs can do 3D spatial reasoning. Three-stage chain: tool invocation assessment -> modality-decoupled cue collection -> geometry-semantics reasoning. ([arxiv:2603.09774](https://arxiv.org/abs/2603.09774))
- **TRACE** (Mar 2026) — Prompting method that induces MLLMs to generate text-based 3D environment representations as intermediate reasoning traces for spatial QA from egocentric video. ([arxiv:2603.23404](https://arxiv.org/abs/2603.23404))

**Perception/reasoning boundary:** These all externalize geometry — perception modules produce structured representations (scene graphs, spatial trees, depth maps), and the LLM reasons over those structures rather than raw pixels.

---

## 5. Related: Spatial VLMs (not agent-based, but relevant context)

- **SpatialVLM** — Trains VLMs on 2B synthetic spatial VQA examples for quantitative spatial reasoning. ([arxiv:2401.12168](https://arxiv.org/abs/2401.12168))
- **SpatialBot** — Uses RGB + depth as dual input for spatial understanding. ([arxiv:2406.13642](https://arxiv.org/abs/2406.13642))

---

## Key Takeaway

The clearest trend is **externalizing geometry from VLMs**: RieMind, GCA, and World2Mind all show that using VLMs only for perception (object detection, segmentation, scene graph construction) while handling spatial/geometric reasoning in an LLM with structured tools significantly outperforms end-to-end VLM approaches. This directly validates our research direction. RieMind on VSI-Bench is particularly relevant since that's the same benchmark used in the Thinking-in-Space paper.
