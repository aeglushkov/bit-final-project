# Reconstruction Agents — Detect → Reconstruct → Refine on Long Video (v1, working)

**Status:** v1 working draft — paper entries are synthesized from three independent
LLM-agent searches (Grok / GPT-5-thinking / Gemini deep-research style outputs);
**none of the papers below have been read end-to-end yet**. Treat each card as a
triage hypothesis, not a verified claim. Verify venue, authors, and method
details before citing.

**Scope:** agent or multi-agent frameworks built on top of multimodal foundation
models (VLMs / video-LLMs / MLLMs) that **reconstruct missing or incomplete
information in long-video / streaming input** by calling generative or
specialist sub-models, then feed the reconstructed signal back into reasoning.
Driven by Su's 2026-04-17 architecture vision (see
[notes/meetings/2026-04-17.md](../meetings/2026-04-17.md)).

## Relevance criteria

| # | Criterion | Strict reading |
|---|---|---|
| **C1** | Agent / multi-agent on top of a frozen VLM/MLLM | Excludes end-to-end fine-tuned monolithic VLMs. |
| **C2** | Detect gaps in visual input | Agent explicitly identifies missing/insufficient information. |
| **C3** | Reconstruct via generative/specialist models that create *new content beyond the original input* | 3D recon, depth, segmentation lifted to 3D, novel-view synthesis, world models, scene-graph synthesis from sparse views. **Pure retrieval / re-captioning of existing frames does NOT count.** |
| **C4** | Refine loop — reconstructed signal feeds back into reasoning | Closed loop, not one-shot. |
| **C5** | Long-video / streaming input | Long video (minutes+), sparse-video stream. Single image / static multi-view / embodied exploration = partial. RGB-D streams = partial-to-strong. |
| **C6** | Venue tier | Top 2024–26 venues > arXiv 2024–26. |

**Hard disqualifiers:** fails C3 (no reconstruction) or fails C1 (not an agent).

## Tier summary

| Tier | What it means | Papers |
|---|---|---|
| **Tier 1** | Strong on all six criteria — read these first | SAVVY · Just-in-Time Digital Twins · LIRA · Embodied VideoAgent · Feature4X |
| **Tier 2** | Strong pattern, fails one criterion (usually long-video) | MindJourney · VADAR · GraphEQA · Fisher-Info Active Mapping · pySpatial |
| **Tier 3** | Strong pattern, arXiv-only (top of preprint pile) | MAG-3D · TAB · Think3D · GCA · Scene-R1 · Thinking with Spatial Code |
| **Tier 4** | Partial / adjacent | GraphPad · 4D Laparoscopic · Agentic 3D Scene Gen · Active 3D Exploration · SpaceTools · SpatiO · 3DThinker · VideoMultiAgents · IR3D-Bench |
| **Disqualified** | Defines the negative space of the brief | VideoAgent · ReViSe · VLM-3R · VAGEN |

## Navigation

- [Tier 1](#tier-1--strong-on-all-criteria)
  - [SAVVY](#savvy-spatial-awareness-via-audio-visual-llms)
  - [Online Reasoning Video Segmentation w/ Just-in-Time Digital Twins](#online-reasoning-video-segmentation-with-just-in-time-digital-twins)
  - [LIRA: Reasoning Reconstruction via MLLMs](#lira-reasoning-reconstruction-via-multimodal-large-language-models)
  - [Embodied VideoAgent](#embodied-videoagent-persistent-memory-from-egocentric-videos-and-embodied-sensors)
  - [Feature4X](#feature4x-monocular-video-to-4d-agentic-ai-with-gaussian-feature-fields)
- [Tier 2](#tier-2--strong-pattern-fails-one-criterion)
- [Tier 3](#tier-3--strong-pattern-arxiv-only)
- [Tier 4](#tier-4--partial--adjacent)
- [Disqualified](#disqualified--defines-the-negative-space)

---

## Tier 1 — strong on all criteria

### SAVVY: Spatial Awareness via Audio-Visual LLMs

`NeurIPS 2025 Oral` · 🏛️ (TBC after verification)

[📄 Paper](https://arxiv.org/abs/2506.05414) · [🚀 NeurIPS poster](https://neurips.cc/virtual/2025/poster/115001)

🏷️ **SUBJECT:** Training-free agent pipeline for 3D spatial QA over long egocentric audio-visual streams.

❓ **PROBLEM:**
- Long egocentric AV streams contain off-screen objects/events the VLM never directly observes.
- Single-pass AV-LLMs cannot answer 3D spatial queries that require integrating cues across rooms and time.
- No principled way to fuse spatial audio (7-mic Aria glasses) with vision into a unified scene state.

💡 **IDEA:** A two-stage agent: stage-1 invokes AV-LLMs and audio-visual specialists to estimate **per-object egocentric trajectories** (incl. off-screen) from vision + spatial audio; stage-2 aggregates trajectories into a **dynamic global map**, then answers via coordinate transform to the query viewpoint.

🛠️ **SOLUTION:**
- **Trajectory specialists:** AV-LLM + specialist tools estimate object tracks from audio + vision.
- **Dynamic global map:** trajectories aggregated into a unified 3D representation that persists across the stream.
- **Query-time coord-transform:** map → query viewpoint to produce the spatial answer.

🏆 **RESULTS:** Beats Gemini-2.5-Pro by **+7.1%** on SAVVY-Bench.

💭 **THOUGHTS:**
- **Closest fit to Su's vision** — long stream + multi-specialist reconstruction + agentic detect-build-refine.
- Verify: how is the "global map" represented? Persistent 3D coordinates? Scene graph? This is the part most relevant to Su's 04-17 architecture.

---

### Online Reasoning Video Segmentation with Just-in-Time Digital Twins

`ICCV 2025` · 🏛️ (TBC)

🏷️ **SUBJECT:** Streaming reasoning segmentation as an LLM-driven agent that builds on-demand digital twins instead of pushing video through a compressed VLM.

❓ **PROBLEM:**
- Implicit / under-specified text queries demand semantic + spatial + temporal reasoning that compressed video-LLM pathways cannot reliably support.
- Existing online segmentation methods either run all tools always (expensive) or rely on a single foundation model (insufficient for compositional queries).

💡 **IDEA:** An LLM **planner** reads the implicit query and decides which specialist vision tools are actually needed; specialists build a **just-in-time dynamic scene graph** preserving semantic/spatial/temporal state over a streaming window; an LLM **reasoner + coder** executes a DAG of operations over the twin to produce frame-level masks.

🛠️ **SOLUTION:**
- **Planner agent:** query analysis → tool selection.
- **Just-in-time digital twin:** dynamic scene graph from segmentation + detection + depth specialists.
- **Reasoner + coder agents:** run a DAG over the twin, refresh as new frames arrive.
- **Benchmark:** 200 videos / 895 implicit text queries.

🏆 **RESULTS:** TBC on read; reported as top-tier on its own benchmark.

💭 **THOUGHTS:**
- **Most explicit detect → reconstruct → refine on streaming video** in the whole survey.
- Architecture is closest in shape to Su's "agent detects gaps → calls specialists → reasons over reconstruction."

---

### LIRA: Reasoning Reconstruction via Multimodal Large Language Models

`ICCV 2025` · 🏛️ (TBC)

🏷️ **SUBJECT:** Introduces *reasoning reconstruction* — given an implicit instruction + RGB-D sequence, output an incremental 3D reconstruction of instruction-relevant instances.

❓ **PROBLEM:**
- Standard 3D grounding assumes the relevant object is named explicitly; implicit instructions break this assumption.
- Pure retrieval of clips / boxes is not enough — the downstream answer needs a 3D representation of the *target*, not the raw stream.

💡 **IDEA:** MLLM **infers instruction-relevant 2D candidate instances + attributes**, back-projects them into an incrementally reconstructed 3D map, fuses across keyframes via a **TIFF module**, then a global LLM stage performs target-instance inference.

🛠️ **SOLUTION:**
- **Implicit-instruction reasoner (MLLM):** identifies missing/relevant objects + attributes.
- **2D → 3D back-projection:** lifts candidates onto an evolving 3D map.
- **TIFF (multi-keyframe fusion):** consolidates candidate evidence over time.
- **Global LLM target-inference stage:** picks the actual target instance from the fused candidates.

🏆 **RESULTS:** TBC on read.

💭 **THOUGHTS:**
- Defines a task that almost paraphrases Su's brief — the *output* is a reconstruction, not just an answer.
- Verify what "TIFF" stands for; the source summary is unclear.

---

### Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors

`ICCV 2025` · 🏛️ (TBC)

[📄 Paper](https://arxiv.org/abs/2501.00358) · [📄 Proceedings PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Embodied_VideoAgent_Persistent_Memory_from_Egocentric_Videos_and_Embodied_Sensors_ICCV_2025_paper.pdf)

🏷️ **SUBJECT:** LLM agent that maintains a persistent 3D object memory built from egocentric video + depth/pose sensors, queried via tool calls.

❓ **PROBLEM:**
- Single-pass egocentric video QA cannot answer questions about objects no longer in view or about events distributed over a long horizon.
- Naïve all-frame ingestion is intractable at egocentric video lengths.

💡 **IDEA:** An LLM agent maintains a **persistent 3D object memory** over the stream; a separate VLM watches new clips and **triggers memory updates when actions change scene state**; the LLM answers EQA-style questions by calling `query_db / temporal_loc / spatial_loc / vqa` over the memory.

🛠️ **SOLUTION:**
- **Persistent 3D object memory:** built from egocentric video + depth + pose.
- **VLM trigger:** detects state changes that warrant a memory update.
- **LLM tool surface:** `query_db`, `temporal_loc`, `spatial_loc`, `vqa`.

🏆 **RESULTS:** **+4.9** on Ego4D-VQ3D, **+5.8** on OpenEQA, **+11.7** on EnvQA.

💭 **THOUGHTS:**
- **Strongly relevant to the memory-mechanism thread** Su flagged on 04-17, not just the reconstruction thread.
- C3 caveat: depth + pose are sensor-precomputed (not live tool calls), but the persistent 3D memory is genuine reconstruction of currently-unseen context.

---

### Feature4X: Monocular Video to 4D Agentic AI with Gaussian Feature Fields

`CVPR 2025` · 🏛️ (TBC, multi-institution)

🏷️ **SUBJECT:** Reconstructs an explicit 4D scene + unified 4D Gaussian feature field from monocular video; LLM feedback loops do segmentation, scene editing, and free-form VQA over the reconstruction.

❓ **PROBLEM:**
- Direct VQA over monocular video misses spatiotemporal context that is *implicit* in the footage but never observed from the right viewpoint.
- Distilling 2D / video foundation models (SAM2, InternVideo2) into a coherent 4D representation is non-trivial.

💡 **IDEA:** Build a **4D Gaussian feature field** by distilling 2D / video FMs into a spatiotemporally coherent representation; then route LLM reasoning through the **reconstructed feature space** rather than raw frames.

🛠️ **SOLUTION:**
- **4D radiance + feature field** from monocular video.
- **Foundation-model distillation:** SAM2 + InternVideo2 → unified 4D feature space.
- **LLM feedback loops** over reconstructed novel views and feature queries for segmentation / editing / VQA.

🏆 **RESULTS:** Reports that 4D-feature-space inference improves spatiotemporal VQA over direct input-video inference *and* reduces latency.

💭 **THOUGHTS:**
- C1 caveat: agent framing is light — "LLM feedback loops" is closer to a tuned pipeline than an explicit ReAct-style agent. Verify on read.
- Most direct precedent for "reconstruct from video, then reason over the reconstruction."

---

## Tier 2 — strong pattern, fails one criterion

### MindJourney: Test-Time Scaling with World Models for Spatial Reasoning

`NeurIPS 2025` · [📄 Paper](https://arxiv.org/abs/2507.12508) · [🚀 NeurIPS poster](https://neurips.cc/virtual/2025/poster/118581)

Frozen VLM ↔ controllable video-diffusion world model. VLM proposes a short camera trajectory; the world model **synthesizes the resulting egocentric views**; VLM re-reasons over the synthesized multi-view evidence. **>+7.7%** on SAT, no fine-tuning.
- **Why Tier 2:** input is a single starting view, not long video — but the architectural loop ("imagine the unseen viewpoint, then re-reason") is the cleanest blueprint to port to streaming video.

### VADAR: Visual Agentic AI for Spatial Reasoning with a Dynamic API

`CVPR 2025` · Caltech · [📄 Paper](https://arxiv.org/abs/2502.06787)

Multi-agent system (Signature / Implementation / Test / Program / Execution agents) that synthesizes a Pythonic API on the fly over Molmo + GroundingDINO + SAM2 + UniDepth + GPT-4o. Closes the gap to GPT-4o on Omni3D-Bench.
- **Why Tier 2:** image / multi-view input, not long video. Already in [agent-based-approaches-v2.md](agent-based-approaches-v2.md) — included here as the cleanest *multi-agent* architectural reference.

### GraphEQA: Real-Time 3D Semantic Scene Graphs for Embodied QA

`RSS 2025 SemRob workshop` · [📄 Paper](https://arxiv.org/abs/2412.14480)

Robot constructs a real-time 3D metric-semantic scene graph as it explores; graph + task-relevant images become multimodal memory for a VLM; hierarchical planner uses the graph for semantic-guided next-best-view exploration. Validated on HM-EQA, OpenEQA, real-world.
- **Why Tier 2:** embodied exploration target, not passive long-video QA; venue is workshop.

### Multimodal LLM Guided Exploration & Active Mapping using Fisher Information

`ICCV 2025` · (Wen Jiang, Boshu Lei, Katrina Ashton, Kostas Daniilidis)

3DGS scene representation + MLLM long-horizon goal selection + Fisher-information-based short-term planning. **Explicitly uses information gain to decide where the gaps are most valuable**, then gathers observations that improve the reconstruction.
- **Why Tier 2:** target is embodied mapping, not long-video QA — but detect-gap-via-Fisher-info is the most principled gap-detection mechanism in the survey.

### pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning

`ICLR 2026` · (Zhanpeng Luo et al.)

MLLM as Python code-generation agent composing spatial tools incl. 3D reconstruction, camera-pose recovery, novel-view rendering. Converts 2D observations into an explorable 3D scene, executes a program over it, then conditions the final answer on both the reconstruction and the original frames.
- **Why Tier 2:** mostly 2D observations, not long-video focused. Strong tool-as-reconstruction blueprint.

---

## Tier 3 — strong pattern, arXiv only

### MAG-3D: Multi-Agent Grounded Reasoning for 3D Understanding

`arXiv 2026`

Planning Agent + Grounding Agent + Coding Agent linked through a shared scene memory (object candidates, reconstructed geometry, visual memory, measurements). Grounding Agent does open-vocab 2D segmentation → depth-and-pose-based 2D-to-3D lifting → 3D-instance consolidation; Coding Agent does explicit geometric verification.
- **Strongest multi-agent direct match** — closest in shape to Su's brain-like multi-agent vision.

### TAB: Think, Act, Build for Zero-Shot 3D Visual Grounding

`arXiv 2026`

Dynamic Think–Act–Build loop on RGB-D streams: VLM does coarse-to-fine frame filtering, reference-view selection, segmentation-based isolation, semantic temporal expansion, initial 3D reconstruction, and **Semantic-Anchored Geometric Expansion** to project the target into unobserved frames and densify the point cloud. Paper explicitly frames itself as repairing the "multi-view coverage deficit" — almost a paraphrase of Su's brief.

### Think3D: Thinking with Space for Spatial Reasoning

`arXiv 2026`

Training-free framework that reconstructs 3D point clouds + camera poses from videos / multi-view images, then lets the VLM agent actively manipulate the 3D scene (camera-based actions, ego/global-view switching, iterative spatial exploration). RL variant teaches smaller models to choose more informative exploration actions.

### GCA: Geometrically-Constrained Agent for Spatial Reasoning

`arXiv 2025` · already in [v2](agent-based-approaches-v2.md) · [📄 Paper](https://arxiv.org/abs/2511.22659)

Two-stage VLM agent (semantic analyst + task solver) over an 8-tool toolbox (VGGT, open-vocab detection, 6-DoF pose, metric scale, OCR, optical flow, code sandbox). RAG-injected geometric formulas. **+12** over Gemini-2.5-Pro on 5 spatial benchmarks.
- **Why Tier 3 here:** image / multi-view oriented, not long-video primary.

### Scene-R1: Video-Grounded LLMs for 3D Scene Reasoning without 3D Annotations

`arXiv 2025`

Two-stage RL-trained pipeline: temporal grounding selects the relevant snippet → image grounding predicts 2D boxes → SAM2 mask propagation → 2D-to-3D lifting for final localisation / 3D VQA. Selected evidence is materially transformed into a 3D representation, not just retrieved.

### Thinking with Spatial Code for Physical-World Video Reasoning

`arXiv 2026`

Spatial encoder combining SAM-2 (object features) + Depth Anything 3 (geometry) jointly performs segmentation, tracking, and 3D reconstruction → produces a temporally coherent **spatial code** of explicit 3D oriented boxes + semantic labels. A text-only LLM is prompted on the spatial code, refined with a spatial rubric reward.
- C1 caveat: less agentic, more pipeline. Gap detection is implicit.

---

## Tier 4 — partial / adjacent

| Paper | Venue | Why down-ranked |
|---|---|---|
| **GraphPad** — Inference-Time 3D Scene Graph Updates for Embodied QA | `CVPR 2025 Workshop` ([arXiv 2506.01174](https://arxiv.org/abs/2506.01174)) | Recon is partial — re-detect / insert nodes, no generative geometry. Workshop venue. |
| **A 4D Representation for Training-Free Agentic Reasoning from Monocular Laparoscopic Video** | `arXiv 2026` | Pattern is right (Depth Anything 3 + CoTracker 3 + SASVi → 4D trajectories → MLLM agent), but reconstruction is offline / pre-computed, and domain is locked to surgical video. |
| **Agentic 3D Scene Generation with Spatially Contextualized VLMs** | `arXiv 2025` ([2505.20129](https://arxiv.org/abs/2505.20129)) | Scene *generation* focus, not video QA. Agentic spatial-context loop is transferable in spirit. |
| **Active 3D Scene Exploration for Multi-Perspective Reasoning** | `arXiv 2026` | Cleanest "imagine the unseen viewpoint" loop on a single image (MLLM → SAM3 → SAM3D-Object mesh → novel-view render → MLLM validate → iterate). **Single image input.** |
| **SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL** | `arXiv 2025` ([2512.04069](https://arxiv.org/abs/2512.04069)) | RL-trained VLM emits `<tool_call>` for SAM2 / DepthPro / Molmo / GraspGen. **Single image / RGB-D snapshot, not video.** |
| **SpatiO: Adaptive Test-Time Orchestration of Vision-Language Agents** | `arXiv 2026` | Three orchestrated roles (Explicit-3D-Reconstruction / Scene-Graph / visual CoT). **Image-only**, video extension is future work. |
| **3DThinker (Think with 3D)** | `CVPR 2026` (arXiv journal-ref) | Aligns VLM internal 3D latent with VGGT latent; reconstruction is *internal latent*, not external tool — softer C3 match. |
| **VideoMultiAgents** | `arXiv 2025` ([2504.20091](https://arxiv.org/abs/2504.20091)) | Three specialist agents (vision / scene-graph / text) on video QA. **No geometric reconstruction** — fails C3. Listed because the scene-graph agent is the closest in spirit. |
| **IR3D-Bench: VLM Scene Understanding as Agentic Inverse Rendering** | `NeurIPS 2025 D&B` | Benchmark, not method. "Understanding by creating" via Blender programs — useful as an evaluation lens for the pattern. |

---

## Disqualified — defines the negative space

These papers fail a hard criterion. Listed because they sharpen what *is* in scope.

| Paper | Venue | Failure |
|---|---|---|
| **VideoAgent** (Wang, Zhang, Zohar, Yeung-Levy) | `ECCV 2024` ([2403.10517](https://arxiv.org/abs/2403.10517)) | Fails **C3**: only retrieves and re-captions frames already in the video — pure "when to look," which Su's prompt explicitly excluded. |
| **ReViSe** (Towards Sparse Video Understanding and Reasoning) | `arXiv 2026` | Same as VideoAgent — selection + captioning, no reconstruction component. |
| **VLM-3R** (Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction) | `CVPR 2026` ([2505.20279](https://arxiv.org/abs/2505.20279)) | Fails **C1**: end-to-end tuned VLM, not an agent framework. Useful as a representational reference, not an architectural one. |
| **VAGEN** (Reinforcing World Model Reasoning for Multi-Turn VLM Agents) | `NeurIPS 2025` ([2510.16907](https://arxiv.org/abs/2510.16907)) | Fails **C3** (no external generative reconstruction; only internal world-model reasoning) and **C5** (interactive POMDPs, not video stream). |

> **Negative-space takeaway:** the "when-to-look" family (VideoAgent, ReViSe) is exactly what Su's 04-16/04-17 framing aims to *move past*. VLM-3R is what Su explicitly *doesn't* want — a single fine-tuned VLM rather than a multi-agent architecture.

---

## Open questions for v2

- Verify all venue tags, author lists, and arXiv IDs against the actual papers — several entries here came from the same agent's bullet-format output and may carry transcription errors.
- For each Tier 1 paper, write a full `analysis.md`-style read after the deep pass.
- Re-evaluate the Tier 1 → Tier 2 boundary once "long-video" is more precisely defined — is RGB-D from an Aria headset over a multi-room tour "long video" in Su's sense, or is it embodied-exploration-adjacent?
- Cross-reference with [memory-mechanisms-v1-working.md](memory-mechanisms-v1-working.md): Embodied VideoAgent and SAVVY both straddle reconstruction and memory — the persistent-3D-memory thread may unify the two surveys.
