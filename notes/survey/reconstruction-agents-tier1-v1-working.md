# Reconstruction Agents — Tier 1 (v1, working)

**Status:** v1 working draft — paper entries are synthesized from three
independent LLM-agent searches; **none of the papers below have been read
end-to-end yet**. Treat each card as a triage hypothesis, not a verified claim.
Verify venue, authors, and method details before citing.

**Scope:** Tier 1 subset of [reconstruction-agents-v1-working.md](reconstruction-agents-v1-working.md) —
papers that satisfy *all six* relevance criteria (agent on top of frozen VLM ·
gap detection · reconstruction via generative/specialist models · refine loop ·
long-video / streaming input · top 2024–26 venue). Driven by Su's 2026-04-17
architecture vision (see [notes/meetings/2026-04-17.md](../meetings/2026-04-17.md)).

For the full ranking (Tier 2 / 3 / 4 / disqualified) and the criteria
definitions, see [reconstruction-agents-v1-working.md](reconstruction-agents-v1-working.md).

## At a glance

| # | Paper | Venue |
|---|---|---|
| 1 | [SAVVY: Spatial Awareness via Audio-Visual LLMs](#savvy-spatial-awareness-via-audio-visual-llms) | `NeurIPS 2025 Oral` |
| 2 | [Online Reasoning Video Segmentation w/ Just-in-Time Digital Twins](#online-reasoning-video-segmentation-with-just-in-time-digital-twins) | `ICCV 2025` |
| 3 | [LIRA: Reasoning Reconstruction via MLLMs](#lira-reasoning-reconstruction-via-multimodal-large-language-models) | `ICCV 2025` |
| 4 | [Embodied VideoAgent](#embodied-videoagent-persistent-memory-from-egocentric-videos-and-embodied-sensors) | `ICCV 2025` |
| 5 | [Feature4X](#feature4x-monocular-video-to-4d-agentic-ai-with-gaussian-feature-fields) | `CVPR 2025` |

---

## SAVVY: Spatial Awareness via Audio-Visual LLMs

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

## Online Reasoning Video Segmentation with Just-in-Time Digital Twins

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

## LIRA: Reasoning Reconstruction via Multimodal Large Language Models

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

## Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors

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

## Feature4X: Monocular Video to 4D Agentic AI with Gaussian Feature Fields

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
