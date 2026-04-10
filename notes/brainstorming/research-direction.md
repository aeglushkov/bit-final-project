# Research Direction: Agent-Based Spatial Reasoning Over Video with Externalized Geometry

**Date:** 2026-04-08
**Status:** Formulated after literature survey of 7 papers (see [literature-survey.md](literature-survey.md))
**Builds on:** Initial idea from Diwei Su (see [../idea-agent-architecture.md](../idea-agent-architecture.md))

---

## Problem Statement

Video-capable multimodal LLMs fail at spatial reasoning despite strong perceptual capabilities. The evidence is consistent across multiple independent studies:

- **71% of errors** are egocentric-allocentric transformation failures, while only 8% are perception errors (Thinking in Space, CVPR 2025 Oral)
- **26.48-point gap** to human performance across 30 spatial task types (SpatialScore, 2025)
- Standard reasoning prompts (CoT, ToT) **hurt** spatial performance by 4% on average — spatial reasoning is fundamentally different from linguistic reasoning
- Spatial fine-tuning causes **catastrophic forgetting** of other capabilities

The core issue: VLMs see objects correctly but cannot transform between egocentric (camera-relative) and allocentric (world-relative) reference frames. This is a geometric reasoning problem being forced through a pattern-matching pipeline.

## Thesis

**An agent that externalizes geometric reasoning — using VLMs only for perception while handling spatial transformations through deterministic geometric tools — will substantially outperform both end-to-end VLMs and prompting-based approaches on video spatial understanding tasks.**

This thesis is empirically grounded:
- RieMind achieves **89.5%** on VSI-Bench with externalized geometry (ground-truth 3DSG + tools), vs **73.6%** for the best fine-tuned model and **~50%** for the best zero-shot VLM
- GCA achieves **+27% over SOTA** on image-based spatial benchmarks by formalizing the semantic-geometric gap
- Even TRACE's prompting-only externalization yields **+7.5%** consistent gains

## The Gap We Fill

The literature reveals a clear gap. No existing work combines all four requirements:

| | Video Input | External Geometry | Practical Perception | Explicit Ego-Allo |
|---|:---:|:---:|:---:|:---:|
| **GCA** (CVPR 2026) | -- | Yes | Yes | Yes |
| **RieMind** (2026) | -- (GT 3DSG) | Yes | -- (ground truth) | Yes |
| **VideoSeek** (2026) | Yes | -- (temporal only) | Yes | -- |
| **TRACE** (2026) | Yes | -- (prompting) | Yes | Prompted |
| **Ours** | **Yes** | **Yes** | **Yes** | **Yes** |

- **GCA** formalizes geometric constraints beautifully but is **image-only** — no video, no temporal reasoning
- **RieMind** proves externalization works but assumes **ground-truth 3D scene graphs** — not practical
- **VideoSeek** navigates video efficiently but has **no spatial tools** — purely temporal
- **TRACE** shows externalization helps via prompting but **VLM still does all the work** — bounded by VLM perception quality

Our contribution: a practical agent for video spatial reasoning that works with real (imperfect) perception, not ground-truth geometry.

## Proposed Approach

### Architecture: Perceive-Formalize-Compute

A three-stage agent pipeline, drawing on the strongest ideas from each paper:

**Stage 1: Perceive (VLM as perception engine)**
Inspired by VideoSeek's multi-granularity toolkit and TRACE's entity registry.

The agent uses the VLM solely for perception through targeted queries:
- **Scene overview** — what objects are present, rough spatial layout (analogous to VideoSeek's `<overview>`)
- **Spatial scan** — depth estimation, object localization in specific frames (analogous to `<skim>`)
- **Precise observation** — fine-grained details of specific objects/regions (analogous to `<focus>`)

The VLM is never asked "what is the spatial relationship between X and Y?" — only "what do you see here?" and "where is object X in this frame?"

**Stage 2: Formalize (Task decomposition)**
Adopted from GCA's C_task = (C_R, C_O) formalism.

Before any computation, the agent formalizes the spatial question:
- **C_R: Reference Frame Constraint** — which coordinate system is needed? (GCA's ablation shows this is 5x more important than the objective)
- **C_O: Objective Constraint** — what geometric quantity to compute?

Extended for video: **C_R(t)** — reference frames that may vary over time (e.g., camera movement, object displacement). This is the novel extension GCA doesn't address.

**Stage 3: Compute (Deterministic geometric tools)**
Inspired by RieMind's geometry tools, but operating on estimated (not ground-truth) data.

Geometric tools for:
- Coordinate transformations (ego-to-allo, frame-to-frame)
- Distance computation (Euclidean, along paths)
- Spatial relation derivation (relative direction, containment, adjacency)
- Size/volume estimation from depth + detection
- Cross-frame trajectory integration

All tools are deterministic — given the same inputs, they produce the same outputs. Uncertainty comes from perception (Stage 1), not reasoning (Stage 3).

### Key Design Principles

1. **Interleave perception and reasoning** (from TRACE). Don't pre-compute a full scene graph then reason over it. TRACE shows one-stage > two-stage: the agent should request visual information as the reasoning demands it, not front-load all perception.

2. **Formalize before computing** (from GCA). The agent must identify the reference frame and objective before invoking tools. This prevents the VLM from hallucinating geometric conclusions.

3. **Coarse-to-fine visual exploration** (from VideoSeek). Start with an overview, zoom into relevant segments, then make precise observations. Most spatial questions don't require processing every frame.

4. **Training-free** (from SpatialScore). Agent approach avoids catastrophic forgetting and works across VLM backbones. SpatialScore shows agent gains (+7.78) are competitive with fine-tuning (+10.47) while being more robust.

5. **Graceful degradation with imperfect perception**. Unlike RieMind's ground-truth assumption, the agent must handle noisy depth estimates, missed detections, and uncertain localization. This is the main engineering challenge.

## Evaluation Plan

**Primary benchmark:** VSI-Bench (Thinking in Space)
- Direct comparison with RieMind's results (89.5% with GT, our target: close that gap with estimated perception)
- 8 task types covering configurational + measurement spatial reasoning
- Established baseline: ~50% (best VLM), 73.6% (best fine-tuned), 89.5% (RieMind GT)

**Secondary benchmark:** SpatialScore
- 30 task types for breadth across spatial skills
- Agent evaluation protocol already built in
- Comparison with SpatialAgent baseline (+7.78)

**Key metrics to report:**
- Overall accuracy on VSI-Bench (MRA for numerical, accuracy for MCA)
- Per-task breakdown (especially relative direction — the hardest task where even RieMind's Qwen degrades)
- Comparison across VLM backbones (to show generalizability, as GCA does)
- Ablation: with/without formalization stage, with/without geometric tools, with ground-truth vs estimated perception (to measure perception pipeline quality)

## Risks and Open Questions

1. **Perception pipeline quality.** The gap between RieMind's 89.5% (GT) and practical performance will depend entirely on how good our depth estimation, object detection, and localization are. If perception noise propagates catastrophically through geometric tools, the approach may not beat simpler prompting methods like TRACE.

2. **Reference frame identification from video.** GCA shows C_R is the critical step, but they let the VLM handle it (30% error rate). Can we do better with video (multiple viewpoints provide more geometric constraints than a single image)?

3. **Dynamic reference frames C_R(t).** No existing work handles time-varying reference frames. This is a novel extension with unknown difficulty.

4. **Cost and latency.** VideoSeek uses ~13 LLM calls per question. Adding geometric tool calls could make our agent expensive. Need to balance accuracy with practical efficiency.

5. **Which VLM backbone?** RieMind shows base model reasoning matters hugely (Qwen 34.7% vs GPT-4.1 87.3% on relative direction). Do we need a proprietary model, or can an open-source model suffice when geometric reasoning is externalized?

## Positioning

This work sits at the intersection of:
- **GCA's formalism** (how to decompose spatial problems) extended to video
- **RieMind's externalization** (what's possible with perfect geometry) made practical
- **VideoSeek's efficiency** (how to navigate video intelligently) applied to spatial reasoning
- **TRACE's insight** (interleave perception and reasoning) as a design principle

The contribution is not any single component but their integration into a practical system that works on real video without ground-truth 3D data.
