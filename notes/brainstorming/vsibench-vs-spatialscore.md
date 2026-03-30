# VSI-Bench vs SpatialScore: Benchmark Comparison

**Date:** 2026-03-30

## Benchmark Scope

| | VSI-Bench (Thinking in Space) | SpatialScore |
|---|---|---|
| **Tasks** | 8 tasks, 3 categories | 30 tasks, 10 categories |
| **Input** | Video only (egocentric walkthroughs) | Image + multi-frame + video |
| **Scenes** | Real indoor only (ScanNet, ARKitScenes) | Real + simulated + AIGC |
| **Scale** | 5K QA, 288 videos | 5K QA from 23 source datasets |
| **Models tested** | 15 | 40 |
| **Venue** | CVPR 2025 Oral | arXiv 2025 |

SpatialScore subsumes VSI-Bench — it's one of SpatialScore's 23 source datasets.

## What Each Measures

**VSI-Bench** focuses on a narrow but deep question: can MLLMs build a mental 3D model from egocentric video? Its 8 tasks test counting, distance estimation, size estimation, relative direction, route planning, and appearance order — all from video walkthroughs of indoor scenes.

**SpatialScore** goes much broader, adding categories VSI-Bench doesn't touch:
- Camera pose & motion (intrinsics, extrinsics, homography)
- Mental animation (spatial folding, 2D/3D rotation, maze navigation)
- Object motion (optical flow-level understanding)
- Depth estimation (relative and absolute)
- Multiple QA formats (judgment, MC, open-ended) vs. VSI-Bench's two (MC + numerical)

## Which Better Evaluates Spatial Capabilities?

**SpatialScore is more comprehensive** — covers far more spatial skill types and modalities. Better for broad characterization.

**VSI-Bench has unique strengths:**

1. **Deeper diagnostic insight.** 71% of errors are egocentric-allocentric transformation failures (not perception failures) — more actionable than aggregate scores.
2. **The CoT finding.** Chain-of-thought *hurts* spatial tasks — foundational insight about the nature of spatial reasoning.
3. **Cognitive maps.** Analysis of how models build local-but-not-global spatial representations connects to cognitive science and points toward solutions.
4. **Purity of measurement.** Tests a coherent, well-defined capability (3D spatial understanding from video). SpatialScore's breadth mixes very different skills.

## Relevance to Our Research

For our direction (externalizing spatial reasoning, separating perception from reasoning):

- **VSI-Bench** is more theoretically useful — identifies the specific bottleneck (ego-allo transformation) that our agent layer aims to solve
- **SpatialScore** is more useful as an evaluation target — 30 tasks would thoroughly test generalization, and SpatialAgent provides a direct baseline to beat

They're complementary: VSI-Bench tells us *where* the problem is; SpatialScore tells us *how big* the problem is across the full surface area.
