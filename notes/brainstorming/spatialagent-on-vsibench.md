# Can SpatialAgent Be Applied to VSI-Bench Directly?

**Date:** 2026-03-30

## Short Answer

Not without modifications. Works partially but has meaningful gaps.

## What Works

- VSI-Bench is one of SpatialScore's 23 source datasets — SpatialAgent was already evaluated on a curated subset
- Tools like DepthAnythingV2, object detection, 3D distance cover tasks: absolute distance, object counting, relative distance

## Friction Points

1. **Video input handling.** VSI-Bench is video-only (egocentric walkthroughs). SpatialAgent expects images/multi-frame. Need a frame sampling step.

2. **Task coverage gaps.** SpatialAgent's 12 tools don't cover all 8 VSI-Bench tasks:
   - Room size — no room dimension estimation tool
   - Route planning — no navigation/path-planning tool (requires multi-step spatial reasoning)
   - Appearance order — temporal/spatiotemporal task with no corresponding tool

3. **Core bottleneck remains.** VSI-Bench shows 71% of errors are egocentric-allocentric transformation failures. SpatialAgent has camera extrinsics and depth tools, but the *reasoning* about ego-allo transformation is still delegated to the VLM (planner/summarizer). Tools externalize perception, not the transformation logic.

## Implication for Our Research

This gap — tools handle perception but reasoning stays in the VLM — is exactly what our agent layer aims to fix. Externalizing the ego-allo transformation as explicit geometric computation could address VSI-Bench's hardest failure mode.
