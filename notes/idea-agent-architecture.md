# Research Idea: Agent Architecture for Spatial Reasoning

## Origin

Proposed by Diwei Su (2026-03-23) based on his experiments with the "Thinking in Space" paper (VSI-Bench).

## Key Observation

When input is **video + question**, VLMs give correct answers. But when asked to reason based on a **caption (text description) generated from the same video**, answers become inconsistent.

This suggests current VLMs are doing **pattern matching** (directly from pixels to answers), not truly understanding spatial information.

**Su's explanation (2026-03-25):** The root cause is that current video-understanding MLLMs only align multimodal signals through data-driven design, without genuine spatial reasoning. The models have latent spatial capabilities but cannot reason explicitly about space.

## Proposed Direction

Design an **agent architecture on top of current SOTA multimodal models** to unlock their inherent spatial reasoning capabilities.

The agent acts as a **reasoning coordinator** — not just asking VLM questions directly, but:

1. **Decompose** complex spatial tasks into manageable sub-problems
2. **Step-by-step reasoning** — conduct explicit spatial reasoning externally
3. **Multi-turn interaction** — query the VLM multiple times, targeting specific frames and regions
4. **Cross-frame integration** — integrate visual cues across different frames and viewpoints
5. **Guide attention** — direct the model to focus on spatial structures rather than superficial patterns
6. **Build consistent representation** — form a coherent spatial representation from gathered evidence

The key distinction: the agent doesn't treat the VLM as a black-box predictor. It orchestrates the VLM's latent spatial capabilities through structured interaction.

## Why This Might Work

- Paper shows only 8% perception errors but 71% egocentric-allocentric transformation errors
- VLMs are good at seeing objects, bad at reasoning about spatial relationships
- The agent externalizes the reasoning part, using the VLM only for perception
- By guiding the model to focus on spatial structures, the agent can help form consistent spatial representations that the VLM cannot build on its own in a single pass

## Open Questions

- Does pattern matching still affect individual perceptual queries?
- What specific agent design would work best?
- Which tasks from VSI-Bench would benefit most?
- How to build/represent the explicit spatial map?

## Next Steps (suggested by Su, 2026-03-25)

1. **Survey agent frameworks in video understanding** — analyze their development thoroughly; could present findings at group meeting
2. **Look at SpatialScore benchmark** — "SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence" — notably supports **agent-based evaluation protocols**
   - Run basic experiments to test its performance
   - Study its code implementation
