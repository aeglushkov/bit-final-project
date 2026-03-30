# Are Tools "Cheating"? What Should MLLMs Do Internally vs. Externally?

**Date:** 2026-03-30

## The Question

SpatialAgent uses 12 specialized tools. Isn't that cheating? Shouldn't MLLMs calculate distances etc. natively?

## Two Schools of Thought

### 1. "Pure model" view — MLLMs should do it all internally
- Tests whether the model has truly learned spatial understanding
- Problem: models trained on 2D image-text data have no ground-truth 3D supervision
- Asking for metric output ("3.2 meters") from pixels is an ill-posed problem (2D→3D ambiguity)

### 2. "Tool-augmented" view — MLLMs orchestrate specialized tools
- VLMs are great for recognition, bad for metric geometry
- Specialized models (DepthAnything, RAFT) are trained with actual 3D supervision
- Mirrors how humans work — we use rulers, maps, GPS for precise spatial tasks
- VLM's job: understanding *what* to compute. Tools: *how* to compute it.

## Key Insight: The Missing Middle

SpatialAgent externalizes **perception** (depth, flow, detection) but still relies on VLM for **reasoning** (interpreting tool outputs, spatial transformations). The 71% ego-allo transformation errors from Thinking in Space happen at the reasoning level, not perception.

What's actually needed:
- **Tools for perception** (depth, pose, segmentation) — legitimate, specialized skills
- **Tools for reasoning** (ego-allo transformation, geometric computation) — this is what's missing
- **Pure MLLM metric geometry** from pixels — unreasonable given training data

## The Real Research Question

Not "tools or no tools" but "which cognitive functions should be externalized vs. learned end-to-end?" Our research direction targets the reasoning gap specifically.
