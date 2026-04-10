# VADAR — Analysis

## Relevance to Our Research Direction

VADAR directly validates the core thesis of our project: **VLMs are bottlenecked by spatial reasoning, not perception**. Their oracle experiments (83% CLEVR, 94.4% Omni3D-Bench with perfect vision) prove that externalizing spatial logic into programs works — the gap comes from vision specialist errors, not reasoning failures.

## What VADAR Gets Right

### Dynamic API > Static DSL
The key insight: ViperGPT and VisProg fail on complex spatial queries because their human-defined APIs can't cover the combinatorial space of spatial subproblems. VADAR's agents create functions like `_is_behind`, `_find_closest_object_3D`, `_count_objects_by_attributes_and_position` on-the-fly. This is a compelling argument against hand-crafted DSLs.

### Separation of Perception and Reasoning
VADAR cleanly separates:
- **Perception** → vision specialists (Molmo, SAM2, UniDepth, GPT-4o VQA)
- **Spatial reasoning** → generated Python programs

This aligns with our hypothesis that VLMs should be used for perception only, with reasoning handled externally.

### Training-Free
No fine-tuning needed. Outperforms LEFT (which needs 10K+ supervised samples) on CLEVR. Scales to new domains (Omni3D-Bench) without retraining.

## Limitations and Gaps

### 1. No Egocentric-Allocentric Transformation
VADAR works in **pixel space + monocular depth**. It never builds an allocentric (world-frame) spatial representation. Depth comparisons are done per-pixel (`depth(image, x, y)`), not in a coherent 3D coordinate system. This means:
- No true 3D distance computation (just depth ordering)
- Spatial relations like "behind" are approximated via depth comparison, not geometric reasoning
- Metric estimates (e.g., "how tall is the table in 3D?") rely on scaling heuristics, not proper projective geometry

**Our opportunity:** An explicit ego-to-allocentric transformation layer would enable geometrically correct spatial reasoning rather than depth-comparison heuristics.

### 2. Single-Image, No Video
VADAR operates on single frames. VSI-Bench (our target) requires video understanding — selecting the right frame and reasoning across viewpoints. VADAR's VSI-Bench-img experiment (Table 7) is a controlled subset where the correct frame is pre-selected.

### 3. Vision Specialist Bottleneck
The 30% gap between execution accuracy and oracle accuracy is enormous. The paper acknowledges this but doesn't address it beyond suggesting "improve specialist models." Specific failure modes:
- Molmo/GroundingDINO missing objects or returning wrong locations
- UniDepth monocular depth errors (especially for distant/small objects)
- SAM2 segmentation failures affecting `same_object` and `get_2D_object_size`

### 4. API Generation is Question-Aware
The Signature Agent sees 15 questions (without answers) to decide what functions to create. This means the API is somewhat tailored to the benchmark distribution. On a truly open-ended deployment, the API might miss needed functions.

### 5. Depth ≠ True 3D
`depth(image, x, y)` returns monocular depth estimates, which are:
- Relative, not metric (UniDepth tries to be metric but has errors)
- A single value per pixel, not a full 3D position
- No camera intrinsics used for back-projection

VADAR's "3D reasoning" is really "2D + depth ordering" — a significant simplification.

### 6. Expensive at Inference
Table 6: ~35.7s per question for execution alone (on A100). The Signature + Implementation agents run once (~57.7s total for 10 questions), but per-question execution is slow due to multiple vision specialist calls.

## Architecture Comparison with Our Approach

| Aspect | VADAR | Our Direction |
|---|---|---|
| Spatial representation | Pixel coords + monocular depth | Allocentric 3D scene graph |
| Ego→allo transform | None (implicit via depth) | Explicit transformation layer |
| Reasoning | Generated Python programs | External spatial reasoning module |
| Perception | Multiple specialists (Molmo, SAM2, UniDepth, GPT-4o) | VLM as perception-only |
| Modality | Single image | Video (multi-frame) |
| API | Dynamically generated | TBD |
| Training | None | None (goal) |

## Key Takeaways for Our Work

1. **Program synthesis for spatial reasoning works** — VADAR proves the paradigm. The oracle results (83–94%) show programs are correct; it's the vision that fails.

2. **Dynamic API generation is valuable** — we should consider whether our agent can similarly grow its spatial reasoning toolkit rather than relying on a fixed set of spatial operations.

3. **The depth-as-3D approximation has a ceiling** — VADAR's approach of comparing `depth(x1,y1)` vs `depth(x2,y2)` for spatial relations is fundamentally limited. True 3D reconstruction (even lightweight, e.g., point cloud from depth + camera intrinsics) would unlock more accurate spatial reasoning.

4. **Vision specialist quality is the real bottleneck** — any agent-based approach will face this. We should investigate which perception tasks are hardest and whether a single strong VLM can replace the multi-specialist pipeline.

5. **Omni3D-Bench is a useful benchmark** — 500 challenging real-world spatial queries with human annotations. Worth evaluating our approach on this alongside VSI-Bench.

## Code Structure (authors' release)

```
code/
├── agents/agents.py          # Agent definitions (Signature, Implementation, Test, Program, Execution)
├── engine/
│   ├── engine.py              # Main evaluation loop
│   ├── engine_utils.py        # Utility functions
│   ├── oracle.py              # Oracle execution with ground-truth annotations
│   └── predefined_modules.py  # Base API functions (loc, vqa, depth, same_object, get_2D_object_size)
├── prompts/
│   ├── api_prompt.py          # API generation prompts
│   ├── signature_prompt.py    # Signature Agent prompt
│   ├── program_prompt.py      # Program Agent prompt
│   ├── modules.py             # Pre-defined module definitions
│   └── vqa_prompt.py          # VQA-specific prompts
├── evaluate.py                # Entry point
├── setup.sh                   # Environment setup
└── requirements.txt
```
