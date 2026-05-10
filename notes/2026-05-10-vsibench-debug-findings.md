# VSI-Bench post-fix debug — findings log (2026-05-10)

Living log of what we discover while investigating why VSI-Bench scores are still low after the hfov fix in commit `620ff9a`. Branch: `experiments/openeqa-validation`.

## Context

- Original VSI-Bench score with EVA paper's pipeline + open-source models (Qwen2.5-7B planner + InternVL2-8B VLM) on dev500: **25.30**.
- Commit `620ff9a` (2026-05-10) fixed a units bug — `mast3r.py` was writing `intrinsics["fov_h"]` in radians, but the paper's `frame2d_to_camera3d_transformation` expects degrees. Effective `fx` was ~66× too large; all 3D AABBs collapsed near the camera trajectory.
- Post-fix re-run on a 100-question subset (`results/subset_fixed.jsonl`): **30.68**.
- Question: where are the remaining ~17 points (vs paper's 47-ish on OpenEQA)?

## Inspector pass (Phase 1 of the OpenEQA validation plan)

Ran `inspect_preprocess.py` and `inspect_memory.py` against four scenes spanning the post-fix score distribution: `bcd2436daf` (0%), `scene0050_01` (7.8%), `578511c8a9` (33.3%), `42897688` (75%).

What's healthy ✓:
- AABB sizes are correct ranges post-fix (chairs 0.5–1 m³, beds 8–12 m³).
- Trajectories: 35–100 m total length, 4–9 m room spans.
- Depth: 0.5–8 m, 100% valid pixels per frame.
- Iteration ceiling almost never hit: 1/100 questions in subset_fixed.

What's not healthy ✗:
- **47–51% of objects are visible in only one frame** in every scene I checked. Same physical chair gets multiple IDs across frames — re-ID over-segmentation under noisy MASt3R poses.
- **Reprojection self-check** drift over 3-frame hops is 175–240 px median in 512×384 images — within plausibly-noisy-but-not-catastrophically-bad range; comparable across worst and best scenes, so not the discriminator.
- **Numeric tasks systematically wrong**: distances PRED < GT, room sizes PRED << GT, sometimes object sizes wildly off.

## Captured agent traces (Phase 2)

Re-ran 20 worst-failing questions stratified across all 10 task types with `return_intermediate_steps=True` (the flag added in this branch). Three concrete failure modes, each backed by a real trace.

### Cause 1 — SQL schema cannot answer dimension/distance questions

The Objects table is `(object_id, category, volume)`. No bbox extents, no positions exposed.

**Trace 296** (object_size_estimation): *"longest dimension of the toilet in cm"*, GT=98.

```
Thought: I need to find the toilet object and measure its longest dimension.
Action: retrieve_objects_by_appearance("toilet")  →  finds toilet object_id 1

Thought: From the observations, I need to identify the toilet and measure its
dimensions. However, the dimensions are not provided. I will proceed with the
available data and assume the longest dimension is the height of the toilet.
Action: SELECT MAX(volume) FROM Objects WHERE category = 'toilet'  →  0.296

Final Answer: 296
```

The agent literally said *"the dimensions are not provided"* and stripped the decimal off `0.296` to produce `296`. The volume the agent retrieved was physically correct (~0.3 m³ for a toilet bowl). The pipeline had the right geometry. The schema just didn't expose what the question asked about.

### Cause 2 — VLM falls back to vibes-based numeric estimation

When the SQL can't answer, the agent asks the VLM. The VLM is bad at metric estimation.

**Trace 4679** (object_abs_distance): *"distance between the mirror and the plant in m"*, GT=1.2.

```
Action: frame_localization("mirror and plant")  →  [66, 107, 65, 67, 108]
Action: frame_VQA("What is the distance between the mirror and the plant in meters?", 66)
  →  "The distance is not directly measurable from the image, but it appears
     to be approximately 2 meters."
Final Answer: 2
```

InternVL2-8B is essentially eyeballing the distance. Off by 67% on this one. Same pattern dominates `room_size_estimation` (PRED 10 m² vs GT 18 m²; PRED 2.5 vs GT 5.8) — the agent calls `frame_VQA("size of this room?")` and gets back qualitative-grade answers.

### Cause 3 — Vocabulary fragmentation + lossy retrieval kill counting

YOLO-World is open-vocabulary: given the same 200-class detection list, it picks one label per detection. Visually-similar physical objects can be assigned DIFFERENT labels.

**Scene `f3d64c30f8`**, question 2595: *"How many crate(s) are in this room?"*, GT=9.

Counts in memory by category for container-shaped detections:
```
crate:   1
box:     9
bin:     9
basket:  7
bucket:  1
```

The same 9 physically-crate-like containers got spread across 5 different category labels.

```
Action: retrieve_objects_by_appearance("crate")  →  returns 10 captions including
        "blue plastic bin", "computer monitor", "vacuum cleaner", "brown paper bag",
        and one actual "green plastic crate labeled Gässer".
Action: SELECT COUNT FROM Objects WHERE category = 'crate'  →  1

Final Answer: 0   (the agent rejected the SQL count after the noisy retrieval)
```

`SELECT COUNT(*) WHERE category='crate'` returns 1 because that's how many detections happened to land on the literal word "crate". The other 8 actual crates are stored under labels the agent's WHERE clause can't see.

#### Visual evidence: the labeling is unreliable in both directions

A frame from `f3d64c30f8` — three container-shaped objects, three different labels:

![Container labeling mismatch — frame from f3d64c30f8](_assets/2026-05-10-vsibench-label-mismatch.jpg)

**`box#8`** (top-left, blue): the bbox is around a CALYCLEAN vacuum cleaner. The detector mislabeled it as "box".
**`box#23`** (centre, blue): a red box of pencils on a table. Reasonable.
**`bin#6`** (right, green): a yellow plastic container next to a chair. Reasonable.

A frame from `578511c8a9` — what's tagged as a chair:

![Mislabeled "chair" — frame from 578511c8a9](_assets/2026-05-10-vsibench-reid-overcount.jpg)

**`chair#145`**: the bbox surrounds a wooden crate / equipment box on a desk, not a chair.

So the category labels can be wrong in two ways simultaneously:
1. **Synonym fragmentation** — visually-equivalent objects get *different* labels (`box` vs `bin` vs `basket` for the same physical type).
2. **Outright mislabels** — the open-vocabulary detector picks an inappropriate label for an unfamiliar object (vacuum cleaner → "box", crate on a desk → "chair").

This means `WHERE category='X'` is fundamentally fragile for counting: an exact match misses (1) and over-counts (2).

## Experiment: extended SQL schema + computed-answer tools

Hypothesis: the ROOT cause for many numeric failures is the schema gap, not the perception. Test: extend the schema to expose what's already in memory, and add tools that compute the right thing.

Implementation (committed in `7441f0c`, `c21ec0d`):
- `Objects` table extended with `min/max xyz`, `cx cy cz`, `length_m`, `width_m`, `height_m`, `longest_edge_m`.
- New tools, opt-in via `build_agent(..., extended_schema=True)`:
  - `get_object_dimensions(object_id)` → "L=X cm, W=Y cm, H=Z cm"
  - `get_distance(id_a, id_b)` → closest-point distance in m
  - `estimate_room_size("")` → convex-hull and bbox-span estimates in m²
- New prompt template `react_vqa_extended.txt` documents the new schema and steers numeric questions toward the computed-answer tools.

### Result on the same 20 worst-failing questions

| Run | Mean score | Note |
|---|---:|---|
| A: original basic | 0.000 | (these are picked as worst-failing) |
| B: basic re-run with traces | 0.164 | run-to-run variance |
| C: extended schema | 0.223 | +0.06 vs B |
| D: extended + count_objects_matching tool | **0.414** | +0.25 vs B (2.5× over basic) |

Per-task delta D vs B:
- `object_size_estimation`  +0.82
- `object_rel_direction_hard`  +0.50
- `object_rel_direction_easy`  +0.50
- `obj_appearance_order`  +0.50 (likely sampling variance)
- `object_abs_distance`  +0.27
- `object_counting`  +0.05 (modest — the tool's threshold needs tuning)
- `route_planning`  0.00
- `object_rel_distance`  0.00
- `object_rel_direction_medium`  0.00
- `room_size_estimation`  -0.14 (convex-hull overshoots multi-room scenes)

**Smoking-gun re-trace of question 296** (toilet "longest dimension in cm"):
- Basic schema: PRED `296`, score `0.0` (volume 0.296 m³ misinterpreted as 296 cm).
- Extended schema: agent calls `get_object_dimensions(1)` → returns `length=80.9 cm width=33.1 cm height=110.7 cm longest_dimension=110.7 cm`; PRED `110.7`, score **0.82**. Remaining ~13% error is bbox-perception inaccuracy, not tool-use.

## Decision: keep extended schema, drop count_objects_matching for now

The new counting tool's threshold (currently 0.20) is novel — there's no analogous knob in the paper to anchor on. On the test scene it overshot (q2595: PRED=40 vs GT=9 with similarity-0.20 retrieval). Picking a robust threshold needs a sweep, which is a separate experiment.

For the next dev500 run, **keep the extended schema** (geometry + dimensions/distance/room_size tools) and **revert `count_objects_matching`** so counting stays on exact-category-match (matching the paper's design). The dev500 results give a clean signal of how much the schema-gap fix alone is worth on the full distribution.

## Open items / next experiments

1. **Counting**: sweep `count_objects_matching` thresholds on a held-out set; or add per-question synonym hints; or pivot to a VLM-verified count.
2. **Re-ID**: 47–51% single-frame fraction in every scene. Paper's thresholds (Visual>0.45, IoU>0.2) were tuned for GT poses — relaxing them under MASt3R-noisy poses might pay off.
3. **Room size estimation**: convex-hull overshoots multi-room scenes; bbox-span undershoots. Need a smarter heuristic.
4. **`get_distance` prompt example**: agent stumbled on the tuple format (tried passing `("refrigerator", "stove")` strings instead of int IDs). Tighten the prompt example.
5. **Tool-input parsing variance**: some questions hit `Agent stopped due to iteration limit` because the agent never recovered from a parse error. Could add a smarter parser or a recovery hint.

## Artifacts

On the server (`~/github-projects/bit-final-project/`):
- `results/subset_fixed.jsonl` — original post-fix 100-question subset (30.68 overall)
- `results/_trace_picks.json` — the 20 worst-failing questions
- `results/_trace_rerun.jsonl` — same 20, basic schema, with traces
- `results/_trace_rerun_extended.jsonl` — same 20, extended schema, no counting tool
- `results/_trace_rerun_extended2.jsonl` — same 20, extended schema + counting tool
- `cache/vsibench/{scene}/_inspect/{preprocess,memory}.html` — per-scene inspector HTML

In repo (under `notes/_assets/`): the two annotated screenshots above.
