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

## dev500 result (2026-05-11): extended schema is a *net regression*

Ran the full 500-question stratified-sample with `--extended-schema` end-to-end (3.5 hr in tmux), comparing against the existing `dev500.jsonl.summary.json`. Both runs use the same 500 question IDs; the basic dev500 was on the pre-hfov-fix cache (May 7), so the comparison conflates the hfov fix with the schema/tools change.

| task | basic (pre-fix) | extended (post-fix) | Δ |
|---|---:|---:|---:|
| **overall** | 25.30 | **22.23** | **−3.07** |
| object_counting | 24.18 | 20.91 | −3.27 |
| object_abs_distance | 17.64 | 20.00 | +2.36 |
| **object_size_estimation** | 31.64 | **7.99** | **−23.64** ← collapse |
| **room_size_estimation** | 13.64 | **37.64** | **+24.00** ← big win |
| object_rel_distance | 20.00 | 20.00 | 0.00 |
| object_rel_direction | 35.33 | 37.33 | +2.00 |
| route_planning | 30.00 | 18.00 | −12.00 |
| obj_appearance_order | 30.00 | 16.00 | −14.00 |

Per-task deltas dwarf the overall delta because they cancel.

### What the extended schema actually buys
- **room_size_estimation +24** — `estimate_room_size` (convex hull / bbox span over object centers) is robustly better than VLM eyeballing across 50 scenes.
- **object_abs_distance +2.4** — `get_distance` modestly helps; limited by AABB inflation/under-segmentation.
- **object_rel_direction +2.0** — noise; new schema added nothing for direction.

### What it breaks
- **object_size_estimation −24** (investigated below — bbox quality + agent heuristic)
- **route_planning −12 and obj_appearance_order −14** — these tasks should still use `frame_VQA`. The new prompt's "for numeric questions use the new tools" instruction is over-rotating the agent away from the VLM path for MCA tasks.

## Why object_size_estimation collapsed — trace investigation

Re-ran the 10 worst object_size_estimation failures with `--extended-schema --capture-trace`. Two stacked failure modes:

### Stage 1 — `retrieve_objects_by_appearance` returns wrong objects when the queried category is outside YOLO-World's vocab

VSI-Bench routinely asks about objects the open-vocabulary detector doesn't reliably catch. CLIP-text retrieval then top-K matches the closest available caption — often something visually unrelated:

| asked for | top retrieval result |
|---|---|
| dishwasher | refrigerator |
| stool | red pot holder / nightstand |
| bathtub | detergent box / showerhead |
| stove | window |
| refrigerator | ceiling light |
| door | gray cabinet |
| ceiling light | thermostat / light switch |

The agent then computes dimensions on the *wrong* object.

### Stage 2 — agent does a ×10 or ×100 "correction" when the bbox number looks small

When `get_object_dimensions` returns a small number (from the wrong object, or from an undersized AABB of the right object), the LLM appends a digit to make it "look reasonable in cm":

| Q (GT in cm) | tool returned (cm) | agent answered |
|---|---:|---:|
| stool (75) | longest=21.9 | **219** (×10) |
| bathtub (135) | longest=8.0 | **80** (×10) |
| refrigerator (183) | longest=105.4 | **1054** (×10) |
| door (210) | longest=37.4 | **3740** (×100) |
| ceiling light (37) | longest=3.7 | **370** (×100) |

The agent has the right *order-of-magnitude* prior on common objects and "fixes" the small bbox value by adding zeros. The result is 2–10× worse than the original VLM eyeball.

### So the schema fix helps where VLM has no prior, hurts where VLM has a strong prior

The basic schema's failure mode was that the agent had no path to a measurement and fell back to `frame_VQA("dimensions in cm?")`. The VLM can't measure, but it has a strong language prior on common-object sizes — toilets ~100 cm, refrigerators ~180 cm. Even when guessing it anchors on the right magnitude.

The extended schema gives the agent a tool whose output is **less reliable than the language prior** for common objects in this pipeline, because:
- many target objects aren't reliably detected (vocab mismatch),
- detected objects are often only seen in 1 frame (so AABB is a slice, not the full object),
- when the AABB is right (e.g. the worst-20 toilet case), the tool wins decisively (0 → 0.82).

## Two underlying root causes (and what to fix)

**A. AABBs are systematically undersized when visibility is low.** Every scene I checked had 47–51% of objects visible in only one frame. Paper's re-ID thresholds (`Visual > 0.45`, `IoU > 0.2`) were tuned for GT poses; under MASt3R-noisy poses, detections of the same physical object don't merge across frames, so the persistent AABB covers one detection mask's depth back-projection — a slice of the real object. The bathtub trace shows it concretely: GT=165 cm, AABB height=79.6 cm (≈0.48× real size).

**B. `retrieve_objects_by_appearance` returns top-K matches unconditionally**, even when the closest match's similarity is low. For queries outside YOLO-World's vocabulary, top-K returns visually unrelated objects with high confidence. The agent then runs the new tools on the wrong object.

**Fixes worth trying (priority order):**
1. **Tighten the extended prompt** so the "use the new tools" instruction is scoped to phrases that match size/distance/room-size question templates only, leaving everything else on the basic flow. Should recover most of the route_planning/obj_appearance_order regressions.
2. **Add a similarity floor** to `retrieve_objects_by_appearance` — if the top result's CLIP-cosine is below some threshold, return "no objects matching that description in this scene" instead of nonsense. Prevents the wrong-object → wrong-dimensions chain.
3. **Suppress `get_object_dimensions` output for low-visibility objects** — if an object is visible in <N frames, the AABB is unreliable; return "insufficient frames" instead of a number, so the agent falls back to the VLM. Same principle as #2 — fail gracefully when the data is bad.
4. **Relax re-ID thresholds** for MASt3R-noisy poses (Visual 0.45 → 0.30, IoU 0.20 → 0.05) and rebuild memory. Would help all tasks, but expensive to rebuild and untested.

## Artifacts

On the server (`~/github-projects/bit-final-project/`):
- `results/subset_fixed.jsonl` — original post-fix 100-question subset (30.68 overall)
- `results/dev500.jsonl` — basic-schema dev500 (pre-hfov-fix; 25.30 overall)
- `results/dev500_extended.jsonl` — extended-schema dev500 (post-hfov-fix; 22.23 overall)
- `results/_trace_picks.json` / `_trace_rerun.jsonl` / `_trace_rerun_extended.jsonl` / `_trace_rerun_extended2.jsonl` — worst-20 traced runs (basic, extended, extended+counting)
- `results/_size_picks.json` — 10 worst object_size_estimation rows from dev500_extended
- `results/_trace_size.jsonl` — same 10 with intermediate_steps captured (the investigation above)
- `cache/vsibench/{scene}/_inspect/{preprocess,memory}.html` — per-scene inspector HTML

In repo (under `notes/_assets/`): the two annotated screenshots above.
