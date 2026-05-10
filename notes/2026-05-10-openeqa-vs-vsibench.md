# OpenEQA vs VSI-Bench — what's actually different

Both benchmarks ask questions about an indoor scene from egocentric video, but they test entirely different parts of the agent. This note pins down the differences.

## Answer format

| Aspect | OpenEQA | VSI-Bench |
|---|---|---|
| Output type | **Open-ended free-text** | **Structured: MCA letter or single number** |
| Example Q | "What is the color of the chair near the desk?" | "What is the length of the longest dimension (length, width, or height) of the toilet, measured in centimeters?" |
| Example GT | "The chair is dark grey." | "98" |
| Grading | LLM-as-judge (GPT-4 scores 1–5, normalized to 0–100 C-score) | Deterministic — exact match for MCA, Mean-Relative-Accuracy for numerics |

OpenEQA's grading is interpretive: a partial answer like *"about three"* when GT is *"three chairs"* might score 4 or 5. VSI-Bench's grading is mechanical: PRED `2.5` vs GT `1.9` runs through `MRA(0.5..0.95)` and you get a fractional score from a fixed formula — no LLM in the loop.

## Question taxonomy

### OpenEQA — 7 categories, all answerable in natural language

- **Object recognition** — "what's on the desk?"
- **Attribute recognition** — "what color is the sofa?"
- **Spatial understanding** — "where is the bedroom?"
- **Object localization** — "where is my book?"
- **Functional reasoning** — "what would I use to write?"
- **World knowledge** — "what room is this?"
- **Object state recognition** — "is the door open?"

Largely tests **"can the agent find and describe things"** — perception-heavy.

### VSI-Bench — 8 task types, strict format requirements

- `object_counting` — integer answer ("How many crate(s)?")
- `object_size_estimation` — number in cm ("longest dimension of the toilet in cm")
- `object_abs_distance` — number in m ("closest-point distance between fridge and stove")
- `object_rel_distance` — multiple choice (A/B/C/D)
- `object_rel_direction` (easy/medium/hard) — multiple choice
- `room_size_estimation` — number in m²
- `route_planning` — multiple choice
- `obj_appearance_order` — multiple choice (sequencing)

Largely tests **spatial reasoning over the scene as a 3D whole** — geometry-heavy. Doesn't care about colors, materials, or world knowledge.

## What the agent has to do

- **OpenEQA** can mostly be answered with `frame_VQA` (let the VLM look at a few frames and describe what's there). The 3D memory is helpful but optional. The paper achieves **47.0 ALL** on the hard subset with this approach.
- **VSI-Bench** punishes VLM-only answers because the VLM can't *measure*. "How many cm is the toilet?" or "how many crates are in the whole scene?" require either:
  - The agent to walk/aggregate across many frames (counting), or
  - Access to 3D bbox geometry (sizes, distances).

  The VLM eyeballing "approximately 2 meters" gets 0–0.18 on MRA.

## Why VSI-Bench is harder for the EVA-paper pipeline

Three reasons that came out of the 2026-05-10 investigation (see `2026-05-10-vsibench-debug-findings.md`):

1. **Strict numeric grading.** PRED 110.7 cm vs GT 98 cm gets 0.82 (mostly correct via MRA); PRED 2 m vs GT 1.2 m gets 0 (off by 67%). A confident-sounding wrong answer scores 0 instead of the partial credit an LLM judge might give.

2. **Schema dependency for size/distance.** VSI-Bench has `object_size_estimation`, `object_abs_distance`, `room_size_estimation` — three task types totalling ~30% of the dataset that *require* 3D geometry. OpenEQA has nothing equivalent. The paper's SQL schema (`object_id, category, volume`) was designed for OpenEQA-style questions and structurally cannot answer "longest dimension in cm" — that's what the 2026-05-10 patch addressed (extended schema + `get_object_dimensions` / `get_distance` / `estimate_room_size` tools).

3. **Counting precision.** VSI-Bench's `object_counting` expects an integer matching a human annotator's count. OpenEQA's "what's on this table" is loose enough to be partially correct. YOLO-World vocabulary fragmentation (crate=1, box=9, bin=9, basket=7 for the same scene's containers) torpedoes VSI-Bench scores but not OpenEQA scores.

## Takeaway

> OpenEQA tests **what** is in the scene; VSI-Bench tests **where, how big, how far, and how many**.

Same data modality (egocentric indoor videos), same perception stack (YOLO-World + CLIP/DINOv2 + 3D AABBs), but the *question types* push entirely different parts of the agent. The EVA paper's tools are well-suited to OpenEQA and structurally undersized for VSI-Bench.

## Related artifacts in this repo

- `notes/2026-05-10-vsibench-debug-findings.md` — the live findings log on VSI-Bench post-fix debugging
- `literature/EmbodiedVideoAgent/analysis.md` §5.2 — paper's OpenEQA results table (47.0 ALL)
- `literature/thinking-in-space/analysis.md` — the paper that introduced VSI-Bench
