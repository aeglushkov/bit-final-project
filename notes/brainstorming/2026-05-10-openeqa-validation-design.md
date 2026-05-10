# OpenEQA Validation Harness — Design

**Date:** 2026-05-10
**Author:** aleksandr (with Claude)
**Status:** Design — pending implementation plan

## Motivation

The eva-eval experiment ports Embodied VideoAgent (EVA) to VSI-Bench but with open-source backbones (Qwen2.5-7B planner + InternVL2-8B VLM) instead of GPT-4o, and gets ~25 overall — well below the ~30–32 the paper reports for the GPT-4o variant on related benchmarks. We don't yet know whether the gap comes from:

- the open-source backbone substitution (expected to cost some points), or
- the MASt3R preprocessing (vs. the paper's ground-truth or DUSt3R poses), or
- a bug in the eva-eval pipeline (agent, tools, memory, or VSI-Bench-specific code).

The cleanest way to disentangle these is to run the same eva-eval pipeline on a benchmark from the EVA paper itself, where the paper publishes a comparable open-source-backbone number. **OpenEQA fits exactly:** the paper reports E-VideoAgent + InternVL2-8B with Habitat ground-truth poses → 41.2 ALL on the hard subset; with DUSt3R-estimated poses → 40.0 ALL.

If our pipeline matches that range, the agent/perception/memory stack is fine and the VSI-Bench gap is VSI-specific (preprocessing or VSI-Bench question handling). If we're significantly below, the bug is in the shared pipeline. Either outcome eliminates a hypothesis.

## Constraints (locked during brainstorming)

| Constraint | Choice |
|---|---|
| Depth/pose source | Habitat ground-truth (use OpenEQA's pre-rendered HM3D RGB-D-pose tuples — no MASt3R) |
| Subset scope | HM3D-only, ~50 stratified-by-category questions |
| Grading | Open-source LLM judge (default: Qwen2.5-7B reused as judge); hooks to swap in GPT-4o later for paper-faithful re-grade |
| Code organization | Bolt-on: new `eval/openeqa.py` + new preprocess module + new scripts. No churn to existing VSI-Bench code. |
| Disk budget | 20–100 GB on server. Per-episode footprint ~30–100 MB long-term, ~1–2 GB peak scratch. Stream-process episodes, drop raw + depth after each. |
| Inspection artifacts | First-class. Every pipeline stage produces a browsable HTML/markdown for human review. Two explicit human gates before final results. |

## Non-Goals

- Reproduce the paper's exact 41.2 ALL number. The judge differs; the subset is small (~50 questions has ~7-pp binomial noise); this is a debug signal, not a reproduction.
- Generalize to ScanNet or full OpenEQA in this iteration (designed to extend, not chosen now).
- Refactor the existing VSI-Bench code. The bolt-on choice protects the current state.
- Compare across multiple VLMs / planners. Same default stack as the user's VSI-Bench run.

## Architecture

```
OpenEQA HM3D episodes (RGB, depth, pose tuples)
        │
        ▼
┌──────────────────────────────────┐
│ preprocess/openeqa_hm3d.py       │   one episode  →  one cache dir
│   adapt_episode(episode_id)      │   in the same schema as MASt3R produces
└─────────────┬────────────────────┘
              ▼
   cache/openeqa_hm3d/<episode_id>/
              {frames/, depth/, intrinsics.json, poses.npy, meta.json}
              ▼
┌──────────────────────────────────┐
│ scripts/02_build_memory.py        │   UNCHANGED — reuses existing module
└─────────────┬────────────────────┘
              ▼
   cache/openeqa_hm3d/<episode_id>/memory.pkl
              ▼
┌──────────────────────────────────┐
│ eval/openeqa.py + 07_run_openeqa │
│   load Qs, agent loop, save predictions JSONL
│   AgentContext / build_agent / six tools / ReAct prompt — REUSED
└─────────────┬────────────────────┘
              ▼
┌──────────────────────────────────┐
│ scripts/08_grade_openeqa.py       │   separate step — re-grading is cheap
└─────────────┬────────────────────┘
              ▼
   results/openeqa_hm3d_dev50_graded.jsonl + .summary.json
```

Reuse vs new:

| Component | Status |
|---|---|
| `AgentContext`, six tools, `react_vqa.txt` | reused verbatim (dataset-agnostic) |
| `02_build_memory.py`, `eva_eval/preprocess/memory.py` | reused verbatim |
| Vocabulary `config/detection_classes.txt` | reused, possibly extended |
| LLM clients, `config/models.yaml` | reused |
| `agent/agent.py` | one-line change: add `return_intermediate_steps=True` flag |
| `eval/openeqa.py` | NEW |
| `preprocess/openeqa_hm3d.py` | NEW |
| `scripts/0{5,6,7,8}_openeqa_*.py` | NEW |
| `scripts/inspect_*.py` (4 inspectors) | NEW |
| `eva_eval/debug/render.py` (HTML/plot helpers) | NEW |
| `scripts-remote/1{1..7}_openeqa_*.sh` | NEW |

## Data Acquisition

### Sources

1. **Questions JSON** — `facebookresearch/open-eqa` GitHub at `data/open-eqa-v0.json` (~5 MB; ~1.6k questions across HM3D + ScanNet). Filter to HM3D entries.
2. **Pre-rendered HM3D episodes** — RGB + depth + pose frame tuples per episode referenced by `episode_history`. Whole HM3D set ~30 GB; we download only sampled episodes (≪1 GB).

### Open assumption (must verify before coding)

The OpenEQA repo's `data/README.md` documents the pre-rendered episode bundle. **First implementation step is to clone the repo and verify the download URL is publicly accessible.** Two outcomes:

- **Pre-rendered bundle exists and is downloadable** → straightforward: download per-episode, adapt, build memory.
- **Bundle missing or private** → fallback: install Habitat-sim + HM3D scans (much heavier — 100+ GB scenes + Habitat env). Plan should fail fast and surface this rather than silently switching to the heavier path.

### Selective download for the 50-question sanity check

1. Download questions JSON.
2. Filter to HM3D, group by `episode_history`.
3. Stratified-sample 50 questions by `category` → identifies ≤50 unique episodes (write `sampled_50.json` for reproducibility).
4. Download only those episodes from the OpenEQA bundle (per-prefix download).

### Cache layout on the server

```
<eva-eval-cache-root>/openeqa_hm3d/
  questions.json                      # filtered HM3D questions
  sampled_50.json                     # the stratified subset we evaluate
  episodes_raw/                       # downloaded RGB-D-pose tuples — TRANSIENT
    <episode_id>/{rgb/, depth/, pose/}
  <episode_id>/                       # processed cache (parallels VSI-Bench)
    frames/{i:06d}.jpg
    depth/{i:06d}.npy                 # TRANSIENT — deleted after memory build
    poses.npy
    intrinsics.json
    meta.json
    memory.pkl
    _inspect/                         # debug artifacts (preprocess.html, memory.html, ...)
```

### Disk hygiene (per-episode stream-process)

For each sampled episode, in order:

1. Download `episodes_raw/<id>/`.
2. `adapt_episode(...)` → writes `frames/`, `depth/`, `poses.npy`, `intrinsics.json`, `meta.json`.
3. Run paper's `ObjectMemory.process_a_frame` over all frames → `memory.pkl`.
4. Delete `episodes_raw/<id>/` and `<id>/depth/`.
5. Move to the next episode.

Long-term per-episode footprint: ~30–100 MB (dominated by `memory.pkl`'s CLIP/DINOv2 features). 50 episodes: ~3–5 GB long-term, ~10 GB peak scratch.

### Verification before declaring data ready

Implementation phase must validate (one-shot, on the first preprocessed episode):

1. RGB count == depth count == pose count.
2. Pose convention is OpenCV cam2world. Habitat→OpenCV transform applied uniformly. **Tested via `inspect_preprocess.py`'s reprojection self-check** (Section "Inspection — preprocess.html").
3. Depth in meters (HM3D Habitat-rendered native unit).
4. Intrinsics match: `fx = fy = W / (2·tan(fov/2))`, `cx = W/2`, `cy = H/2`.

## Preprocessing Module

`eva_eval/preprocess/openeqa_hm3d.py`. Single-purpose, ~120 lines.

### Public API

```python
def adapt_episode(
    episode_raw_dir: Path,    # downloaded RGB + depth + pose tuples
    out_dir: Path,            # where to write cache schema
) -> dict:                    # returns the meta dict
    """One episode → one cache dir compatible with eva_eval/preprocess/memory.py."""
```

### Behavior

1. Read `rgb/<i>.png` (or whatever extension OpenEQA ships), `depth/<i>.npy` (or 16-bit png), `pose/<i>.txt` or `poses.json`. Exact format confirmed in implementation phase.
2. Apply Habitat→OpenCV pose conversion (one fixed 4×4 transform). Habitat is `+Y up, -Z forward`; OpenCV is `+Y down, +Z forward`.
3. Re-encode RGB as JPEG q=85 → `frames/<i:06d>.jpg`.
4. Save depth as float32 metric meters → `depth/<i:06d>.npy`.
5. Build `intrinsics.json` from OpenEQA's stored FOV.
6. Save `poses.npy` as `(N, 4, 4) float32`.
7. Write `meta.json` with `{video, fps, n_frames, timestamps, source: "openeqa_hm3d"}`. Timestamps are evenly spaced placeholders (the paper's `process_a_frame` only uses them as keys, not for actual timing).

### What it does NOT do

- No video decoding (frames already exist).
- No SfM (poses already known).
- No frame subsampling (disk budget allows native rate).
- No coordinate-system magic beyond the single Habitat→OpenCV transform.

## Inspection & Debug Artifacts

Each pipeline stage produces a browsable artifact. All inspectors are cache-format-agnostic (work on both OpenEQA and existing VSI-Bench caches), live in `scripts/inspect_*.py`, and write static HTML/markdown into `<cache_dir>/_inspect/`. No notebook server, no live viewer — just `scp` and open in a browser.

Decoupled from the pipeline scripts so they can be re-run without re-running preprocessing.

### inspect_preprocess.py `<cache_dir>` → `_inspect/preprocess.html`

- Frame strip thumbnails at 0%, 25%, 50%, 75%, 100%
- Same five frames with depth colorized (`inferno`); depth range printed (catches mm-vs-m unit errors)
- Camera trajectory 2D top-down plot (x/z positions, orientation arrows). Catches drift, jumps, missing loop closures.
- **Reprojection self-check** — pick a depth pixel from frame 0, lift to 3D world point, project into all other frames, draw a red dot. Stable dot = poses correct. Wandering dot = pose convention bug. **This is the test for the Habitat→OpenCV conversion.**
- Header table: `n_frames`, `fov`, image size, depth min/max/mean, source dataset.

### inspect_memory.py `<cache_dir>` → `_inspect/memory.html`

The most-requested view. Verifies detection + 3D lifting + memory accumulation.

- **Object catalog table**: every object with `id`, `category`, `n_frames_visible`, `volume`, `state`, plus a thumbnail of the best frame with the 3D bbox rendered on it (uses `AgentContext.render_object_bbox`, already exists). Sorted by `n_frames_visible` descending.
- **Per-frame visualization**: every Nth frame (default N=10) with all visible objects' 3D bboxes projected and labeled `id: category`. Saved as `_inspect/frame_NNNN.jpg` and embedded in the HTML.
- **Category histogram**: detection counts per YOLO-World class. Catches vocabulary mismatch.
- **Auto-flagged sanity warnings:**
  - 0 objects → memory build broken
  - <5 categories → vocabulary too narrow
  - >50% of objects with `n_frames_visible == 1` → re-ID failing
  - All volumes >100 m³ or <0.001 m³ → unit mismatch in depth

### inspect_agent_trace.py `<jsonl> <question_id>` → markdown file

Full ReAct trace for one question:

- Question, ground truth, prediction, score
- Numbered sequence of (Thought N / Action N / Action Input / Observation N) tuples
- Embedded image when action is `frame_VQA` or `object_VQA` — the actual frame the VLM saw, with bbox if applicable
- Final Answer vs Ground Truth at bottom, with judge rationale if available

Requires `intermediate_steps` capture (one-line change in `agent/agent.py`).

### inspect_grading.py `<graded_jsonl>` → `<output>.inspect.html`

- Per-category aggregate scores
- Worst-10 predictions per category: question / GT / prediction / score / judge rationale
- Best-10 predictions per category
- Histogram of judge scores (1–5 distribution)

### Quick-win: run inspectors on existing VSI-Bench caches

Inspectors only depend on the cache schema, which the user's VSI-Bench data already follows. Once `inspect_preprocess.py` and `inspect_memory.py` exist (plan Phase 1, ~1 day), they can be run against existing VSI-Bench caches to surface the 25-score bug *before* OpenEQA setup completes. Highest-EV part of the plan.

## Evaluation Module

`eva_eval/eval/openeqa.py`. Parallels `eval/vsibench.py` 1-to-1.

### Question loader

```python
def load_openeqa_questions(
    questions_json: Path,        # data/open-eqa-v0.json
    dataset: str = "hm3d",       # "hm3d" | "scannet" | "all"
    limit: int | None = 50,
    stratified: bool = True,
    seed: int = 42,
) -> list[dict]:
```

OpenEQA per-question schema:

```json
{
  "question_id": "...",
  "episode_history": "hm3d-v0/<episode_id>",
  "question": "...",
  "answer": "...",            // gold answer (free-text)
  "category": "object_recognition" | "spatial_reasoning" | ...
}
```

Filter `episode_history.startswith("hm3d-v0/")`, stratify by `category` using existing `eva_eval/eval/sampler.stratified_indices`. Save the chosen indices to `cache/openeqa_hm3d/sampled_50.json` for reproducibility.

### Question formatting

Open-ended QA — single concatenated prompt, no MCA/NA branching:

```python
OPENEQA_PRE_PROMPT = "These are frames from an indoor scene exploration video."

def format_question(q: dict) -> str:
    return f"{OPENEQA_PRE_PROMPT}\n{q['question']}"
```

No "answer with a single number" or "answer with the option letter" — OpenEQA expects free-text answers.

### Episode → cache_dir mapping

```python
def episode_cache_dir(cache_root: Path, episode_history: str) -> Path:
    # "hm3d-v0/00000-kfPV7w3FaU5" → <cache_root>/openeqa_hm3d/00000-kfPV7w3FaU5
    return cache_root / "openeqa_hm3d" / episode_history.split("/", 1)[1]
```

### Run loop

Mirrors `vsibench.py:run`. JSONL row schema:

```json
{
  "id": "...",
  "episode_id": "...",
  "category": "...",
  "question": "...",
  "ground_truth": "...",
  "prediction": "...",
  "intermediate_steps": [...]    // when capture_trace=True (default)
}
```

**No `score` field at this stage.** Grading is a separate command.

### LLM-as-judge grader (separate entry point)

`scripts/08_grade_openeqa.py` — reads predictions JSONL, calls judge, emits graded JSONL with `score` (1–5) and `judge_rationale` fields added.

OpenEQA official grading prompt (paper appendix):

```
You are an AI assistant who will help me evaluate the response given the question
and the correct answer. To mark a response, output a single integer between 1 and 5.
5 = response perfectly matches the answer.
1 = response is completely incorrect.
Question: {question}
Correct Answer: {answer}
Response: {response}
Output a single integer:
```

Default judge: Qwen2.5-7B (open-source). Swappable via `--judge` flag (`--judge gpt-4o` to re-grade for paper-faithful comparison later).

### Aggregation

```python
def aggregate(rows: list[dict]) -> dict:
    """Returns {overall: float, per_category: {cat: float}, n_questions: int}."""
```

**C-score normalization:** `100 * (mean_score - 1) / 4`. So 5 → 100, 1 → 0. Matches OpenEQA paper format.

### What's not in this module

- No second-pass refinement / self-consistency.
- No question-type-specific prompt engineering.
- No multi-judge ensemble (hooks exist to add later).

## Runbook

### New Python entry points (`experiments/eva-eval/scripts/`)

| Script | Purpose |
|---|---|
| `05_download_openeqa.py` | Clone/pull `facebookresearch/open-eqa`, copy questions JSON, verify episode bundle download URL. Errors out clearly if bundle isn't accessible. |
| `06_preprocess_openeqa.py` | For a list of `episode_history` IDs (default: read `sampled_50.json`): download → adapt → build memory → cleanup raw + depth → next. Streams. Idempotent. |
| `07_run_openeqa.py` | Run agent over sampled questions. Mirrors `03_run_vsibench.py`. `--limit`, `--resume`, `--planner`, `--output`, `--capture-trace`. |
| `08_grade_openeqa.py` | Read predictions JSONL, call judge, emit graded JSONL. `--judge` flag picks model. |
| `inspect_preprocess.py` | Section 4.1 inspector. |
| `inspect_memory.py` | Section 4.2 inspector. |
| `inspect_agent_trace.py` | Section 4.3 — args: `<jsonl> <question_id>`. |
| `inspect_grading.py` | Section 4.4 — args: `<graded_jsonl>`. |

### New shell wrappers (`experiments/eva-eval/scripts-remote/`)

Mirror existing convention (`08_dev500.sh`, `09_full_eval.sh`).

| Shell script | What it does |
|---|---|
| `11_openeqa_setup.sh` | Calls `05_download_openeqa.py`. |
| `12_openeqa_sample.sh` | Stratified-sample 50 questions, write `sampled_50.json`. |
| `13_openeqa_preprocess.sh` | Read `sampled_50.json`, run `06_preprocess_openeqa.py`. |
| `14_openeqa_inspect_first.sh` | Run `inspect_preprocess.py` + `inspect_memory.py` on first preprocessed episode. **Stop and inspect.** |
| `15_openeqa_run.sh` | Agent over the 50 questions. |
| `16_openeqa_grade.sh` | Grade with default Qwen2.5 judge. Optional 2nd arg swaps judge. |
| `17_openeqa_inspect_results.sh` | Run `inspect_grading.py` on graded output. |

Each script: `set -euo pipefail`, sources existing `_env.sh`, prints "==> done. Next: bash scripts-remote/...sh" at the end.

### Recommended execution order with stage gates

```
11_openeqa_setup.sh         [verify download URL accessible]
       ▼
12_openeqa_sample.sh        [eyeball sampled_50.json]
       ▼
13_openeqa_preprocess.sh
       ▼
14_openeqa_inspect_first.sh ★ HUMAN GATE ★
                            "do detections look right?
                             do bboxes land on objects?
                             does trajectory make sense?"
       ▼
                            (optional: also run inspect_memory.py
                             against existing VSI-Bench cache)
       ▼
15_openeqa_run.sh
       ▼
inspect_agent_trace.py      [eyeball 2-3 traces]
       ▼
16_openeqa_grade.sh
       ▼
17_openeqa_inspect_results.sh ★ HUMAN GATE ★
                              "compare to paper's 41-43 ALL.
                               look at worst-10 — is judge fair?
                               look at best-10 — false positives?"
```

The two human gates (★) are deliberate. Don't skip them.

### What's not in the runbook

- No all-in-one orchestrator script (would defeat the gated structure).
- No CI hooks.
- No multi-VLM matrix (follow-up work).

## Acceptance Criteria

The harness is "done" when:

1. `13_openeqa_preprocess.sh` produces a cache for every sampled episode without errors.
2. `14_openeqa_inspect_first.sh` opens cleanly in a browser; rendered 3D bboxes visibly land on objects in their frames.
3. `15_openeqa_run.sh` produces a predictions JSONL with all 50 questions answered (errors logged but allowed) and `intermediate_steps` populated.
4. `16_openeqa_grade.sh` produces graded JSONL with C-scores in [0, 100] for every row.
5. `17_openeqa_inspect_results.sh` produces per-category breakdown + worst-/best-10 examples.

**Not promised:** that the score will match the paper's 41.2 ALL. Different judge, small subset (~7-pp binomial noise), open-source backbone. **Decision rule for next steps:**

- Score in 30–50 range with sensible per-category distribution → agent stack is working; VSI-Bench gap is VSI-specific (preprocessing or VSI question handling). Investigate VSI-Bench code.
- Score <20 or any category at 0% → bug in shared pipeline (agent, tools, memory, or preprocessing). Use inspectors to localize.

## Risks and Open Items

| Risk | Mitigation |
|---|---|
| Pre-rendered HM3D bundle isn't publicly downloadable | First implementation step verifies this. Fail-fast with clear message about Habitat-sim fallback. |
| Habitat→OpenCV pose conversion is wrong | `inspect_preprocess.py` reprojection self-check catches this on the first episode, before scaling to all 50. |
| YOLO-World vocabulary doesn't match HM3D content | Inspector flags it (<5 categories warning). Easy to extend `config/detection_classes.txt`. |
| Open-source judge is too lenient or too strict | Inspector shows worst-/best-10 with judge rationale. If clearly biased, swap to GPT-4o (one flag) for a few-dollar re-grade. |
| 50 questions is too small to draw conclusions | Decision rule above accommodates noise; if borderline, scale to ~200 questions (~30 GB total — still within budget). |
| OpenEQA per-question schema differs slightly from what's documented | Implementation phase confirms by reading actual JSON before writing the loader. |

## Estimated Effort

- Phase 1 (debug-first): inspectors + agent trace capture, ~1 day. Yields debug signal on existing VSI-Bench cache immediately.
- Phase 2: preprocessing + grading + run loop, ~2 days.
- Phase 3: shell wrappers, end-to-end run, ~0.5 day.
- Total: ~3.5 person-days plus compute time for preprocessing + agent runs.
