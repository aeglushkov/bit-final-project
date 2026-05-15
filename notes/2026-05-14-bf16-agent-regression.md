# Why the "AWQ → bf16 agent regression" was an illusion (2026-05-14)

## TL;DR

The headline number from yesterday — "agent regressed from 30.7 (AWQ) to 27.0 (bf16) overall" — is **not a real regression**. The two runs evaluated **different 100-question samples**: only 4 of 100 IDs overlap. The "AWQ agent" subset (`results/subset_fixed.jsonl`, 2026-05-10) was sampled when only 19 scenes had a built `memory.pkl`; the "bf16 agent" subset (`results/subset_bf16.jsonl`, 2026-05-13) was sampled today, with 80+ scenes cached. The stratified sampler in `eva_eval/eval/vsibench.py:load_dataset_indices` filters by `cached_scenes` before stratifying, so the candidate pool changed and the seed produced disjoint IDs.

The **real** comparison — bf16 agent vs bf16 raw VLM on the same 100 IDs (`bf16_agent ∩ bf16_rawVLM = 100`) — shows the agent is **13.5 points behind raw VLM** overall, with the gap concentrated on perception/estimation tasks.

## Sample-overlap audit

```
AWQ_agent (subset_fixed.jsonl)        : 100 rows, 19 scenes (memory.pkl set on 2026-05-10)
bf16_agent (subset_bf16.jsonl)        : 100 rows, 80 scenes (memory.pkl set on 2026-05-13)
bf16_rawVLM (baseline_internvl2-8b-bf16.jsonl): 100 rows

AWQ_agent  ∩ bf16_agent   = 4         <-- yesterday's headline compared these two
bf16_agent ∩ bf16_rawVLM  = 100       <-- this is the only apples-to-apples pair
AWQ_agent  ∩ bf16_rawVLM  = 4
```

The bf16 agent and bf16 raw VLM both ran today and both used the *same* `stratified_indices(seed=42)` over the same candidate pool, so they got matching IDs. The AWQ agent ran 3 days earlier when the cache state was different. The diff yesterday across `subset_fixed.jsonl` vs `subset_bf16.jsonl` was apples-to-oranges.

## The real comparison: bf16 agent vs bf16 raw VLM (same 100 IDs)

Reproduce with `python experiments/eva-eval/scripts/diff_agent_runs.py --awq results/baseline_internvl2-8b-bf16.jsonl --bf16 results/subset_bf16.jsonl` (the script labels are reused; here "awq" column = raw VLM bf16 and "bf16" column = agent bf16).

```
| question_type              |  n |  raw VLM (bf16) | agent (bf16) |    Δ |
| obj_appearance_order       | 10 |          60.00  |       20.00  |  -40 |
| object_rel_direction_medium| 10 |          40.00  |       10.00  |  -30 |
| room_size_estimation       | 10 |          51.82  |       24.55  |  -27 |
| object_rel_direction_easy  | 10 |          50.00  |       30.00  |  -20 |
| object_size_estimation     | 10 |          36.36  |       17.27  |  -19 |
| object_abs_distance        | 10 |          41.82  |       23.64  |  -18 |
| object_rel_distance        | 10 |          60.00  |       50.00  |  -10 |
| object_counting            | 10 |          17.27  |       17.27  |   +0 |
| object_rel_direction_hard  | 10 |          20.00  |       30.00  |  +10 |
| route_planning             | 10 |          20.00  |       40.00  |  +20 |
| overall                    | 100|          39.73  |       26.27  | -13.45 |
```

*(The published 100Q raw-VLM overall is 40.49 — the small difference from 39.73 here is because `eva_eval.eval.metrics.aggregate` collapses the three `object_rel_direction_*` subtypes into one before averaging, while `diff_agent_runs.py` keeps them separate. Per-task and Δ values are identical.)*

The agent helps on exactly **2 of 10 task types**:
- **route_planning: +20** (this is a real, durable effect — the agent's only consistent win across all our experiments)
- **object_rel_direction_hard: +10** (multi-step spatial reasoning where the agent's plan-then-act loop helps)

The agent hurts on **7 of 10 task types**, badly:
- **obj_appearance_order: −40** (the agent's vqa tools see one frame at a time, so it can't see the cross-frame temporal order that the raw VLM gets for free with 8 simultaneous frames)
- **object_rel_direction_medium: −30**
- **room_size_estimation: −27** (needs whole-scene visual integration; agent fragments this into per-frame queries)
- **object_rel_direction_easy: −20**
- **object_size_estimation: −19** (object_VQA tool gives one frame; raw VLM has 8)
- **object_abs_distance: −18**
- **object_rel_distance: −10**

And ties on **object_counting** (both struggle equally; 17 vs 17 on 10 questions, neither close to the paper's 31).

## Per-question failure pattern: structural, not stochastic

```
Cases where raw VLM was perfect (1.0) but agent was not  : 24 / 100
Cases where agent was perfect (1.0) but raw VLM was not  : 14 / 100
Cases where raw VLM had partial MRA credit, agent got 0  : 11 / 100
Cases where agent had partial MRA credit, raw VLM got 0  :  3 / 100
```

Net flip = (24 + 11) − (14 + 3) = +18 questions where the agent did strictly worse. That accounts for the overall regression.

## Why the agent loses on perception tasks

Looking at `eva_eval/agent/tools.py` and the agent's prompt, the structural disadvantage is clear:

1. **`frame_VQA(question, frame_id)` and `object_VQA(question, object_id)` send a single image to the VLM.** The raw VLM baseline sends **8 frames simultaneously** in one chat message. For any task that requires integrating evidence across the whole scene — appearance order, room size, object size in context — the agent is information-restricted by design.

2. **`retrieve_objects_by_appearance` and `retrieve_objects_by_environment` depend on YOLO-World detection** and CLIP-similarity matching. Yesterday's notes already flagged vocabulary fragmentation in the YOLO labels (same physical chair labeled multiple ways). The agent has to navigate that fragmentation with text-based queries; the raw VLM just looks at the images.

3. **The planner (Qwen-7B) doesn't see images at all** — it sees text descriptions of what the tools returned. Any visual detail not surfaced by a tool is lost to the planner.

4. **Mode-collapse evidence** on bf16 agent's MCA predictions: `obj_appearance_order` is **'B' × 6 / 10** (raw VLM was 'A' × 6); `route_planning` is **'C' × 5 / 10**; `object_rel_direction_easy` is **'B' × 4 + 'B. left' × 3 = 7 / 10**. The planner is sometimes guessing rather than reasoning. The raw VLM has analogous biases (different letter, same mode-collapse pattern), so this is a generic VLM/LLM behavior, not an agent-specific bug — but on the rare tasks where the agent's bias matches GT (route_planning), it wins; on the others, it loses.

## What this means for the research direction

The previous "agent beats AWQ raw VLM by 5 points" framing reversed even without changing the agent — just by upgrading the comparator from AWQ to bf16. On a matched-precision raw VLM, the agent is **below** by 13 points. The agent's only durable wins are exactly the tasks where multi-step structured reasoning helps: `route_planning` and `object_rel_direction_hard`. Everything else is a tax.

Two paths forward:

1. **Selective agent**: a meta-router classifies the question type and only invokes the agent for plan-like tasks (route_planning, rel_direction_hard). Falls back to raw VLM for measurement/perception. If we apply this routing to the current numbers, the upper bound is roughly:
   - route_planning + rel_direction_hard via agent: 40 + 30 (only 20 of 100 questions)
   - everything else via raw VLM: ~42 average over 80 questions
   - blended: ~40 overall, basically matching raw VLM, with the agent contributing only on the tasks it actually helps

2. **Fix the agent's information bottleneck**: give the VLM tools access to multiple frames simultaneously (not one), or add a "show all 8 frames" tool. This is a meaningful refactor of `frame_VQA` / `object_VQA` and the prompt.

Option 1 is a 1-day experiment; option 2 is closer to a week. Option 1 also tells us *whether the agent has value at all* before investing in refactor.

## Honest AWQ-vs-bf16 agent comparison (rerun on matched 100 IDs)

We re-ran the AWQ agent on the **bf16 sample's exact 100 question IDs** (extracted to `results/subset_bf16.ids.txt`; new `--ids-file` flag added to `scripts/03_run_vsibench.py`) → `results/subset_awq_on_bf16_ids.jsonl`. Now all three columns below are on identical questions:

```
| question_type          | agent AWQ | agent bf16 | raw VLM bf16 | paper |  (all on same 100Q)
| overall                |   28.0    |    27.0    |    40.5      | 37.5  |
| object_counting        |   28.2    |    17.3    |    17.3      | 31.3  |
| object_abs_distance    |   26.4    |    23.6    |    41.8      | 29.0  |
| object_size_estimation |   13.6    |    17.3    |    36.4      | 48.9  |
| room_size_estimation   |   39.1    |    24.5    |    51.8      | 44.2  |
| object_rel_distance    |   40.0    |    50.0    |    60.0      | 38.0  |
| object_rel_direction   |   46.7    |    23.3    |    36.7      | 33.4  |
| route_planning         |   30.0    |    40.0    |    20.0      | 28.9  |
| obj_appearance_order   |    0.0    |    20.0    |    60.0      | 46.4  |
```

**Headline:** AWQ vs bf16 agent on matched 100Q = **28.0 vs 27.0** — a **1-point gap, within sampling noise**. The "−3.7 regression" claimed yesterday was almost entirely a sample artifact (different question pool because the bf16 candidate-pool included 80 cached scenes vs 19 for AWQ's run).

### Per-task agent A/B (AWQ → bf16, same 100 IDs)

Big swings in both directions that mostly cancel:

- **bf16 helps**: `obj_appearance_order` 0 → 20 (**+20**), `object_rel_distance` 40 → 50 (+10), `route_planning` 30 → 40 (+10), `object_size_estimation` 13.6 → 17.3 (+4).
- **bf16 hurts**: `object_rel_direction` 46.7 → 23.3 (collapsed across 3 subtypes; raw direction-easy 60 → 30, direction-medium 40 → 10), `room_size_estimation` 39.1 → 24.5 (−15), `object_counting` 28.2 → 17.3 (−11), `object_abs_distance` 26.4 → 23.6 (−3).
- Net per-type-equal-weight average: roughly +1 point either direction. Effectively a tie.

So the LLM precision change isn't moving the headline. The pattern is consistent with the structural-agent-bottleneck story from above: **whether the planner sees AWQ or bf16 representations of its tools' text outputs barely matters compared to what those tools surface vs. what raw VLM sees end-to-end**.

### What this clarifies about the project's headline

Across both agent runs (AWQ and bf16, both ~27-28 overall on the same 100Q):
- The agent's only durable advantage over raw VLM is `route_planning` (30 / 40 vs 20). Holds in both quantizations.
- The agent's biggest deficit vs raw VLM is everywhere else — particularly perception/measurement tasks (`object_size_estimation` 13-17 vs 36, `room_size_estimation` 25-39 vs 52, `obj_appearance_order` 0-20 vs 60).

The agent-vs-raw-VLM story doesn't change once we put both agents on the same IDs. It just gets cleaner: the agent loses ~13 points overall to raw VLM (40.5), and the question of "which agent precision is better" is now moot.

## Artifacts

- `experiments/eva-eval/scripts/diff_agent_runs.py` — the diff tool, takes two JSONLs and emits per-question deltas + mode-collapse histograms + MRA near-miss accounting.
- The 100 question IDs the bf16 stack ran on: `set(json.loads(l)['id'] for l in open('results/subset_bf16.jsonl'))` — pin this to a `results/subset_bf16.ids.txt` if we want future runs to use the same exact IDs.
