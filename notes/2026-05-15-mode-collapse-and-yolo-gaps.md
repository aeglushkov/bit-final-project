# Mode-collapse and YOLO target-missing — drill-down + cheap experiments

(2026-05-15, follow-up to
[2026-05-15-si-agent-deep-dive.md](2026-05-15-si-agent-deep-dive.md))

## TL;DR

The deep-dive named two dominant failure modes for the SI agent
without explaining *why* they fire. This note grounds both with
trace-level evidence and runs two cheap prompt-only experiments to
test the smallest possible interventions.

**Mode collapse — drill-down findings:**

- 34/39 mode-collapse rows are **PLAN_BUT_QUIT**: the planner
  *writes out a multi-step plan* in its Thought (e.g., "first I'll
  use `frame_localization` to find relevant frames, then `frame_VQA`
  to ask about them") and then quits after step 1, or with zero tool
  calls.
- The same templated Thought appears across multiple rows verbatim
  for `room_size_estimation` — InternVL3 has a learned template that
  doesn't translate into execution.
- **Cross-stack diff is the load-bearing finding:** on the 39 SI
  mode-collapse questions, **bf16 (Qwen-7B planner) scored 8/39
  correctly** (~21%); SI (InternVL3 planner) scored 0/39. With
  identical tools, identical prompt, only the planner LLM differs.
  Qwen is the better ReAct planner.

**M4 prompt-only intervention (force tool use):**

- Rule #6 added to react_vqa.txt: "you MUST call at least one of
  [frame_VQA, object_VQA, retrieve_*, query_db] before Final Answer;
  frame_localization alone is not sufficient."
- Re-ran the 39 mode-collapse IDs on the SI stack.
- **Overall +9.8 pts** on those 39 Qs (1.6% → 11.4%). Avg trajectory
  length **doubled** (0.9 → 2.05 steps). 4/39 rows newly correct.
- But on the hardest tasks (rel_direction_medium/hard, rel_distance,
  route_planning) the score stays 0% — the planner now uses tools,
  it just can't reason from them. Confirms M2: prompt helps,
  planner is the deeper limit.

**YOLO target-missing — drill-down findings:**

- The 162-class runtime detection list (`config/detection_classes.txt`)
  covers **51–90% of all VSI-Bench questions per task type**, not
  0.1–49% as the deep-dive note reported. The deep-dive read the
  wrong file (the paper's 60-class `customized_classes` from
  `literature/EmbodiedVideoAgent/code/detection_classes.py`, which
  is **not** what the agent actually runs against). The actual
  vocab gap is smaller than originally claimed.
- Of the 6 counting questions originally tagged YOLO_MISS, only
  **2 (trash can, trash bin) are true vocab-misses**. The other 4
  (door, pillow, bookshelf, whiteboard) ARE in the 162-class vocab
  — they're **DETECTOR misses** (YOLO-World failed to detect a
  present-and-in-vocab object).
- Real top-missing nouns across all 5130 VSI-Bench questions:
  `trash bin` (381), `ceiling light` (217), `computer mouse` (158),
  `trash can` (118), `power strip` (72), `computer tower` (51),
  `cutting board` (46), `exhaust fan` (46), `shoe rack` (17),
  `washing machine` (6). Several have head-noun matches in YOLO
  (`trash`, `ceiling`, `computer`, `cutting`, `rack`), which retrieves
  partially via CLIP similarity.
- **Retrieve_* call quality (Y3):** **53% of retrieve_* calls return
  WRONG** captions (queried "stool", got cabinet / placemat / bench).
  Zero EMPTY. The top-10 CLIP-similarity gate returns *something*
  every time; that "something" matches the query only half the time.

**Y5 prompt-only intervention (no-data sentinel for counting):**

- Rule #6 (in a separate branch): "For counting questions, if no
  retrieved caption matches the target noun, your Final Answer MUST
  be '0'. Captions describing something different do not count."
- Re-ran the 6 originally-YOLO_MISS counting IDs.
- **Overall +6 pts** (12% → 18%) — **whiteboard hit 100%** (5 → 2,
  correct), and the four "5"s in original predictions collapsed to
  varied numbers (1, 14, 4, 2, 0, 1). The fabrication-as-`5` pattern
  is broken.
- But mixed: door regressed (4, 64% partial credit → 1, 0%) because
  the sentinel pushed the planner off a near-correct guess.
- Avg trajectory steps on these 6 jumped from ~1 to ~5 — like M4,
  the rule indirectly forced more tool engagement.

## Ranked recommendations for the next implementation plan

Promoted from the deep-dive's guesses, validated by M2/M4/Y3/Y5:

1. **Swap the planner back to Qwen2.5-7B-Instruct while keeping the
   SenseNova-SI VLM.** Strongest single lever. M2: Qwen got 8/39
   correct on questions where InternVL3 got 0/39. Effort: spin up a
   "hybrid" stack (Qwen text-only on morgen + SI VLM on neo) — both
   conda envs already exist, just a tweak to
   `scripts-remote/03_start_servers_*` and `config/models.yaml`.
   Expected upper bound on full 100Q: **+8 to +12 points overall**
   (extrapolating the 8/39 recovery to the full set).

2. **Adopt M4's force-tool-use prompt rule.** Validated: +9.8 pts on
   the 35-39 mode-collapse subset. Free; commit the patch.
   Expected on full 100Q: **+3 to +5 points overall**.

3. **Stricter CLIP-similarity gating in `_do_retrieve`** to suppress
   WRONG retrievals. Currently the top-10 is unconditional; a
   similarity floor would drop spurious matches. Combine with
   Y5's no-data sentinel so the planner trusts the empty result.
   Effort: ~20 lines in `eva_eval/agent/tools.py:73-96` + similarity
   threshold tuning on traces. Expected: +2 to +4 points (across
   all 100Q, not just counting — `retrieve_*` is used in
   abs_distance + size_estimation too).

4. **Add ~10 missing nouns to `config/detection_classes.txt`** —
   trash bin, ceiling light, computer mouse, trash can, power
   strip, computer tower, cutting board, exhaust fan, shoe rack,
   washing machine. Then rebuild memory.pkl for the 512 cached
   scenes (~9 hours wall-clock, Phase 2 / MASt3R reusable).
   Expected on full 100Q: **+1 to +3 points** (smaller than I
   initially estimated because detector accuracy is the bigger
   constraint than vocab).

5. **Investigate YOLO-World detection failure modes** for in-vocab
   objects that go undetected (door, bookshelf — both in the
   162-class list but missing from memory). Plausibly: confidence
   threshold too tight, frame sampling missed the object, or
   YOLO-World needs prompt-tuning per scene. Effort: read 10
   memory builds, look for the per-frame YOLO outputs. Deferred —
   separate diagnostic plan.

The top-3 fixes are all <1-day efforts. Combined upper-bound
expected lift on the full 100Q: **+13 to +21 points** (from 26 to
39–47). That would close roughly half the gap to the raw VLM at 60.

## Method log

### Mode-collapse trajectories (M1 — pattern-mining)

Across the 35 MODE_COLLAPSE + 4 EARLY_STOP_AT_LOCALIZE rows:

```
PLAN_BUT_QUIT       34   the planner writes a Thought outlining a
                          multi-step plan, then emits Final Answer
                          before executing it
NO_THOUGHT           4   route_planning rows whose model output didn't
                          fit the ReAct format; LangChain's
                          handle_parsing_errors=True ate the failure
OTHER                1
```

Three representative `room_size_estimation` Thoughts are **literally
identical**:

> "To estimate the size of the room, I need to identify the frames
> that show the room and then determine its dimensions. Since the
> question asks for the size in square meters, I should first find
> the relevant frames that include the room."

After this Thought the planner calls `frame_localization("room")`,
gets back `[1, 25, 18, 32, 11]`, and then writes Final Answer (a
guess: `15`, `10.5`, `10.0` against gt `7.1`, `26.9`, `35.1`). The
plan is correct; execution stops before step 2.

### Cross-stack diff (M2 — load-bearing experiment)

On the 39 SI mode-collapse IDs:

```
                        SI agent    bf16 agent
correct (score>=0.5)      0           8        of 39
bf16 strictly better    11
SI strictly better       4
both score 0            24
                       per-task averages
obj_appearance_order      0.0%        0.0%
object_abs_distance      12.1%       12.1%
object_counting           0.0%        9.1%
object_rel_direction_easy 0.0%       33.3%   <- big delta
object_rel_direction_hard 0.0%       66.7%   <- biggest delta
object_rel_direction_medium 0.0%      14.3%
object_rel_distance       0.0%       20.0%
room_size_estimation      3.9%       14.3%
route_planning            0.0%       25.0%
```

**Qwen-7B is a strictly stronger ReAct planner than InternVL3-8B on
these tasks.** The SI backbone gain (+20 raw-VLM points) was real;
the planner-side regression (-8 to -12 inferred points) was hidden
inside the agent layer.

### Prompt audit (M3)

Five clauses in the agent's prompt stack license early-exit:

1. **react_vqa.txt ATTENTION #5** (verbatim): *"If the information
   is insufficient for a precise response, generate a response based
   on the available data."* — explicit permission to guess.
2. **ReAct format definition:** the prompt allows `Thought: I now
   know the final answer\nFinal Answer: ...` to follow any prior
   thought, with **no rule** that an Action must precede Final
   Answer.
3. **`MCA_POST_PROMPT`** appended to every MCA question:
   *"Answer with the option's letter from the given choices
   directly."* The word "directly" biases the model toward a bare
   letter.
4. **`NA_POST_PROMPT`**: *"Do not response anything other than a
   single number!"* — same bias for numeric tasks.
5. **agent_scratchpad starts empty:** on first iteration, the
   model can write a Thought and immediately emit Final Answer
   without ever populating the scratchpad with an Action.

M4 patches #2 with explicit rule #6 (force tool use); leaves
#1/#3/#4/#5 untouched.

### YOLO coverage (Y1/Y2, corrected with the runtime 162-class file)

```
qtype                       parsed   all_in_yolo   any_missing   coverage
object_size_estimation         953        862            91        90.5%
object_counting                565        498            67        88.1%
object_abs_distance            834        672           162        80.6%
object_rel_direction_easy      217        175            42        80.6%
object_rel_direction_hard      373        279            94        74.8%
object_rel_direction_medium    378        271           107        71.7%
obj_appearance_order           618        384           234        62.1%
object_rel_distance            710        367           343        51.7%
```

`room_size_estimation` and `route_planning` don't reference a
specific noun, so coverage is N/A.

Top truly-missing nouns (sum across all task types):

```
trash bin           381   ceiling light       217   computer mouse  158
trash can           118   power strip          72   computer tower   51
cutting board        46   exhaust fan          46   shoe rack        17
washing machine       6
```

Several of these (`trash X`, `ceiling X`, `computer X`, `cutting X`,
`X rack`) have head-noun matches in YOLO, so CLIP-similarity can
recover them partially.

### Retrieve_* quality (Y3)

Across 32 retrieve_* calls in the 100Q traces:

```
WRONG    17    53.1%   no caption matches the queried noun
USEFUL   15    46.9%   at least one caption substring-matches
EMPTY     0     0.0%   the gate returns 10 candidates unconditionally
```

Most-WRONG query nouns: `stool` (4 wrong), `table` (2), and various
ill-formatted queries from the planner concatenating multiple nouns
or forgetting to strip tuple syntax (`("shoe rack", "laptop")` sent
to a 1-arg tool).

### Memory rebuild cost (Y4)

- **512 scenes** currently cached on morgen
- **Phase 3** (`07_full_phase3.log` on 2026-05-07): start 06:27,
  end 15:14 = ~9 hours total → **~1 min/scene** for memory build
- **Phase 2** (MASt3R poses) is detector-agnostic. No re-run needed
  when adding YOLO classes.
- Expanding from 162 → ~175 classes: ~9-hour rebuild, single
  command (`launch.sh 07_full_phase3`).

### Experiments (M4 + Y5)

```
M4: force-tool-use prompt rule, 39 mode-collapse IDs
  overall                1.6% → 11.4%  Δ +9.8
  avg trajectory steps   0.9  → 2.05
  rows score>=0.5        0    → 4
  rows still <=1 step    39   → 32     <- prompt rule respected on 7 rows

Y5: no-data sentinel for counting, 6 YOLO_MISS counting IDs
  overall                12.1% → 18.2%   Δ +6.1
  fabricated "5" preds   4/6   → 0/6     <- pattern broken
  rows that hit gt=0/100 0     → 1
  avg trajectory steps   1.1   → 4.8     <- much more tool use
```

## Artifacts

- `experiments/eva-eval/scripts/analyze_mode_collapse.py`
- `experiments/eva-eval/scripts/extract_vsibench_target_nouns.py`
- `experiments/eva-eval/scripts/audit_retrieve_outputs.py`
- `experiments/eva-eval/results/mode_collapse_thoughts.json`
- `experiments/eva-eval/results/mode_collapse_ids.txt` (39 rows)
- `experiments/eva-eval/results/y5_counting_ids.txt` (6 rows)
- `experiments/eva-eval/results/vsibench_target_nouns.csv`
- `experiments/eva-eval/results/vsibench_target_nouns_summary.json`
- `experiments/eva-eval/results/retrieve_audit_si.json`
- `experiments/eva-eval/results/subset_si_mode_collapse_force_tool.jsonl`
- `experiments/eva-eval/results/subset_si_counting_zero_sentinel.jsonl`
- branch `exp/m4-force-tool` — the M4 prompt patch
- branch `exp/y5-no-data-sentinel` — the Y5 prompt patch

## What this changes vs the deep-dive note

- "Single-frame VLM saturation" was only 5/100; **mode-collapse is
  35/100** — confirmed. Pattern is now specifically PLAN_BUT_QUIT.
- "Backbone swap is not the cause of agent failures" — partially
  reversed: the *VLM* swap helped, but the *planner* swap (Qwen →
  InternVL3) **hurt by 8+ points** on the mode-collapse subset.
  The two backbone choices in the SI stack pull in opposite
  directions. We should reconsider whether SI's gains are worth
  the planner regression.
- "6/10 counting questions ask about objects YOLO can't detect" was
  wrong on attribution — only 2 are true vocab-misses; 4 are
  detector-accuracy failures.
- The "expand YOLO classes" fix is smaller than originally claimed
  (+1 to +3 vs the +5 to +10 estimate).

## Out of scope (becomes follow-up plans)

- **Build the hybrid stack** (Qwen planner + SI VLM) and re-run
  the full 100Q. The biggest expected lever from this drill-down.
- **Commit M4's force-tool rule to the harness branch** and re-run
  the full 100Q to confirm the +3 to +5 extrapolation.
- **Tune `_do_retrieve` similarity threshold** + adopt the Y5
  no-data sentinel as a default.
- **Diagnose why YOLO-World fails to detect in-vocab objects** like
  door / bookshelf / pillow when they're clearly present (this
  needs a frame-level audit, separate from the agent layer).
