# Why the SI agent scores 26 when its own VLM scores 60 — deep dive

(2026-05-15, follow-up to
[2026-05-15-si-backbone-swap.md](2026-05-15-si-backbone-swap.md))

## TL;DR

The "agent gives the VLM one frame at a time" thesis from
[2026-05-14-bf16-agent-regression.md](2026-05-14-bf16-agent-regression.md) is
**only a small part of the gap, not its dominant cause**. With trajectory
data captured per row, the failure breakdown across all 100 questions is:

```
MODE_COLLAPSE                35   planner shortcuts to a bare letter / round number with <=1 tool call
CORRECT                      26   (informational)
OTHER                        22   tool calls happen but the planner mis-uses the outputs (wrong object retrieved, can't do math, hallucinated count)
PLANNER_IGNORED_TOOL          6   tool returned a usable answer; planner picked something else
FRAME_SINGLE_INSUFFICIENT     5   single-frame restriction on a multi-frame question
EARLY_STOP_AT_LOCALIZE        4   only called frame_localization; never opened a frame
ITER_LIMIT                    1
COUNT_SQL_EXACT_MATCH         1
```

**~57 of 74 failing questions are agent-orchestration bugs, not VLM
bottleneck.** The 60 → 26 gap is dominated by the planner not using its
tools (mode-collapse + early-stop-at-localize = 39%) plus tool-use that
produces garbage (OTHER + planner-ignored = 28%). Only **5 of 100** are
the canonical "single-frame VLM saturation" failure the prior note
predicted.

Five concrete root causes, in expected-ROI order for a follow-up fix
plan:

1. **Planner skips to a guess on direction / room-size / counting
   questions.** 7/10 `rel_direction_medium`, 6/10 `rel_direction_easy`,
   6/10 `room_size_estimation`, 5/10 `rel_distance` are MODE_COLLAPSE.
   Fix: prompt enforcement + per-task playbooks that *require* opening
   frames before answering.
2. **6/10 counting questions ask about objects YOLO-World cannot
   detect** (door, pillow, bookshelf, whiteboard, trash can / bin).
   When memory has zero entries, the agent fabricates a count (4 of
   those 6 predict `5`). Fix: expand the detection class list, AND
   teach the planner to answer "0" when memory is empty rather than
   guessing.
3. **2/10 counting questions are fragmented in memory.** Keyboard
   GT=2 → memory has **35**; chair GT=6 → memory has **51**. Sasha's
   long-running hypothesis is confirmed for ~20% of counting cases.
   Fix: harden static-object re-ID (DINOv2 / CLIP similarity
   thresholds) and/or count via the `query_db` SQL path with
   deduplication.
4. **Tool outputs are unanchored from the question.** Stool-size
   question retrieves placemats, tables and a bench; the planner
   answers with the bench's dimension (104 cm) instead of saying the
   stool wasn't found. Fix: stricter retrieval (CLIP-similarity gate
   + "no match" sentinel) and a prompt rule that forbids answering
   from an object the retrieval explicitly described as something
   else.
5. **The agent can't do geometry on bbox coords.** A trajectory that
   pulls bbox `[0.42, 0.32, 0.69, 0.66]` for a shoe rack and
   `[0.45, 0.46, 0.53, 0.53]` for a laptop then guesses the distance
   as `0.11` (ground truth: 4.0). Fix: a dedicated tool that takes
   two object IDs and returns the Euclidean distance from their 3D
   AABB centers (we already have the 3D info in memory; the planner
   just isn't asked to use it).

The single-frame structural fix (`frame_VQA_multi`) is still useful
for `obj_appearance_order` (5/10 of that task type), but it's a
narrow fix, not the dominant one we thought.

## Observability — what was missing

Until this dive, every JSONL row stored only the planner's final
prediction. The agent's ReAct steps (its thoughts, the tools it
called, what each tool returned) were discarded by LangChain's
`AgentExecutor`. We couldn't see why a specific question failed.

Fixed in commit `aca9eab`:

- `agent.py:84-90` — pass `return_intermediate_steps=True` to
  `AgentExecutor`.
- `vsibench.py` — new `serialize_trajectory()` flattens each step to
  `{thought, tool, tool_input, observation}` and attaches it as a
  `trajectory` list on every row. Observations truncated to 1000
  chars so VLM caption paragraphs don't bloat the JSONL.

Re-running the 100Q on the SI stack with traces enabled gave
identical scores (26.4 overall, ±0 vs the prior `subset_si.jsonl`),
confirming the change is pure observability with no behavior
perturbation.

Result file: `results/subset_si_with_traces.jsonl`. Two diagnostic
scripts consume it:
[`scripts/audit_memory_objects.py`](../experiments/eva-eval/scripts/audit_memory_objects.py)
(Phase C) and
[`scripts/classify_agent_failures.py`](../experiments/eva-eval/scripts/classify_agent_failures.py)
(Phase D).

## Per-task failure histograms

Read this as: "for each question type, where do the 7-of-10
failures cluster?"

```
obj_appearance_order (n=10)
    FRAME_SINGLE_INSUFFICIENT  5
    MODE_COLLAPSE              3
    ITER_LIMIT                 1
    CORRECT                    1

object_abs_distance (n=10)
    EARLY_STOP_AT_LOCALIZE     3
    CORRECT                    3
    PLANNER_IGNORED_TOOL       2
    OTHER                      2

object_counting (n=10)
    OTHER                      7    <- all but one are YOLO_MISS in memory.pkl
    CORRECT                    1
    COUNT_SQL_EXACT_MATCH      1
    MODE_COLLAPSE              1

object_rel_direction_easy (n=10)
    MODE_COLLAPSE              6
    OTHER                      2
    CORRECT                    2

object_rel_direction_hard (n=10)
    CORRECT                    3
    MODE_COLLAPSE              3
    PLANNER_IGNORED_TOOL       3
    OTHER                      1

object_rel_direction_medium (n=10)
    MODE_COLLAPSE              7    <- worst single bucket
    CORRECT                    2
    OTHER                      1

object_rel_distance (n=10)
    MODE_COLLAPSE              5
    CORRECT                    5

object_size_estimation (n=10)
    OTHER                      5    <- e.g. "stool" retrieval returned a bench
    CORRECT                    4
    PLANNER_IGNORED_TOOL       1

room_size_estimation (n=10)
    MODE_COLLAPSE              6    <- planner doesn't look at frames before answering
    CORRECT                    3
    EARLY_STOP_AT_LOCALIZE     1

route_planning (n=10)
    OTHER                      4
    MODE_COLLAPSE              4
    CORRECT                    2
```

Two patterns dominate:

- **MODE_COLLAPSE is the dominant failure for every relational and
  size task** (rel_direction_easy/medium/hard, rel_distance,
  room_size_estimation, route_planning).
- **OTHER is the dominant failure for tasks where tool outputs are
  semantically structured but easy to misread**: counting (all but
  one OTHER row is YOLO_MISS — see §below), size_estimation (the
  planner anchors its answer to the wrong retrieved object).

`obj_appearance_order` is the only task where the
single-frame-VLM thesis (`FRAME_SINGLE_INSUFFICIENT`) dominates,
matching intuition: ordering questions need cross-frame integration
that single-frame tools can't provide.

## Memory.pkl audit: counting failures aren't agent failures (mostly)

`scripts/audit_memory_objects.py` ran on the 10 counting scenes,
loaded each `memory.pkl`, and counted `Object3D` entries per YOLO
category. Status histogram: `{YOLO_MISS: 6, FRAGMENT: 2, UNDERCOUNT: 1, CORRECT: 1}`.

```
    id  noun       in_yolo  gt  memory  agent  raw_vlm  status      memory top-3
  4455  keyboard       ✓     2    35     0      2       FRAGMENT    book=46, cup=37, keyboard=35
  4601  chair          ✓     6    51     3      4       FRAGMENT    book=65, cabinet=60, chair=51
  4456  backpack       ✓     3     3     4      2       CORRECT     book=46, cup=37, keyboard=35
  2642  table          ✓    11     1     6      4       UNDERCOUNT  chair=20, cabinet=14, box=10
  4393  door           ✗     5     0     4      2       YOLO_MISS   towel=10, bottle=10, box=9
  4397  pillow         ✗     4    14*    2      2       YOLO_MISS   bottle=31, cup=30, book=27
  2540  bookshelf      ✗     2     0     5      2       YOLO_MISS   book=24, box=19, keyboard=11
  2533  whiteboard     ✗     2     5*    5      2       YOLO_MISS   telephone=27, tv=25, box=21
  2618  trash can      ✗     2     0     5      2       YOLO_MISS   box=21, mouse=11, chair=10
  4493  trash bin      ✗     2     0     5      2       YOLO_MISS   bucket=11, box=11, bottle=10
```
*`memory.pkl` carries a category not declared in the canonical YOLO class
list — the actual detector vocab on morgen has drifted from `detection_classes.py`'s
`customized_classes`; the audit reports the literal category string as-is.*

Three things jump out:

1. **6/10 questions ask about object categories YOLO can't detect at
   all** (door, pillow, bookshelf, whiteboard, trash can, trash bin).
   In four of those six, the agent predicts `5` — a fabricated
   number. The agent doesn't know its memory is empty for that
   category; the prompt doesn't tell it how to answer "I don't have
   data on this".
2. **FRAGMENT (Sasha's hypothesis) is real but limited.** Keyboard
   becomes 35 entries; chair becomes 51 entries. Both have
   ground-truth in single digits. Memory inflates by ~6–17× on these
   two. But fragmentation is *only* 2/10 cases — not the dominant
   counting failure mode.
3. **The CORRECT row (backpack)** is the proof-of-concept that the
   pipeline *can* work end-to-end: memory has the right count,
   agent gets it within 1 of the answer. This is the bar to raise.

## Three illustrative trajectories

These came out of the trace data and are quoted verbatim so future
readers can see exactly what the agent did. Observations are
truncated to ~200 chars.

### MODE_COLLAPSE — room_size_estimation, id=532, gt=7.1, pred=15

```
Thought: To estimate the size of the room, I need to identify the frames
         that show the room and then determine its dimensions...
Action:  frame_localization
Input:   "room"
Obs:     The most relevant frame indices are [1, 25, 18, 32, 11]

Final Answer: 15
```

The planner correctly identified that it needed to look at frames.
It correctly used `frame_localization` to find them. It then *did
not open a single frame*. The bare number `15` is a guess. Same
pattern on id=625 (gt=35.1, pred=10.0) and id=1240
(rel_direction_easy, gt=A, pred=B) — both stop after one
`frame_localization` call.

### PLANNER_IGNORED_TOOL — object_size_estimation, id=467, gt=76, pred=104

```
Thought: To find the length of the longest dimension of the stool, I need
         to first identify the stool in the video frames...
Action:  retrieve_objects_by_environment
Obs:     The objects that satisfy 'stool' are {95: 'cabinet', 57: 'placemat',
         60: 'table', 59: 'placemat', 58: 'placemat'...}

Thought: The observation indicates that there are no objects matching the
         description "stool" in the environment...
Action:  retrieve_objects_by_appearance
Obs:     {63: 'chair', 61: 'chair', 23: 'stove', 67: 'chair', 60: 'table',
         95: 'cabinet'...}

Thought: ...there are several objects that visually resemble a stool...
Action:  frame_localization
Obs:     [92, 49, 50, 51, 20]

Action:  frame_VQA
Obs:     This is a picture of a bench. Answer to the question: The longest
         dimension of the bench is 1.04 meters.

Final Answer: 104
```

The agent did everything except notice that no tool ever returned a
*stool*. It walked through three retrievals that produced cabinets,
placemats, chairs, a bench — and answered with the bench's length.

### OTHER (math) — object_abs_distance, id=1731, gt=4.0, pred=0.11

```
(8 tool calls; final two:)

Action:  object_VQA  [first time]
Obs:     There is a shoe rack and a laptop. The shoe rack is at
         [0, 424, 544, 848] and the laptop is at [434, 444, 544, 546].

Action:  object_VQA  [second time, same frame]
Obs:     There is a shoe rack and a laptop. The shoe rack is at
         [0.42, 0.32, 0.69, 0.66] and the laptop is at [0.45, 0.46,
         0.53, 0.53] in the image.

Final Answer: 0.11
```

The agent extracted normalized bbox centers, presumably did
`sqrt((0.42-0.45)^2 + (0.32-0.46)^2) ≈ 0.14`, and emitted `0.11`.
The ground truth is `4.0` *meters*. The agent treated image-space
normalized coordinates as if they were a metric distance. Same
core failure as the size_estimation trajectory above — the planner
has no idea what units it's working in, and the prompt never
introduces scale information.

## Resize hypothesis — skipped, with rationale

The original plan included a 100Q re-run with `tools.py` resizing
frames to 448×448 before sending to the VLM (mirroring the
baseline path). After Phase D landed and showed mode-collapse is
the dominant failure (35%) and only 5/100 are single-frame-VLM
issues, the expected ROI of the resize test dropped sharply:
resize can only help VLM-saturated tool calls, but most failing
questions don't make tool calls at all, or make tool calls whose
outputs the planner then misuses.

Decision: skip the 2-hour re-run; revisit if a follow-up plan
implements the structural multi-frame tool and we want a clean A/B
on visual-token budget. The patch I drafted (a `_resize_for_vlm`
helper + three call-site wraps in `tools.py`) is documented here
so the next session can apply it directly:

```python
# in eva_eval/agent/tools.py near the top
_VLM_INPUT_SIZE = (448, 448)

def _resize_for_vlm(image):
    if image is None:
        return image
    if getattr(image, "size", None) == _VLM_INPUT_SIZE:
        return image
    return image.resize(_VLM_INPUT_SIZE)

# then wrap each of the three ctx.vlm.vqa(image, ...) call sites:
#   do_retrieve_objects_*:  ctx.vlm.vqa(_resize_for_vlm(image), DESCRIBE_OBJECT_PROMPT)
#   do_frame_vqa:           ctx.vlm.vqa(_resize_for_vlm(image), DESCRIBE_THEN_ANSWER.format(...))
#   do_object_vqa:          ctx.vlm.vqa(_resize_for_vlm(image), DESCRIBE_THEN_ANSWER.format(...))
```

## Ranked fixes (revised)

Promoted from the original plan based on Phase D's data:

| # | Fix | Targets | Expected ROI | Effort |
|---|---|---|---|---|
| 1 | **Stop the planner from short-circuiting.** Add a system rule: "you MUST call at least one of [frame_VQA, object_VQA, retrieve_*] before emitting Final Answer for any question type in {rel_direction_*, room_size, rel_distance, abs_distance, route_planning}." | the 35 MODE_COLLAPSE rows + 4 EARLY_STOP_AT_LOCALIZE | **+10 to +20 pts overall** (very rough upper bound: 39 rows × ~40% recoverable from properly-tool-using runs) | 1 hour prompt rewrite |
| 2 | **Add an `aabb_distance(object_id_a, object_id_b)` tool** that returns the 3D Euclidean distance between two object AABB centers from memory. Geometry is in `Object3D` already; planner just needs the verb. | object_abs_distance (3 EARLY_STOP rows + 2 PLANNER_IGNORED + 2 OTHER = 7/10) | **+5 to +10 pts** on its task type | 0.5 day (tool + prompt + 5Q test) |
| 3 | **"No-data sentinel" in counting.** If `retrieve_objects_by_appearance` returns no candidate whose VLM caption substring-matches the query noun, the planner must answer `0`. | 4 of 6 YOLO_MISS counting rows currently predict `5`; switching those to `0` is +0 score (gt is non-zero) but stops fabrication. Pair with #4. | +0 to +2 (depends on whether benchmark even has true-zero counts) | 1 hour |
| 4 | **Expand YOLO-World class list** to include door, pillow, bookshelf, whiteboard, trash can, trash bin, mirror, curtain, lamp (common VSI-Bench targets). Requires rebuilding memory.pkl for affected scenes. | 6 YOLO_MISS counting rows + an unknown number of failures in other task types where the planner couldn't retrieve the target object (e.g. the "stool" question above) | **+5 to +10 pts** if rebuild is feasible | ~3-5 hours (rebuild memory.pkl for 80+ scenes, ~5 min each = 6–8 hours wall-clock) |
| 5 | **Multi-frame `frame_VQA_multi(question, frame_ids)` tool** that sends N frames (default 8) in one VLM message, mirroring the raw-VLM baseline. | obj_appearance_order (5/10 FRAME_SINGLE_INSUFFICIENT), possibly room_size_estimation | **+3 to +5 pts** | 0.5 day |
| 6 | **Fix object re-ID fragmentation** (keyboard 35, chair 51 in single scenes). Tighten DINOv2 / CLIP similarity thresholds in `static_object_reid`. | 2 FRAGMENT counting rows + likely additional perceived noise on retrieve_* tools across scenes | +1 to +3 pts | 1 day (paper code change, no API stability guarantees) |
| 7 | Image-resize patch (skipped above). Likely +0 to +1 in isolation; bundle with the multi-frame tool fix if/when that lands. | All tool VLM calls | +0 to +1 | 30 min |

The first three are highest-ROI: a prompt rule, a 50-line tool, and
a sentinel string. Combined, they target ~50 of the 74 failing
questions, with no architecture changes.

## What didn't pan out

For the record — hypotheses ruled out by the trace data:

- **"Iteration limit exhaustion"** — only 1/100 questions hit the
  30-iteration cap (Q2855 `obj_appearance_order`). Compute budget
  is not the bottleneck. (Already noted in
  [2026-05-15-si-backbone-swap.md](2026-05-15-si-backbone-swap.md).)
- **"The agent is failing because the SI backbone makes the planner
  worse"** — the bf16 agent and SI agent agree on 50% of questions
  and disagree on a different 50%, with no systematic direction.
  Backbone swap is not the cause of agent failures.
- **"Single-frame VLM saturation dominates"** — only 5/100 rows
  classify as FRAME_SINGLE_INSUFFICIENT. The structural-bottleneck
  framing from 2026-05-14 was right about the *mechanism* (the
  agent can't see what the raw VLM sees) but wrong about the
  *dominant cause* — the planner mostly isn't even calling the
  tools that would let it see anything.

## Artifacts

- `experiments/eva-eval/results/subset_si_with_traces.jsonl` —
  100 rows with per-question `trajectory` field
- `experiments/eva-eval/results/memory_audit_counting.json` —
  Phase C output
- `experiments/eva-eval/results/agent_failures_si.json` —
  Phase D output (per-row classification + histograms)
- `experiments/eva-eval/scripts/audit_memory_objects.py`,
  `scripts/classify_agent_failures.py` — diagnostic scripts; both
  re-runnable on future agent JSONLs.
