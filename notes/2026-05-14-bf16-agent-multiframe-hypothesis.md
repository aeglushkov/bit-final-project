# Why bf16 lifts raw VLM by +15 but not the agent (2026-05-14)

## TL;DR

The bf16 advantage we saw over AWQ on the raw VLM (+15 points on the 100Q subset) is **entirely a multi-frame phenomenon**. At single-frame VQA the bf16 model is actually 3 points *worse* than AWQ. The agent — which only ever calls the VLM with one image at a time (`eva_eval/agent/tools.py:92, 115, 126`) — scores at the 1-frame raw-VLM ceiling regardless of quantization, because the multi-frame strength bf16 unlocks lives in a code path the agent doesn't have a tool for.

## The 4-cell experiment

Same 100 question IDs (`results/subset_bf16.ids.txt`), same eval driver, same prompt, same lmdeploy serving stack — only `--n-frames` and the quantization toggled:

```
| raw VLM            |  n_frames=1  |  n_frames=8  |  delta 1→8 |
| AWQ                |    28.14     |    36.33     |   +8.19    |
| bf16               |    25.04     |    40.49     |  +15.45    |
| delta AWQ→bf16     |    −3.10     |    +4.16     |            |
```

Reproduce: see `results/baseline_internvl2-8b{,-bf16}{_1f,_8f}.jsonl.summary.json`.

Read across:
- **Multi-frame helps both quantizations**, but bf16 benefits twice as much (+15 vs +8). The bf16 model's spatial-integration capacity is what got cut by INT4 quantization, not its single-image perception.
- **At 1 frame, AWQ is slightly better than bf16.** The headline "bf16 raw VLM beats AWQ raw VLM by 15 points" turns into "...as long as you give it 8 frames." Take frames away and the ordering reverses.

## Where the agent sits in this picture

```
| | overall on the same 100 IDs |
|---|---:|
| raw VLM AWQ  @ 1 frame  | 28.14 |
| agent AWQ              | 28.00 |  <-- ≈ 1f raw-VLM AWQ
| raw VLM bf16 @ 1 frame  | 25.04 |
| agent bf16             | 27.00 |  <-- close to 1f raw-VLM bf16
| raw VLM AWQ  @ 8 frames | 36.33 |
| raw VLM bf16 @ 8 frames | 40.49 |
```

The agent's score under each quantization is **almost exactly the 1-frame raw-VLM score under the same quantization**:
- AWQ: 28.0 (agent) vs 28.1 (1f raw VLM) → 0.1-point gap.
- bf16: 27.0 (agent) vs 25.0 (1f raw VLM) → +2 points (probably from `route_planning`, the agent's only durable advantage — see prior notes).

The agent's whole tool stack — object retrieval, SQL queries, ReAct loop, multi-turn planning — is contributing **essentially nothing on average** over what you'd get from asking the VLM a single frame. The remaining 13 points to the 8-frame ceiling are unreachable because the agent's tool surface doesn't include a multi-frame VLM call.

## Per-task pattern that supports the structural story

Multi-frame's value isn't uniform across tasks. Compare 1f → 8f deltas:

```
| task                       | AWQ Δ(1→8) | bf16 Δ(1→8) | needs multi-frame? |
| object_counting            |   +23.6    |   +14.5     |  yes (see whole scene) |
| object_rel_distance        |   +30.0    |   +40.0     |  yes |
| object_abs_distance        |   +14.5    |   +10.9     |  yes-ish |
| obj_appearance_order       |   +10.0    |   +30.0     |  YES (cross-frame temporal) |
| room_size_estimation       |    +2.7    |   +29.1     |  yes (whole-scene integration) |
| object_size_estimation     |    −5.5    |    −0.9     |  no (per-frame visual detail) |
| object_rel_direction       |    +0.0    |   +10.0     |  borderline |
| route_planning             |   −10.0    |   −10.0     |  no (instructions in question) |
```

Tasks needing cross-frame integration (`object_counting`, `object_rel_distance`, `obj_appearance_order`, `room_size_estimation`) get big multi-frame boosts. Single-image tasks (`object_size_estimation`, `route_planning`) don't, sometimes worsen. This is the expected shape.

The agent loses every multi-frame-favoring task to the raw 8-frame VLM. The only task where the agent *outperforms* the 8-frame raw VLM is `route_planning` (40/30 vs 20) — which is exactly the task that doesn't benefit from multi-frame (the question gives explicit step-by-step instructions to follow, so the agent's structured plan-then-execute loop is genuinely useful). Coherent story.

## Implication for the project

The +13-point gap between the agent and the 8-frame raw VLM is **entirely the cost of single-frame VLM tools**. Concretely:

1. **Adding a multi-frame VLM tool** (`frames_VQA(question, frame_ids)` calling a new `ChatModel.vqa_multi(images, question)`) is the highest-leverage agent change available. Upper bound on agent score from this single change is roughly the 8-frame raw-VLM number (40+) on tasks where multi-frame helps, kept at agent-style numbers on tasks where it doesn't. The path to 35-40+ overall is plausible without other changes.

2. **Further model upgrades won't help until that change lands.** A bigger / smarter / less-quantized model only matters at the call sites that already see enough context; the agent doesn't have one. Switching to InternVL2.5 or a different VLM would be wasted effort on the current tool surface.

3. **The route-planning advantage is real and orthogonal.** Even after the multi-frame tool fix, route_planning will still favor the agent. The right framing of the project's contribution is: "agent layer adds value on multi-step reasoning tasks; baseline raw VLM is better on direct perception" — *not* "agent beats raw VLM overall."

## Concrete next experiment (out of scope for this note)

Prototype `frames_VQA` and `ChatModel.vqa_multi` (multi-image OpenAI content array, mirroring `baseline.py:44-51`). Update `eva_eval/prompts/react_vqa.txt` to prefer it. Re-run the 100Q subset on the bf16 split stack with `EVA_PLANNER=qwen2.5-7b-text-bf16 EVA_VLM=internvl2-8b-bf16`. Headline target: > 36 overall (i.e. clearly above raw 8f AWQ, plausibly competitive with or above raw 8f bf16 once the route_planning win is included).

If that hits, the agent finally adds value over raw VLM in matched precision. If it doesn't, the structural problem is deeper than the tool surface and we'd need to look at the planner's information aggregation across multiple tool returns.

## Artifacts

- `experiments/eva-eval/results/baseline_internvl2-8b{,-bf16}{_1f,_8f}.jsonl(.summary.json)` — the four runs.
- `experiments/eva-eval/results/subset_bf16.ids.txt` — the pinned 100 IDs all four runs (and the two agent runs) used.
- `experiments/eva-eval/scripts/compare_baselines.py` — for rendering tables.
- This note's predecessor: `notes/2026-05-14-bf16-agent-regression.md` (which resolved the sample-mismatch puzzle and set up this question).
