# InternVL2-8B raw-VLM baseline: 12-point gap vs paper Table 1 (2026-05-13)

Investigation of why our raw-VLM run (no agent) on VSI-Bench produces **25.27 overall** while the paper reports **37.5** for the same model — and what that gap means for the agent-vs-baseline comparison.

## Per-task scores side-by-side

Paper numbers from Table 1 (`literature/thinking-in-space/Thinking in Space.pdf`, page 5, row `InternVL2-8B`). Ours from `experiments/eva-eval/results/full_qwen_internvl2.jsonl.summary.json` (full 5130-question test split, 8 frames, lmdeploy serve).

| Task | Type | Paper | Ours (raw VLM) | Gap |
|---|---|---:|---:|---:|
| Object Count | NA | 31.3 | 21.2 | **−10.1** |
| Abs Distance | NA | 29.0 | 16.8 | **−12.2** |
| Object Size | NA | 48.9 | 27.5 | **−21.4** |
| Room Size | NA | 44.2 | 15.1 | **−29.1** |
| Rel Distance | MCA | 38.0 | 29.2 | −8.8 |
| Rel Direction | MCA | 33.4 | 39.4 | **+6.0** |
| Route Plan | MCA | 28.9 | 28.4 | −0.5 |
| Appearance Order | MCA | 46.4 | 24.6 | **−21.8** |
| **Overall** | | **37.5** | **25.3** | **−12.2** |

Note: `analysis.md` had transcribed the InternVL2-8B overall as 37.1; the paper PDF says 37.5.

## Pattern

The gap is **not uniform**:

- **NA (numeric/measurement) tasks collapse hard** — room_size −29, obj_size −21, abs_distance −12, object_count −10. These need rich per-frame visual detail.
- **Appearance Order collapses (−22)** — explicitly cross-frame temporal reasoning.
- **Rel Direction and Route Plan essentially match the paper** (and Rel Direction even exceeds it by 6 points) — these are answerable from one or two salient frames.
- **Rel Distance is moderately down (−9)** — also primarily configurational.

So the gap concentrates on (a) tasks that need fine per-frame visual detail and (b) tasks that need to integrate across the whole frame sequence. Tasks that can be solved by picking a good single frame are unaffected.

## Implication for the suspect list

This pattern points away from the "only-cached scenes" hypothesis (Step 1 of the plan) — that would predict a more uniform downshift or scene-correlated bias. Three real suspects, in roughly descending magnitude:

### Suspect A — Model quantization (newly identified)

`experiments/eva-eval/scripts-remote/03_start_servers.sh:59` serves `OpenGVLab/InternVL2-8B-**AWQ**` (4-bit weight quantization, TurboMind backend). The paper's 37.5 is on **`OpenGVLab/InternVL2-8B`** (full precision, bf16; `literature/thinking-in-space/code/evaluate_all_in_one.sh:127` + `internvl2.py:208-214`).

AWQ INT4 quantization on dense visual reasoning is known to lose several points on benchmarks that rely on fine spatial discrimination. This was not on the original suspect list and is plausibly the largest single contributor.

### Suspect B — Frame transport / preprocessing

- Authors send a single string `"Frame1: <image>\nFrame2: <image>\n…Frame8: <image>\n{prompt}"` where `<image>` is a placeholder token bound to image embeddings, plus **dynamic multi-patch tiling at 448×448 + a thumbnail** per frame (≥2 patches per frame, so 16+ visual tokens total for 8 frames). See `literature/thinking-in-space/code/lmms_eval/models/internvl2.py:52-118, 343-350`.
- We send an **OpenAI-style multi-image content array** (one `image_url` base64-JPEG per frame, no positional template, no dynamic tiling). See `experiments/eva-eval/eva_eval/eval/baseline.py:44-51`.

Two concrete losses from this:

1. **Visual-token budget**: without dynamic tiling, each frame is 1 patch (~256 visual tokens) instead of 2+ (≥512). For tasks that depend on fine spatial detail (room_size, obj_size), halving visual tokens is plausibly catastrophic.
2. **Frame ordering / cross-frame coherence**: the model was trained to consume `Frame{i}:` placeholders; sending an opaque multi-image array drops that explicit positional cue. This squares with the Appearance Order collapse, which explicitly probes "first-time appearance order".

Rel Direction and Route Plan are typically answerable from a single salient frame, so neither degradation hurts them.

### Suspect C — Cached-scenes restriction

The full-test-set baseline doesn't suffer the cached-scenes bias (we ran on all 5130 questions, not on the 100Q stratified subset). So this is ruled out as a contributor to the *full-set* gap. Still worth noting for the 100Q subset comparison.

## What this means for the agent comparison

Our headline agent number is 30.68 on the 100-question stratified subset (`results/subset_fixed.jsonl`). We've been implicitly framing this as "agent (30.68) > raw VLM (~25)". The new framing should be:

- **vs. our own raw VLM under matched protocol** (lmdeploy multi-image array): agent gains ~5 points overall, with most of the win concentrated in route_planning (60 vs ~28).
- **vs. paper's reported raw VLM** (native HF + dynamic tiling): agent is currently *below* the paper's raw-VLM baseline (30.68 vs 37.5).

The right resolution is to run the raw VLM under the authors' protocol — either via lmdeploy native API + frame-placeholder template, or by invoking the authors' unmodified `lmms-eval` on the same 100 question IDs (Step 2 of the plan). Until we do, claims that the agent layer adds value over a *reproducible* raw VLM are unsupported.

## Decision

Skip refining Step 1 further. Move to Step 2: run the authors' unmodified `lmms-eval` with the full-precision (non-AWQ) InternVL2-8B and their native frame transport on a 100-question stratified sample from the test split. If the headline overall comes out close to 37.5, both Suspects A+B are the cause; we then have a path to either (i) match their protocol in our serving stack or (ii) honestly disclose our baseline is under their protocol and recompute the agent's headline accordingly.

## Where things stand right now

Reproduce the live table with `python experiments/eva-eval/scripts/compare_baselines.py --col ... --col ...` (see `scripts/compare_baselines.py`).

```
| question_type          |       agent (ours) | raw VLM (lmdeploy AWQ) |      paper Table 1 |
|------------------------|--------------------|------------------------|--------------------|
| overall                |               30.7 |                   25.3 |               37.5 |
| object_counting        |               20.9 |                   21.2 |               31.3 |
| object_abs_distance    |               23.6 |                   16.8 |               29.0 |
| object_size_estimation |               31.8 |                   27.5 |               48.9 |
| room_size_estimation   |               19.1 |                   15.1 |               44.2 |
| object_rel_distance    |               30.0 |                   29.2 |               38.0 |
| object_rel_direction   |               30.0 |                   39.4 |               33.4 |
| route_planning         |               60.0 |                   28.4 |               28.9 |
| obj_appearance_order   |               30.0 |                   24.6 |               46.4 |

n_questions: agent=100 (subset_fixed.jsonl), raw VLM lmdeploy=5130 (full set), paper=test split (~5130)
```

Honest reading:
- **Agent vs paper's raw VLM**: −6.8 overall. The agent is currently *below* the paper's reported raw-VLM number. Every per-task cell is below the paper except `route_planning` (+31.1) and `object_rel_direction` (−3.4).
- **Agent vs our raw VLM**: +5.4 overall, with the entire win concentrated in `route_planning` (+31.6) and modest gains on numeric/measurement tasks (+4 to +7).
- **Raw VLM vs paper**: −12.2 overall, with biggest drops on `room_size_estimation` (−29), `obj_appearance_order` (−21.8), `object_size_estimation` (−21.4).

## Resolution — bf16 split stack run (afternoon of 2026-05-13)

Both LLMs now run full-precision via the split stack documented in `scripts-remote/03_start_servers_bf16.sh`: Qwen-7B-Instruct (bf16) on morgen's 3090, InternVL2-8B (bf16) on neo's 5090, joined by SSH tunnel. Two 100-question stratified runs on `subset_fixed.jsonl`:

```
| question_type          | agent AWQ (100Q) | agent bf16 (100Q) | raw VLM AWQ (5130Q) | raw VLM bf16 (100Q) | paper Table 1 |
| overall                |        30.7      |        27.0       |        25.3         |       **40.5**      |       37.5    |
| object_counting        |        20.9      |        17.3       |        21.2         |        17.3         |       31.3    |
| object_abs_distance    |        23.6      |        23.6       |        16.8         |        41.8         |       29.0    |
| object_size_estimation |        31.8      |        17.3       |        27.5         |        36.4         |       48.9    |
| room_size_estimation   |        19.1      |        24.5       |        15.1         |        51.8         |       44.2    |
| object_rel_distance    |        30.0      |        50.0       |        29.2         |        60.0         |       38.0    |
| object_rel_direction   |        30.0      |        23.3       |        39.4         |        36.7         |       33.4    |
| route_planning         |        60.0      |        40.0       |        28.4         |        20.0         |       28.9    |
| obj_appearance_order   |        30.0      |        20.0       |        24.6         |        60.0         |       46.4    |
```

Reproduce with `python experiments/eva-eval/scripts/compare_baselines.py --col …`.

### What the table says

1. **The 12-point gap to paper was AWQ + dynamic-patch explosion.** A bf16 raw-VLM run on the stratified 100-question subset scores **40.5 overall**, slightly *exceeding* the paper's reported 37.5. Suspect A (AWQ quantization) was dominant; Suspect B (frame transport / dynamic tiling) contributed because lmdeploy's vision processor was reading `max_dynamic_patch=12` from `config.json` and expanding each 512×384 frame into ~13 patches (~3300 visual tokens). Resizing each frame to 448×448 in `eva_eval/eval/baseline.py:_encode_image_b64` forces 1×1 aspect ratio → 1 patch per frame, matching the authors' `max_num=1`. After that fix, prompts dropped from ~26 K tokens to ~2 K and lmdeploy stopped bouncing them.

2. **The agent is now *below* the bf16 raw VLM by 13 points** (27.0 vs 40.5). On every task except `route_planning` and `object_rel_distance`, calling the bf16 VLM directly beats running it through the agent. The agent's previous 5-point lead over the AWQ raw VLM was an artifact: the planner stayed reasonably useful while the VLM was crippled by INT4 quantization. With both at full precision, the agent's tool layer is destroying signal rather than aggregating it.

3. **Route planning is the agent's only durable win** (40-60 across both quantizations vs 20-29 for raw VLM and paper). That delta is real — structural multi-step reasoning where the agent's plan-then-act pattern actually helps. Everywhere else, the marginal "value" of the agent appears to be negative once the raw VLM is competent.

4. **The bf16 agent regressed vs. the AWQ agent** (27.0 → 30.7). Possibly noise on 100 questions per type (~12 per task), possibly Qwen2.5-7B-Instruct bf16 producing more verbose tool-use that confuses the agent's parsing, possibly the ReAct prompt being tuned to the quantized planner's idiosyncrasies. Worth checking but lower priority than the headline finding.

### Implications for the research direction

The headline framing "agent layer beats raw VLM" was load-bearing on the AWQ baseline. With a matched-precision raw VLM, that framing reverses on 7 of 8 task types. The honest comparison going forward is **bf16 agent vs bf16 raw VLM**, and the agent currently loses overall.

Two ways to read this:
- **Pessimistic**: the agent's tool stack (vocabulary fragmentation in YOLO labels, schema gaps in the Objects table, VLM estimation noise propagated through retrieval) injects more error than the raw VLM produces by itself. The whole "external spatial reasoning" thesis needs the raw VLM to be the bottleneck to make sense — and it isn't on this subset.
- **Constructive**: route_planning's persistent +20-point delta shows the *kind* of task where the agent does add value (multi-step, plan-then-act). Future agent versions should be measured on those tasks specifically, and the agent layer should be selective — only invoked when the question type warrants it. A meta-router that picks raw-VLM-direct for measurement/perception tasks and the agent only for route-planning-like tasks would dominate both columns.

### Open items

- The bf16 raw VLM number (40.5) is on 100 stratified questions, not the full 5130. Subset variance per task is ~10-15 points. Running the full set on bf16 would tighten the comparison to the paper's 37.5 — possible follow-up if the question is whether 40.5 is real or an unlucky stratification.
- The protocol divergence we previously hypothesized as Suspect B (OpenAI multi-image vs `Frame{i}:<image>` template) turns out to be effectively neutralized once we resize frames to 448×448 — lmdeploy's dynamic_preprocess then produces the same patch count as the authors' `max_num=1` setting. No need to pursue a Frame-template port further unless we're chasing the last ~3 points.
- The bf16 agent regression vs AWQ agent (-3.7 overall) is interesting and worth a trace inspection on the worst-degraded task types (object_size_estimation, obj_appearance_order both lost ~10-14 points).
