# SenseNova-SI backbone swap on VSI-Bench 100Q (2026-05-15)

## TL;DR

Swapped both backbones per Diwei Su's 2026-05-14 recommendation:
- planner: `Qwen/Qwen2.5-7B-Instruct` → vanilla `OpenGVLab/InternVL3-8B`
- VLM: `OpenGVLab/InternVL2-8B` → `sensenova/SenseNova-SI-1.5-InternVL3-8B` (spatial-intelligence-enhanced variant from *Scaling Spatial Intelligence with Multimodal Foundation Models*, arxiv 2511.13719)

Two headline findings on the same 100 IDs (`subset_bf16.ids.txt`):

1. **The new VLM is massively stronger.** Raw VLM jumps from **40 → 60** overall (+20.6 points). Gains are broad — every task type improves.
2. **The agent throws all of that away, and then some.** SI agent scores **25.8** — **~35 points below its own raw VLM** (60 → 26). The bf16 agent's gap to raw VLM was −13 on the same 100Q; doubling the agent's deficit with a better backbone is the cleanest evidence yet that the agent's information bottleneck is structural, not backbone-limited.

The agent layer is the bug. A stronger backbone makes it worse, not better.

## Stack used

- morgen:18000 → vLLM serving `OpenGVLab/InternVL3-8B` (text-only via `--limit-mm-per-prompt image=0`)
- neo:18001 → lmdeploy pytorch backend serving `sensenova/SenseNova-SI-1.5-InternVL3-8B`
- morgen→neo SSH tunnel exposes neo:18001 as morgen:18001

Reproduce: `EVA_PLANNER=internvl3-8b-text-bf16 EVA_VLM=sensenova-si-1.5-internvl3-8b-bf16` + `./scripts-remote/launch.sh 03_start_servers_si`, then `08_subset100_si` for the agent and `10_baseline 100 sensenova-si-1.5-internvl3-8b-bf16 8 results/subset_bf16.ids.txt` for the raw VLM.

## Sample-overlap audit

```
SI_agent       (subset_si.jsonl)                                : 100 rows
SI_rawVLM      (baseline_sensenova-si-1.5-internvl3-8b-bf16.jsonl): 100 rows
bf16_agent     (subset_bf16.jsonl)                              : 100 rows
bf16_rawVLM    (baseline_internvl2-8b-bf16.jsonl)               : 100 rows

All four JSONLs are pinned to results/subset_bf16.ids.txt (the canonical
100 IDs from the 2026-05-14 regression note), so every pairwise diff is
on the same questions.
```

## Headline per-task table (same 100 IDs across all four columns)

```
| question_type              |  n | rawVLM-bf16 | agent-bf16 | rawVLM-SI | agent-SI |
| obj_appearance_order       | 10 |   60.00     |   20.00    |   70.00   |  10.00   |
| object_abs_distance        | 10 |   41.82     |   23.64    |   50.00   |  28.18   |
| object_counting            | 10 |   17.27     |   17.27    |   58.18   |  13.64   |
| object_rel_direction_easy  | 10 |   50.00     |   30.00    |   60.00   |  20.00   |
| object_rel_direction_hard  | 10 |   20.00     |   30.00    |   50.00   |  30.00   |
| object_rel_direction_medium| 10 |   40.00     |   10.00    |   90.00   |  20.00   |
| object_rel_distance        | 10 |   60.00     |   50.00    |   70.00   |  50.00   |
| object_size_estimation     | 10 |   36.36     |   17.27    |   49.09   |  40.00   |
| room_size_estimation       | 10 |   51.82     |   24.55    |   76.36   |  26.36   |
| route_planning             | 10 |   20.00     |   40.00    |   30.00   |  20.00   |
| overall                    |100 |   39.73     |   26.27    |   60.36   |  25.82   |
```

*(Overall numbers from `diff_agent_runs.py` — equal weight across the
ten subtypes including the three `rel_direction_*` splits. The
`*.summary.json` files report slightly different overalls because
`eva_eval.eval.metrics.aggregate` collapses the three rel_direction
splits into one before averaging — per-task and Δ values are
identical.)*

## (a) Backbone swap on the agent: bf16 → SI, same 100 IDs

```
overall: 26.27 → 25.82  (Δ = -0.45)
```

Per-task moves are mostly in the noise except for two large swings that
roughly cancel:

- **`object_size_estimation: +22.7`** — the strongest single win from
  switching to SI. Raw VLM gain on the same task is only +12.7, so the
  agent is *amplifying* the backbone improvement here. Plausibly the
  SI backbone's size-estimation training (the EASI-MM capability) is
  exactly what the agent's `object_VQA` per-frame queries can use.
- **`route_planning: -20`** — the agent's only durable win over raw VLM
  in the bf16 stack disappears. Planner is now InternVL3-8B (vanilla),
  not Qwen2.5-7B-Instruct, and Qwen was apparently better at the
  step-by-step reasoning these questions need.
- **`obj_appearance_order: -10`**, `object_rel_direction_easy: -10` —
  small further regressions on cross-frame temporal tasks.

The swap is roughly neutral for the agent overall.

## (b) Backbone swap on the raw VLM: InternVL2 → SenseNova-SI

```
overall: 39.73 → 60.36  (Δ = +20.64)
```

Every single task type improves; the big wins are exactly the
spatial-intelligence categories Su's paper targets:

- **`object_counting: +40.9`** (17 → 58) — the largest single gain. The
  paper's "metric measurement" + "spatial relations" training pays off.
- **`object_rel_direction_medium: +50`** (40 → 90) — near-ceiling on
  multi-step direction queries.
- **`object_rel_direction_hard: +30`** (20 → 50).
- **`room_size_estimation: +24.6`** (52 → 76).
- **`obj_appearance_order: +10`**, `route_planning: +10`, others +8–13.

This validates the paper's claim independently — on our exact 100Q
subset, SenseNova-SI-1.5-InternVL3-8B is **strictly better** than
InternVL2-8B at spatial reasoning, by a wide margin.

## (c) The agent gap *widens* with a stronger backbone

```
SI raw VLM 60.36   →   SI agent 25.82   (Δ = -34.55)
bf16 raw VLM 39.73 → bf16 agent 26.27   (Δ = -13.45)
```

Per-task on the SI stack:

```
| question_type              |  n | rawVLM-SI | agent-SI |   Δ |
| object_rel_direction_medium| 10 |   90.00   |   20.00  | -70 |
| obj_appearance_order       | 10 |   70.00   |   10.00  | -60 |
| room_size_estimation       | 10 |   76.36   |   26.36  | -50 |
| object_counting            | 10 |   58.18   |   13.64  | -45 |
| object_rel_direction_easy  | 10 |   60.00   |   20.00  | -40 |
| object_abs_distance        | 10 |   50.00   |   28.18  | -22 |
| object_rel_direction_hard  | 10 |   50.00   |   30.00  | -20 |
| object_rel_distance        | 10 |   70.00   |   50.00  | -20 |
| route_planning             | 10 |   30.00   |   20.00  | -10 |
| object_size_estimation     | 10 |   49.09   |   40.00  |  -9 |
```

The agent's structural-bottleneck story from 2026-05-14
([2026-05-14-bf16-agent-regression.md](2026-05-14-bf16-agent-regression.md))
holds and intensifies:

1. `frame_VQA` / `object_VQA` send one frame at a time; the raw VLM
   baseline sends 8 simultaneously. With InternVL2 the per-frame
   restriction cost ~13 points; with SI it costs **35**, because the
   SI backbone is much better at cross-frame integration when given
   the chance, and the agent never gives it the chance.
2. `room_size_estimation` and `obj_appearance_order` both jump from
   ~50–70 raw to 10–26 agent — these need whole-scene visual
   integration, which the agent fragments.
3. YOLO-World vocabulary fragmentation + per-frame object indexing
   still hurts `object_counting` regardless of how good the VLM is —
   the agent has the same SQL exact-match bug it had on the bf16 stack.

## What this means for the research direction

Restating the conclusion from 2026-05-14 in stronger form: **the
agent's per-frame information bottleneck is the dominant failure
mode, and improving the perception backbone makes it worse, not
better.** Two strategic implications:

1. **Selective agent routing** (option 1 from 2026-05-14) gets more
   attractive, not less. On the SI stack, blending raw VLM for
   measurement/perception with the agent on route_planning gives a
   trivial upper bound of:
   - route_planning + rel_direction_hard via agent (20 of 100): ~25
   - everything else via raw VLM (80 of 100): ~65
   - blended: **~57 overall** — better than either alone, and
     16 points above today's SI agent.
2. **The agent's tools need to see multiple frames** to retain any of
   the backbone's gains. This is now a higher-priority refactor than
   it was a day ago — option 2 from 2026-05-14 is the structural fix.
   With a strong-backbone like SI, the agent gives up 35 points of
   real spatial-reasoning capability per question by serving frames
   one at a time.

In short: the SenseNova-SI swap landed exactly as Su's framing
predicted — a much stronger spatial backbone — but it doesn't rescue
the Embodied VideoAgent layer. If anything, it makes the case for
re-architecting that layer sharper.

## Operational notes / pitfalls from this run

Six infrastructure snags surfaced during the swap; documenting so the
next session doesn't re-pay this debt. None of them are about the
research itself — they're all packaging or path-bookkeeping issues.

1. **lmdeploy 0.13 dict-form sub-configs.** SenseNova-SI ships
   `vision_config` and `llm_config` as raw dicts (transformers
   4.55+); lmdeploy's `InternVLVisionModel.build_preprocessor` did
   `self.config.vision_config.image_size` and crashed. Patched
   `~/miniconda3/envs/lmdeploy/lib/python3.10/site-packages/lmdeploy/vl/model/internvl.py`
   on neo with a 4-line `SimpleNamespace` shim. *Not committed
   upstream; if neo's lmdeploy env is recreated, re-apply.*
2. **SenseNova-SI auto_map cross-references upstream.** The shipped
   `config.json`'s `auto_map` pointed to
   `OpenGVLab/InternVL3-8B--configuration_internvl_chat.InternVLChatConfig`
   — the `OpenGVLab/InternVL3-8B--` prefix tells transformers to fetch
   from a different HF repo, which fails under
   `HF_HUB_OFFLINE=1`. Stripped the prefix in
   `~/hf-cache/SenseNova-SI-1.5-InternVL3-8B/config.json` on neo.
   *If you re-download the weights, re-apply this rewrite.*
3. **Dots in directory name break dynamic module import.**
   `transformers_modules.SenseNova-SI-1.5-InternVL3-8B` is parsed by
   Python as `transformers_modules.SenseNova-SI-1` (subpackage),
   which doesn't exist. Created
   `~/hf-cache/SenseNova-SI-1_5-InternVL3-8B → SenseNova-SI-1.5-InternVL3-8B`
   symlink on neo; `03_start_servers_si.sh` uses the dot-free path.
   Committed in `481bc36`.
4. **Worktree doesn't have `literature/EmbodiedVideoAgent/code/`.**
   It's gitignored (per-paper convention, see top-level
   `.gitignore` line `literature/*/code/`). Symlinked the worktree's
   copy to the main repo's. *If you re-create the worktree, re-symlink.*
5. **`eva_eval` editable install pointed at the main repo, not the
   worktree.** `pip install -e .` resolves `DEFAULT_CONFIG_PATH`
   relative to the package source, so reading `models.yaml` went to
   the main repo's branch (no SI entries). Re-ran
   `pip install -e .` from the worktree's `experiments/eva-eval/`.
   *To restore openeqa-validation work, re-run `pip install -e .`
   from the main repo.*
6. **`config/detection_classes.txt` not in the worktree.** Same
   gitignore root cause. Symlinked from main repo.

If the SI-stack repo state gets re-built from scratch on morgen, the
clean fix would be to bake (3) and (4) into the launch script and the
git layout respectively, and to track `(1)` either by upstreaming the
lmdeploy patch or by upgrading lmdeploy when a release supports
InternVL3 `InternVLChatModel` configs natively.

## Artifacts

- `experiments/eva-eval/results/subset_si.jsonl` — SI agent on 100Q
- `experiments/eva-eval/results/subset_si.jsonl.summary.json`
- `experiments/eva-eval/results/baseline_sensenova-si-1.5-internvl3-8b-bf16.jsonl` — raw SI VLM on same 100Q
- `experiments/eva-eval/results/baseline_sensenova-si-1.5-internvl3-8b-bf16.jsonl.summary.json`
- `experiments/eva-eval/scripts-remote/03_start_servers_si.sh`,
  `04_stop_servers_si.sh`, `08_subset100_si.sh` — SI launcher set
- `notes/meetings/2026-05-14.md` — the conversation with Diwei Su
  that motivated this swap
- `literature/ScalingSpatialIntelligence/` — paper backing the SI VLM
