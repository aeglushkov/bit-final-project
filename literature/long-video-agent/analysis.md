# "LongVideoAgent" — Paper Analysis

## 1. Paper Summary

**"LongVideoAgent: Multi-Agent Reasoning with Long Videos"** (Liu et al., HKUST, ACL 2026 Main; arXiv 2512.20618, Dec 2025) proposes a multi-agent framework for hour-scale video question answering. A central **MasterAgent** LLM coordinates two specialist agents — a **GroundingAgent** that temporally localizes question-relevant clips and a **VisionAgent** that extracts fine-grained textual observations from those clips — over a bounded multi-round loop. For open-source master backbones (Qwen2.5-3B / 7B), the master is fine-tuned with **GRPO** using rule-based rewards (per-step structural validity + terminal answer correctness), while the grounding and vision agents stay frozen. The authors introduce two new episode-level benchmarks, **LongTVQA** and **LongTVQA+**, built by aggregating TVQA / TVQA+ clips into hour-scale TV episodes, and report large gains over non-agentic baselines — e.g., AgenticRL-Qwen2.5-7B goes from 46.10 → **60.20** on LongTVQA (+14.10) and from 60.30 → **70.80** on LongTVQA+ (+10.50), matching closed-source GPT-5-mini.

---

## 2. What Problem Does This Paper Solve?

Current long-video QA systems fail in one of two ways:

1. **Single-pass MLLMs** ingest the whole video in one context window, which forces aggressive compression / downsampling and loses fine-grained evidence (object identity, on-screen text, brief actions). Once this information is destroyed in the visual-to-text projection, no amount of downstream reasoning can recover it.
2. **Existing tool-augmented agents** (e.g., VideoAgent) use generic captioning / retrieval tools and do not train the planner — so they under-use the LLM's reasoning ability and miss subtle cues.

The paper's thesis is that long-video QA should be an *agentic*, *iterative*, and *RL-trainable* process, where the LLM actively decides **what to look at, when to look again, and when enough evidence has been gathered**.

---

## 3. Method

### 3.1 System Architecture

The system is a bounded, multi-round action loop with three agents (Figure 2 in the paper):

```
User question + subtitles
        │
        ▼
   ┌────────────┐        <request_grounding>
   │ MasterAgent│ ───────────────────────────▶ GroundingAgent  ──▶ returns <clip_X>
   │   (LLM)    │        <visual_query>
   │            │ ───────────────────────────▶ VisionAgent     ──▶ returns textual observation
   │            │        <answer>
   │            │ ───────────────────────────▶ final answer to user
   └────────────┘
        ▲                (loops for ≤ K rounds)
        │
     accumulates subtitles + clip tags + vision outputs
```

The master sees only **text**: subtitles, the question, any `<clip_X>` tags the grounding agent has returned, and any textual observations from the vision agent. Raw frames never enter the master's context — all visual information is filtered through the vision agent. This is what makes the architecture model-agnostic: any instruction-tuned LLM can serve as the master.

### 3.2 Action Schema and System Prompt

The system prompt (Table 1 in the paper) defines exactly three allowed actions per turn, each ending in an XML-like closing tag:

| Action | Tag | When to use |
|---|---|---|
| **Visual query** | `<visual_query>…</visual_query>` | Current text info is insufficient; need visual details conditioned on the current `<clip_X>` |
| **(Re)grounding** | `<request_grounding>` (self-closing) | No clip yet, or current clip contradicts the accumulating evidence |
| **Answer** | `<answer>…</answer>` | Evidence is sufficient — emit the final concise MCQ answer and terminate |

The prompt also imposes soft guidelines: (1) be conservative with tool calls; (2) do not hallucinate visual details — only call the vision agent for facts not inferable from subtitles; (3) each turn targets the *current* `<clip_X>` — if none exists, prefer (re)grounding before visual query.

### 3.3 Multi-Turn Rollout (Algorithm 1)

Pseudocode mirrors Algorithm 1 from the paper:

```
Inputs: subtitles S, question q, video V, master π_θ, max steps K
rollout y ← ∅;  t ← 0
while t < K:
    generate action string y_t from π_θ(· | S, q, V, y) until it closes with
      </visual_query> | </request_grounding> | </answer> | <eos>
    y ← y + y_t
    if y_t contains <visual_query>:
        d ← VisionAgent(parse(y_t), V)
        y ← y + d
    elif y_t contains <request_grounding>:
        clipTag ← GroundingAgent(q, S)
        y ← y + clipTag + S(clipTag)             # localized subtitles appended
    elif y_t contains <answer>:
        return parse(y_t)                         # terminate
    else:
        y ← y + "The action is not correct. Only <visual_query>, <request_grounding>, or <answer>."
    t ← t + 1
return y
```

Two design points worth flagging:

- **Grounding returns localized subtitles too.** When a clip is grounded, both the tag and the subtitles inside that clip's window are appended, so grounding improves *textual* context as well as giving the vision agent a handle.
- **Malformed actions are reprimanded, not silently ignored.** The master receives an explicit "action is not correct" message and retries — giving RL a clean signal for structural validity.

### 3.4 RL Formulation (GRPO, rule-based rewards)

Long-video QA is cast as a finite-horizon decision process. For each rollout τ, two rewards are summed:

- **Structural validity**   `r^fmt_t ∈ {0,1}` — grants 1 if the action string at step t contains exactly one top-level tag with proper closure and no extraneous text, else 0.
- **Answer correctness**   `r^ans ∈ [0,1]` — awarded at termination via exact match against the multiple-choice answer; if no valid `<answer>` appears, `r^ans = 0`.

The trajectory-level reward is:

```
R(τ) = α · Σ_t r^fmt_t   +   r^ans
```

with α > 0 weighting the per-step structural shaping. GRPO (standard clipping + entropy regularization) optimizes the master policy on sampled rollouts; advantages are computed sequence-level with a learned value baseline. The grounding and vision agents remain *frozen* throughout — only the master's weights move.

**Why GRPO and not PPO?** The paper does not give a direct comparison; GRPO is a natural fit because each question produces N parallel rollouts from the same initial state, and group-relative advantages remove the need for a separate critic over long, tool-interleaved trajectories.

---

## 4. Datasets: LongTVQA / LongTVQA+

### 4.1 Motivation

TVQA and TVQA+ (Lei et al., 2018/2020) are standard VideoQA datasets but their clips are short (60–90s), which does not stress long-context reasoning. The authors *aggregate* all clips belonging to the same TV episode into a single hour-scale sequence — and keep every ancillary annotation intact.

### 4.2 Construction

| Source | QAs | Clips | Notes |
|---|---|---|---|
| TVQA | 152.5K | 21.8K (60–90s each) | Subtitles + moment annotations across six TV shows |
| TVQA+ | 29.4K | 4,198 | Adds precise timestamps + 310.8K frame-level bounding boxes on referenced entities, mainly from *The Big Bang Theory* |

Aggregation:

1. Merge the visual stream, subtitles, and all associated questions per episode.
2. Re-index clip timestamps into the episode timeline.
3. For TVQA+, preserve bounding boxes at their corresponding frames.

The result is **LongTVQA** (coarser, episode-only) and **LongTVQA+** (with spatial grounding). Experiments report results on the original validation splits after this episode-level aggregation.

---

## 5. Experiments

### 5.1 Setup

| Setting | Default |
|---|---|
| GroundingAgent | Grok-4-fast-reasoning |
| VisionAgent | GPT-4o |
| Evidence window | 1 clip (no adjacent clips) |
| Max steps K | 5 |
| Primary metric | Answer Accuracy (MCQ) |
| Secondary metric | Grounding Accuracy (in localization ablations) |

**Training (open-source masters).** Qwen2.5-3B / 7B masters are fine-tuned with GRPO: lr 5 × 10⁻⁶, up to 2,000 optimization steps, KL coefficient 10⁻³, batch 4, rollout count N = 4, temperature 1.0. On 4 × H800, the 7B variant trains in ~12 h and the 3B variant in ~6 h.

**Non-agent baselines.** The same base LLM serves as the master but consumes the full subtitles directly and does not invoke grounding or vision. Closed-source GPT-4o and Gemini-2.5 Pro baselines additionally process the full long video (Subtitle + Frame input) without any agentic scaffolding.

### 5.2 Main Results (Table 2)

Numbers are validation Answer Accuracy (%) on LongTVQA / LongTVQA+. Parenthesized green deltas are over the *immediately preceding* non-agentic or non-RL row.

**Closed-source baselines and agentic variants:**

| Method | Multi-agent | Input | RL | LongTVQA | LongTVQA+ |
|---|:---:|---|:---:|---:|---:|
| GPT-4o | ✗ | Subtitle+Frame | ✗ | 70.78 | 78.32 |
| Gemini-2.5 Pro | ✗ | Subtitle+Frame | ✗ | 78.90 | 81.28 |
| GPT-5-mini | ✗ | Subtitle | ✗ | 62.40 | 66.70 |
| Agentic-GPT-5-mini | ✓ | Subtitle+Frame | ✗ | 71.11 (+8.71) | 78.90 (+12.20) |
| Grok | ✗ | Subtitle | ✗ | 76.90 | 81.80 |
| **Agentic-Grok** | ✓ | Subtitle+Frame | ✗ | **82.65 (+5.75)** | **85.60 (+3.80)** |

**Open-source LLMs:**

| Method | Multi-agent | Input | RL | LongTVQA | LongTVQA+ |
|---|:---:|---|:---:|---:|---:|
| DeepSeek-R1 671B | ✗ | Subtitle | ✗ | 68.99 | 75.04 |
| Agentic-DeepSeek-R1 671B | ✓ | Subtitle+Frame | ✗ | 70.30 (+1.31) | 79.70 (+4.66) |
| Agentic-Qwen2.5-3B | ✓ | Subtitle+Frame | ✗ | 23.50 | 27.70 |
| AgenticRL-Qwen2.5-3B | ✓ | Subtitle+Frame | ✓ | 47.40 (+23.90) | 50.10 (+22.40) |
| Agentic-Qwen2.5-7B | ✓ | Subtitle+Frame | ✗ | 46.10 | 60.30 |
| **AgenticRL-Qwen2.5-7B** | ✓ | Subtitle+Frame | ✓ | **60.20 (+14.10)** | **70.80 (+10.50)** |

**Takeaways:**

1. The multi-agent framework is a uniformly positive delta across *every* backbone tested.
2. Agentic RL is essential for small open-source models — Qwen2.5-3B is unusable without it (23.5%) but becomes competitive with it (47.4%).
3. **AgenticRL-Qwen2.5-7B reaches parity with closed-source GPT-5-mini (60.20 vs 62.40)** under the same agentic protocol, despite being ~100× smaller.
4. Agentic-Grok is the overall winner at 82.65 / 85.60 — closed-source master + closed-source grounding/vision agents + no RL.

### 5.3 Ablations (Table 4)

**(a) Contribution of each component** — starting from a text-only baseline, grounding and vision each add clean, additive gains:

| Setting | Acc (%) |
|---|---:|
| Non-agent (text-only) | 64.3 |
| Multi-agent (Grounding only) | 69.0 (+4.7) |
| Multi-agent (Grounding + Vision) | **74.8** (+5.8) |

Grounding alone narrows the context the LLM has to reason over; vision then supplies the fine-grained cues that subtitles cannot express.

**(b) Max steps K** — increasing the per-question action budget saturates around K = 5:

| K | Grounding Acc (%) | Answer Acc (%) |
|---|---:|---:|
| 2 | 67.00 | 68.30 |
| 5 | 71.00 | 73.67 |
| 10 | 72.00 | 73.67 |

Extra steps beyond K = 5 still nudge grounding accuracy up slightly but give no additional answer-accuracy gains — the remaining errors are not fixable by more tool calls.

**(c) Evidence window size** — expanding the grounding window from 1 to 3 adjacent clips yields the single biggest controllable headroom in the paper:

| Window | Grounding Acc (%) | Answer Acc (%) |
|---|---:|---:|
| 1 | 71.67 | 70.33 |
| 2 | 78.67 (+7.00) | 75.00 (+4.67) |
| 3 | **81.94** (+3.27) | **77.26** (+2.26) |

The paper still defaults to Window = 1 because larger windows require more visual queries and latency, not because Window = 1 is optimal.

**(d) VisionAgent backbone** — stronger perception → better answers:

| Vision model | Grounding Acc (%) | Answer Acc (%) |
|---|---:|---:|
| Qwen3-VL-235B | 71.00 | 73.67 |
| GPT-4o | **73.30** | **78.00** |

The gain is large enough (+4.33 answer acc) that the authors adopt GPT-4o as the default.

### 5.4 Qualitative Traces

Tables 3 and 5 show full execution traces where the master iteratively refines its understanding. In Table 5 ("What side of the bed is Sheldon when he is closer to the window?"), the master first grounds the scene, then issues a visual query ("bedroom scene layout"), receives an ambiguous description, *re-issues a more targeted visual query* ("which side of the bed is next to the window"), and finally answers correctly. These traces illustrate the system's ability to re-query when early observations are ambiguous — behavior that emerges from the RL training.

---

## 6. Code Overview

The authors' repository is under `literature/long-video-agent/code/` (read-only — do not modify per project convention):

```
long-video-agent/code/
├── README.md                  # install + quickstart
├── VERL_README.md             # the verl RL stack
├── src/
│   ├── dataset/               # LongTVQA / LongTVQA+ loading
│   ├── evaluation/            # eval harness
│   ├── smoke_test/
│   └── utils/
├── videoagent/
│   ├── action_generation.py   # rollout / action tokenization
│   ├── reward.py              # rule-based structural + answer reward
│   └── tensor_helper.py
├── verl/                      # vendored verl RL framework
├── recipe/                    # training recipes
├── scripts/
│   ├── download_and_prepare_longtvqa.sh
│   ├── eval_unified_api.sh
│   ├── eval_unified_local.sh
│   ├── merge_lora_adapter.sh
│   ├── quickstart_qwen_2_5_3B_grpo.sh
│   ├── train_qwen_2_5_7B_grpo.sh
│   └── train_qwen_2_5_7B_dapo.sh
└── setup.py
```

**Install (from `README.md`):**

```bash
conda create -n lvagent python=3.11
conda activate lvagent
pip install vllm
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

**Branches.** `main` is the stable training path; `newversion` is an experimental stack rebuilt on a newer `verl` — the released `longvideoagent-qwen2.5-7b` checkpoint was trained on `newversion`.

**Published checkpoints (HF `longvideoagent/`):**
- `longvideoagent-qwen2.5-3b`
- `longvideoagent-qwen2.5-7b`

**Entry points worth reading first** when exploring the code: `videoagent/action_generation.py` (the rollout loop that implements Algorithm 1), `videoagent/reward.py` (the rule-based reward), and `scripts/train_qwen_2_5_7B_grpo.sh` (the canonical training invocation).

---

## 7. Critical Analysis

### 7.1 Strengths

1. **Clean, minimal action schema.** Exactly three actions with XML-like closing tags gives RL a crisp structural-validity signal that does not require dense per-token rewards.
2. **Rule-based rewards only.** Structural validity + terminal answer correctness is enough to learn multi-step tool coordination — no learned reward model, no human preference data.
3. **Large gains on small open-source backbones.** AgenticRL takes Qwen2.5-3B from 23.5 to 47.4 on LongTVQA (+23.9). The framework clearly lifts underpowered models.
4. **Model-agnostic framing.** Because the master only ever sees text, any instruction-tuned LLM plugs in — closed-source or open-source — without architectural changes.
5. **Clean additive ablation.** The grounding → vision contribution split (64.3 → 69.0 → 74.8) gives a clear decomposition of where the gains come from, rarely this clean in agentic-system papers.
6. **Practical release.** Training code, evaluation code, both datasets, and two checkpoint sizes are all public.

### 7.2 Limitations and Open Questions

1. **Proprietary frozen agents.** The default GroundingAgent is Grok-4-fast-reasoning and the default VisionAgent is GPT-4o — both closed-source API models. Practical reproducibility and per-query cost are non-trivial, and the Qwen3-VL-235B vision ablation shows ~4 points of accuracy are left on the table with the best open-source alternative.
2. **TV-show-only evaluation.** LongTVQA and LongTVQA+ are both derived from TVQA / TVQA+, which are dominated by sitcoms (heavily TBBT). Generalization to movies, lectures, egocentric / instructional video, or surveillance footage is untested.
3. **Window = 1 is a cost compromise, not an optimum.** Table 4c shows Window = 3 delivers +6.93 answer-accuracy over Window = 1 — a larger headroom than some of the main architectural gains — but the paper still reports Window = 1 as default. A curriculum or adaptive-window policy would be a natural follow-up.
4. **K = 10 plateau is unexplained.** Grounding accuracy still rises from K = 5 → 10 (71.0 → 72.0) while answer accuracy is flat. Is this a policy ceiling (the master can't turn better grounding into better answers) or an environment ceiling (remaining errors are not fixable with more tool calls)? The ablation does not distinguish these.
5. **Reward shaping is minimal.** No dense perception reward, no grounding-accuracy reward, no step-cost penalty beyond the implicit length penalty from K. It is unclear whether fancier shaping would lift small models further or whether the rule-based reward is already near-optimal.
6. **No head-to-head with VideoAgent / VideoTree / Koala.** Related work is cited but the main results table compares only to non-agentic baselines of the same backbones, not to other agentic systems — so the magnitude of the delta over prior agentic work is not directly measured.
7. **Bounding-box annotations underused.** LongTVQA+ preserves 310.8K frame-level bounding boxes but nothing in the main method consumes them — they appear to be retained "for future work" rather than used as supervision for grounding or vision agents.

---

## 8. Quick Reference

| | |
|---|---|
| **Title** | LongVideoAgent: Multi-Agent Reasoning with Long Videos |
| **Authors** | Runtao Liu\*, Ziyi Liu\*, Jiaqi Tang, Yue Ma, Renjie Pi, Jipeng Zhang, Qifeng Chen |
| **Affiliation** | HKUST |
| **Venue** | ACL 2026 Main (arXiv 2512.20618, 23 Dec 2025) |
| **Architecture** | MasterAgent (LLM) + GroundingAgent (temporal) + VisionAgent (perception), ≤ K-round loop |
| **Training** | GRPO on master only; frozen grounding + vision agents; rule-based rewards (`r^fmt + r^ans`) |
| **Benchmarks** | LongTVQA, LongTVQA+ (episode-level aggregation of TVQA / TVQA+) |
| **Default tools** | Grounding = Grok-4-fast-reasoning; Vision = GPT-4o; K = 5; Window = 1 |
| **Open-source masters** | Qwen2.5-3B, Qwen2.5-7B (checkpoints on HF) |
| **Headline numbers** | AgenticRL-Qwen2.5-7B: **60.20 / 70.80** (+14.1 / +10.5) · Agentic-Grok: **82.65 / 85.60** |
| **Best ablation** | Grounding + Vision: 64.3 → 74.8 (+10.5) over text-only baseline |
| **Code** | `github.com/longvideoagent` (vendored in `code/`) |
| **Datasets** | `huggingface.co/datasets/longvideoagent/LongTVQA{,_plus}` |
