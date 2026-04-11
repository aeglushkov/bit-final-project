# Paper Analysis: ReAgent-V — A Reward-Driven Multi-Agent Framework for Video Understanding

**Authors:** Yiyang Zhou*, Yangfan He*, Yaofeng Su, Siwei Han, Joel Jang, Gedas Bertasius, Mohit Bansal, Huaxiu Yao (UNC-Chapel Hill; University of Washington)
**Venue:** NeurIPS 2025 (arXiv:2506.01300, June 2025)
**Date read:** 2026-04-11

**Links:** [Paper](https://arxiv.org/abs/2506.01300) | [Code](https://github.com/aiming-lab/ReAgent-V)

---

## 1. Core Contribution

ReAgent-V is a modular agentic framework that turns single-pass video-LVLM inference into a three-stage pipeline — entropy-calibrated frame selection, tool-augmented reasoning, and critic-driven multi-perspective reflection — while producing *inference-time* reward signals that double as a data-curation source for downstream SFT / DPO / GRPO training. The single main claim is that a lightweight, training-free orchestration layer on top of existing VLMs yields consistent gains (+6.9% video understanding, +2.1% video reasoning, +9.8% VLA alignment) with fewer frames and smaller training sets than conventional reward-model pipelines, *without* retraining the base VLM.

The novelty sits in the coupling: reward signals are not an offline post-hoc score but a live evaluation report that (a) triggers the reflection loop with fresh tool calls and (b) labels samples as "reflection-worthy" for later RL fine-tuning.

---

## 2. Relevance to Our Research

### 2.1 Relevance Rating: **Medium**

ReAgent-V shares our architectural bet — an external agent layer on top of a VLM — but does not address spatial reasoning mechanistically. Spatial information flows as unstructured text (scene-graph triples, "upper-right corner", object tags) rather than through geometric representations. VSI-Bench is one of six reasoning benchmarks in Table 2, but it is used as an evaluation target for data curation, not as a motivation for any spatial primitive.

That said, ReAgent-V is the most directly usable **scaffold** we have seen: the tool-factory pattern, the critic rubric, and the three-persona reflection loop are all orthogonal to the reasoning domain and could be specialized for spatial tasks.

### 2.2 Key Takeaways for Our Direction

- **Validates the VLM-as-perception / agent-as-reasoning decomposition.** The paper explicitly frames reflection as correcting a single-pass VLM rather than replacing it, which mirrors our hypothesis that the VLM handles perception and an external layer handles reasoning.
- **Reusable reflection template.** The three-persona rubric (conservative = answer-only edit, neutral = entity-level correction, aggressive = rewrite the reasoning chain) maps naturally to three spatial failure modes we care about: answer slip, perception error, and ego-allo transform error.
- **LLM-as-judge rubric is extensible.** The five critic dimensions (visual alignment / temporal accuracy / option disambiguation / reasoning specificity / linguistic precision) are all language-level — none of them probe spatial grounding. Adding an "ego-allo consistency" or "metric plausibility" axis is a concrete first experiment.
- **Importance score as a reflection trigger.** Curating GRPO training data by keeping only samples with importance score < 5/10 is a cheap, automatic hard-sample miner that we could run directly on VSI-Bench.
- **Challenge to an assumption:** Table 4 shows that the aggressive persona *alone* (which rewrites the reasoning chain) degrades accuracy on three of four backbones — the paper's own evidence that naive reasoning-chain rewrites hurt more than they help. This is evidence *for* external structured reasoning rather than free-form LLM re-reasoning.

### 2.3 Perception vs. Reasoning Evidence

Our core hypothesis: VLMs are strong at perception, weak at spatial reasoning.

| Aspect | Evidence from ReAgent-V |
|---|---|
| Perception capability | Treated as a solved substrate — tools (OCR, Grounding-DINO, scene graph, CLIP, caption model) provide perceptual grounding; VLM consumes tool outputs as text. |
| Reasoning capability | Explicitly delegated to the agent layer: critic + three reflection personas + meta-agent fusion. Paper acknowledges that "static reasoning hampers the model's ability to self-correct" (§1). |
| Ego-allo transformation | **Absent.** No component handles frame-of-reference transformations; spatial info is semantic tokens only. |
| Error attribution | Partially addressed via the five-dimension critic rubric and per-persona ablation (Table 4), but none of the dimensions isolate spatial errors. |

---

## 3. Method

### 3.1 Approach Summary

Three-stage pipeline (Algorithm 1, p. 4) per (video V, query Q) pair:

1. **Stage 1 — Frame selection.** ECRS (§2.1) iteratively picks frames until the threshold is exceeded, with a fallback that guarantees ≥32 frames.
2. **Stage 2 — Tool-augmented reasoning.** The target agent selects a subset `T' ⊆ T` from the tool factory, runs each tool on `(Q, F)` to get intermediate results `R`, and generates an initial answer `A_0`.
3. **Stage 3 — Evaluation and reflection.** A critic agent inspects `A_0`; if rejected, it generates sub-questions `{q_i}`, re-invokes tools, produces an evaluation report `E` with scalar reward and structured feedback, and then the target agent regenerates the answer from three perspectives `(t_c, t_n, t_a)`. A meta-agent fuses them if all confidences exceed 0.6, else picks the highest-confidence revision.

### 3.2 Architecture (from Figure 1, p. 2)

```
Video + Query
     │
     ▼
┌───────────────────────────┐
│ Stage 1: ECRS             │  CLIP sim × RGB-entropy → iterative threshold
│  (Frame Selection)        │  → keyframe set F
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 2: Tool Factory     │  target agent picks T' ⊆ {OCR, ASR,
│  + Target Agent           │   Grounding-DINO, Scene Graph, CLIP,
│                           │   SharedGPT4Video, Caption}
│                           │  → R = tool outputs → initial answer A_0
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 3: Critic +         │  critic emits sub-questions {q_i},
│  Multi-Perspective        │   updates tools, produces report E
│  Reflection               │  → 3 persona answers (t_c, t_n, t_a)
│                           │   with confidence p^(t)
│                           │  → meta-agent fuse or highest-confidence
└───────────┬───────────────┘
            │
            ▼
        Final answer A_final
                │
                ▼
      Reward trace → SFT / DPO / GRPO curation
```

Modularity is explicit: target agent, critic agent, task templates, and tool set are all configurable; the framework is training-free and adds no parameters to the base VLM.

### 3.3 Spatial Representation

| Representation | Description | Strengths | Weaknesses |
|---|---|---|---|
| Scene-graph text triples | "man (left of) woman", "bottle (on) table" — generated from Grounding-DINO detections and baked into the multimodal prompt | Composable; model-agnostic; cheap | Lossy; discards geometry; no metric scale |
| Bounding-box text descriptions | "object in upper-right of frame" — frame-relative region annotations in prompts | Easy to inject; interpretable | Frame-relative only (no 3D); no ego-allo mapping |
| Caption model output | Scene-level natural-language captions | Integrates holistic context | Coarse; not spatially grounded |

**No cognitive map, no metric coordinates, no ego-centric transforms.** The framework treats every spatial question as a language-level reasoning problem over semantic tokens.

### 3.4 ECRS Frame Selection (§2.1, Eqs. 1–4)

For each frame `f_i` and query `Q`, compute CLIP cosine similarity `s_i = e_i·q / (‖e_i‖‖q‖)`.

Per-frame entropy is the average over R/G/B channel histogram entropies (Eq. 2):

```
H_i = (1/3) Σ_{c∈{R,G,B}} − Σ_{j=0}^{255} p_{j,c}^{(i)} log₂ p_{j,c}^{(i)}
```

ECRS (Eq. 3) combines the two:

```
ECRSᵢ = sᵢ · Hᵢ / Σ_k H_k
```

Iterative selection (Eq. 4) applies an exponentially-increasing threshold `k·α^m·τ` (with `m` = iteration index) so later rounds keep only progressively higher-ECRS frames. Minimum of 32 frames is guaranteed by falling back to top-ECRS frames from the previous round.

**Motivation (§2.1 and Figure 2):** pure CLIP-score selection tends to pick many semantically similar frames (e.g., all frames containing a person, for questions about a single action) — ECRS suppresses these low-entropy duplicates. Code: `code/ReAgent-V/ReAgentV_utils/frame_selection_ecrs/ECRS_frame_selection.py`.

Effectiveness is demonstrated in Table 3: ECRS cuts per-sample inference time substantially (e.g., LLaVA-Video-72B: 83.2s → 68.2s on LongBench) while improving accuracy across backbones.

### 3.5 Tool Factory (§2.2)

Tools available: OCR, ASR, Grounding-DINO, scene graph, CLIP retrieval, SharedGPT4Video, caption model (the paper's Figure 1 shows the factory set). Tool selection is prompt-based: the target agent is given a tool-selection prompt describing each tool's function and chooses `T' ⊆ T` based on the query. Selected tools run against `(Q, F)` and outputs are concatenated into the context for the initial answer and later re-run with new sub-questions during reflection.

Code implementation: `code/ReAgent-V/ReAgentV_utils/tools/tool_selection.py` and `extract_modal_info.py` orchestrate the binary per-tool flags and gather outputs into a single `modal_info` dict.

### 3.6 Critic Agent and Evaluation Report (§2.3)

The critic is the same LVLM run with a critic prompt. It:
1. Decides whether `A_0` is satisfactory (binary trigger for reflection).
2. Generates sub-questions `{q_i}` to localize errors.
3. Re-invokes tools from `T` with the new sub-questions, producing `R_update`.
4. Emits an **evaluation report** `E` containing a scalar reward and structured feedback along five dimensions, each scored 0.0–5.0:
   - Visual Alignment
   - Temporal Accuracy
   - Option Disambiguation
   - Reasoning Specificity
   - Linguistic Precision

Figure 5 (case study, p. 8) shows a real evaluation report with all five scores populated. The `importance_score` used for GRPO curation is the critic's overall sufficiency signal derived from this report — the paper states samples with importance score **< 5 out of 10** are retained as reflection-worthy GRPO training examples (p. 6, "Video LLM Reasoning"). The exact composition of the importance score from the five dimensions is not made explicit in the paper.

### 3.7 Multi-Perspective Reflection (§2.3, Eq. 5, Appendix B)

Three persona prompts produce revised answer/confidence pairs:

- **Conservative (t_c):** changes only the final answer, never the reasoning chain.
- **Neutral (t_n):** updates *entities* in the scene (object identity, color, position) based on updated context, preserving the reasoning chain.
- **Aggressive (t_a):** rewrites both the reasoning steps and the entities.

Each persona computes `p^(t) = P(A^(t) | F, Q, R_update, E)`. Algorithm 1 line 16 specifies the aggregation rule:

```
if min(p^(c), p^(n), p^(a)) > 0.6:
    A_final ← meta-agent fusion of the three revised answers
else:
    A_final ← argmax_{t} p^(t)
```

So the paper uses a single 0.6 threshold across personas. (The code in `ReAgentV.py` wires in separate per-persona thresholds 0.6 / 0.7 / 0.8 — an implementation detail that diverges slightly from the published algorithm.)

Appendix B contains the full prompt templates for the critic, the three personas, and the meta-agent fusion step. The meta-agent is instructed to extract common components, remove contradictions, and synthesize a single unified answer.

---

## 4. Agent / Decomposition Aspects

### 4.1 Task Decomposition

Fully decomposed, explicitly staged pipeline — not free-form ReAct. The decomposition is fixed (three stages, fixed number of personas) but each stage is prompt-driven and configurable. No learned orchestrator; the "decisions" are (a) which tools to invoke, (b) whether to trigger reflection, and (c) how to aggregate persona outputs — all performed by the same VLM with different prompts.

### 4.2 Multi-Turn VLM Interaction

Yes. A single query triggers at minimum six VLM calls:
1. Tool selection (initial)
2. Initial answer generation `A_0`
3. Critic evaluation (sub-question generation + evaluation report `E`)
4–6. Three persona reflections
7. (Conditional) Meta-agent fusion

No dynamic loop termination — the same number of calls runs on every sample, regardless of difficulty. Table 3 shows a clear inference-time cost, though ECRS claws some of it back.

### 4.3 External Reasoning Modules

External *tools* (OCR, Grounding-DINO, Whisper-like ASR, scene graph generator) exist, but all reasoning — including critic scoring, sub-question generation, and persona reflection — stays inside the VLM via different prompts. There is **no symbolic reasoning, no geometric computation, no coordinate transform, no planning algorithm.** In our framing, ReAgent-V externalizes tool invocation but keeps reasoning inside the VLM.

### 4.4 Implications for Our Agent Design

**Adopt:**
- Three-stage scaffold (frame selection → tool-augmented answer → critic + reflection).
- Prompt-level tool routing via a tool factory the target agent inspects.
- Structured critic rubric as the reward mechanism (cheap, training-free, interpretable).
- `importance_score < threshold` as a hard-sample miner on VSI-Bench.

**Adapt:**
- Rework the critic rubric to include a spatial-reasoning axis (ego-allo consistency, metric plausibility, object-identity continuity).
- Rework the three personas around spatial failure modes instead of answer-edit scope:
  - Perception-corrector (was the object correctly identified?)
  - Frame-corrector (was it visible in the selected frames?)
  - Geometry-corrector (was the ego-allo transform right?)
- Add explicit spatial tools to the factory: depth estimator, camera-pose estimator, metric-scale probe, 3D bounding box detector.

**Avoid:**
- Treating spatial info as unstructured text tokens inside the prompt.
- Running reflection on every sample — the paper's own Table 4 shows that on easy samples the reflection can *hurt* (e.g., Qwen2-VL-7B drops from 58.3 to 58.2 on VideoMME with reflection enabled). An early-exit based on initial confidence is a cheap win.
- Pure aggressive-style reasoning rewrites — Table 4 and Figure 4 show the aggressive persona alone is the weakest of the three.

---

## 5. Evaluation

### 5.1 Benchmarks & Metrics

The paper uses **12 datasets across three applications** (plus SIMPLER for VLA). Only video benchmarks shown below.

| Benchmark | Tasks | Metric | Relevant to us? |
|---|---|---|---|
| **VideoMME** | Short/Medium/Long video QA | Accuracy | Medium |
| **LongBench** | Long-form video understanding | Accuracy | Medium |
| **NextQA** | Next-frame reasoning | Accuracy | Low |
| **EgoSchema (subset)** | Egocentric video QA | Accuracy | Medium — egocentric framing |
| **LVBench** | Long video understanding | Accuracy | Low |
| **MLVU** | Multi-task long video | Accuracy | Low |
| **VSI-Bench** | Visual-spatial intelligence (Yang et al. 2025) | Accuracy (MRA for NA tasks) | **High** — directly measures what we care about |
| **VideoMMMU** | Multi-discipline video knowledge | Accuracy | Low |
| **MMVU** | Multi-task video understanding | Accuracy | Low |
| **MVBench** | Multi-task video reasoning | Accuracy | Medium |
| **TempCompass** | Temporal reasoning | Accuracy | Low |
| **SIMPLER (VLA)** | Robotic task completion | Success rate | Medium — embodied VLA downstream |

ReAgent-V is evaluated on VSI-Bench *only* in the data-curation application (Table 2) — the framework is not run as an end-to-end inference wrapper on VSI-Bench. So what VSI-Bench measures here is the quality of the GRPO training data curated *by* ReAgent-V, not the quality of ReAgent-V's own spatial answers.

### 5.2 Key Results

**Table 1 — Video Understanding.** ReAgent-V consistently lifts strong backbones:
- **Qwen2.5-VL-72B + ReAgent-V:** LongBench 66.4 (+5.9 over 60.5), NextQA 84.3, EgoSchema 76.2, LVBench 41.2, MLVU 74.2, VideoMME overall **75.1** (+1.0 over 74.1 baseline, matching GPT-4o at 71.9).
- **LLaVA-Video-72B + ReAgent-V:** LongBench 64.9 (+4.2), VideoMME overall 73.5 (+2.5).
- **Qwen2.5-VL-7B + ReAgent-V:** LongBench 54.3 (+7.6), EgoSchema 61.9 (+6.6).

Average improvement ceiling reported as **+6.9%** (abstract).

**Table 2 — Video Reasoning via Data Curation (Qwen2.5-VL-7B base).** ReAgent-V selects 52k samples from the Video-R1-260k pool using evaluation-report importance scores < 5/10, then trains via GRPO:

| Strategy | Data | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|---|---|---|---|---|---|---|---|
| Original | — | 27.7 | 47.8 | 59.2 | 57.4 | **72.2** | 53.1 |
| SFT | 260k | 31.8 | 47.4 | 61.3 | 59.4 | 69.2 | 52.8 |
| Vanilla GRPO | 116k | 32.3 | 45.8 | 60.6 | 60.9 | 69.8 | 53.8 |
| **GRPO + ReAgent-V** | **52k** | **33.1** | **47.9** | **63.0** | **61.4** | 70.3 | 54.2 |

Notable: ReAgent-V's curated set is **45% the size of SFT's** and **45% the size of vanilla GRPO's**, yet wins on 5 of 6 benchmarks. Average improvement **+2.1%** (abstract).

**Table 3 / Figure 3 — VLA Alignment on SIMPLER.** OpenVLA-7B + ReAgent-V TPO vs. GRAPE (second-best):
- In-domain: 51.5 vs. 32.0 → +19.5
- Subject generalization: 34.7 vs. 27.0
- Physical generalization: 41.3 vs. 33.8 (approximate read)
- Semantic generalization: 47.0 vs. 34.5
- Average: 47.2 vs. 33.7 → **+13.5 absolute, +9.8% "overall" as reported in abstract**

**Table 4 — Reflection Ablation.** Removing reflection drops all four backbones on at least two of three benchmarks (LongBench / VideoMME / EgoSchema). Figure 4 further shows that each persona alone underperforms the full ensemble — aggressive in particular has the lowest "corrected-answer accuracy."

### 5.3 Baselines & Comparisons

**VLM baselines:** GPT-4o, Gemini-1.5-Pro (proprietary); LLaVA-Video-7B/72B, Qwen2.5-VL-7B/72B, Qwen2-VL-7B, InternVL-2.5-8B, VideoChat2-7B, LLaVA-NeXT-Video-7B, ShareGPT4Video-8B, Kangaroo, LLaVA-Video-7B, Long-LLaVA-7B, BIMBA-LLaVA (LLaMA3.2-8B and Qwen2-7B variants).

**Agent-framework baselines:** VideoAgent (general video agent), VideoMemAgent (memory-augmented video agent). Both appear in Table 1 and Table 3; ReAgent-V beats both while using fewer frames.

**Preference-optimization baselines (VLA):** OpenVLA-SFT, OpenVLA-DPO, OpenVLA-TPO with GRAPE (template reward).

### 5.4 Failure Analysis mapped to our taxonomy

The paper does not provide a per-error-type breakdown, but the ablations and case study let us infer:

| Error type | How ReAgent-V addresses it |
|---|---|
| **Perception errors** (VLM misidentifies objects) | Partially — neutral persona explicitly rewrites entities based on updated tool context (Figure 5 case study shows neutral agent correcting "White Castle building"). |
| **Egocentric-allocentric transformation errors** | **Not addressed.** No component does frame-of-reference reasoning. |
| **Relational reasoning errors** | Partially — scene graph tool provides explicit spatial predicates as text, but reasoning over them is still VLM-internal. |
| **Integration errors across frames** | ECRS selects higher-quality frames; reflection re-runs tools on sub-questions, but the total frame budget (32–64) caps long-video integration. |

The case study in Figure 5 — the "White Castle / 1920" burger question — is symbol-grounded and temporally dependent, and shows the reflection loop successfully switching the answer from B (literal interpretation) to C (historical/contextual). None of the published case studies are spatial.

---

## 6. Connections & Gaps

### 6.1 Related work ReAgent-V cites that we should read

| Paper | Why relevant |
|---|---|
| Video-R1 (Feng et al. 2025, arXiv:2503.21776) | The 260k video reasoning dataset ReAgent-V curates from — relevant if we want our own curated VSI-Bench training set. |
| Long-LLaVA-7B / LLaVA-Video (Zhang et al. 2024) | Strong open backbones for video QA; candidates for our own agent wrapper experiments. |
| VideoAgent (ECCV 2024) | Closest prior agentic framework — worth comparing pipeline shape. |
| VideoMemAgent (ECCV 2024) | Memory-augmented video agent, already in our literature folder (`literature/long-video-agent/`). |
| GRAPE (Zhou et al. 2025) | VLA preference-optimization baseline with template rewards; relevant if we move toward embodied tasks. |

### 6.2 What this paper does NOT address

Acknowledged by authors: static reasoning limitations in prior LVLMs; label costs of SFT; offline-reward models' inability to capture real-time reasoning state.

Gaps we identify:
- **No spatial primitives.** Treats all spatial information as semantic tokens; no ego-allo transform, no metric reasoning, no cognitive maps.
- **Fixed-length reflection.** Every sample pays the 6+ VLM-call cost whether or not reflection is needed; no early-exit.
- **Binary tool routing.** Tools are either called or not; no confidence-weighted combination or per-tool fallback.
- **Hand-tuned reflection thresholds (0.6, or 0.6/0.7/0.8 in code) are not calibrated.** The confidence scores are raw LVLM outputs; there is no calibration analysis.
- **Single-backbone critic.** The critic is the same VLM as the target agent; systematic shared biases cannot be corrected.
- **Importance score definition is informal.** "< 5/10" is used as a threshold but the mapping from the five dimensions to a single score is not specified in the paper.

### 6.3 How this paper relates to other papers we've read

- **vs. `literature/thinking-in-space/`:** Opposite ends of the same problem. Thinking-in-Space identifies spatial reasoning as *the* bottleneck in MLLMs (via VSI-Bench) and shows that CoT/ToT actively hurt on spatial tasks, suggesting prompt-based reasoning layers are insufficient. ReAgent-V builds exactly such a prompt-based reasoning layer and demonstrates gains on VSI-Bench — but only through *data curation for training*, not inference-time spatial reasoning. Together they reinforce our thesis: a generic reflection/reward agent is not enough; the agent has to do geometry externally.
- **vs. `literature/spatialscore/`:** SpatialScore + SpatialAgent is the closest precedent to our target: explicit spatial tools (depth, 3D bbox, camera pose) wired into an agent loop. ReAgent-V's tool factory pattern plus SpatialScore's tool set would be a natural combination.
- **vs. `literature/lvagent/` and `literature/long-video-agent/`:** Share the multi-agent video framework setup. ReAgent-V's novelty is the reward-driven reflection, not the decomposition itself.
- **vs. `literature/videothinker/`, `literature/vadar/`, `literature/riemind/`:** All share the inference-time refinement theme; ReAgent-V is distinguished by its unified critic rubric and the reward-as-curation closed loop.

---

## 7. Concrete Ideas Sparked

Actionable for our agent design and experiments:

1. **Spatial critic rubric.** Specialize ReAgent-V's five-dimension critic into a spatial-specific rubric: (a) ego-allo frame consistency, (b) metric plausibility of distances/sizes, (c) object-identity continuity across frames, (d) cognitive-map completeness, (e) chain-of-reasoning validity against the map. Plug this into the ReAgent-V scaffold verbatim.

2. **Spatial reflection personas.** Rework the three personas as perception-corrector / frame-corrector / geometry-corrector, each with a dedicated prompt template that references the critic's spatial rubric. Keeps the ensemble idea but re-anchors it to failure modes that actually matter on VSI-Bench.

3. **Spatial tool factory.** Port the tool-selection pattern (prompt-based binary routing) but replace the tool set with depth estimation, camera-pose estimation, 3D bounding box detection, metric scale probe, and a structured scene-graph-with-geometry tool. Use the same `modal_info` injection pattern.

4. **`importance_score`-based early exit.** Run only the initial-answer stage on samples where the critic's overall score > threshold. Measure inference-time savings vs. accuracy loss on VSI-Bench.

5. **Reward trace → VSI-Bench hard-sample miner.** Apply ReAgent-V's data curation exactly as published to VSI-Bench's training split (or a scraped Video-R1 subset filtered to spatial questions), and use the bottom-scoring ~5% as a "hardest VSI-Bench" diagnostic set for our own agent.

6. **Replication check on VSI-Bench.** Run ReAgent-V (not just its curated training set) as an end-to-end inference wrapper on VSI-Bench with a Qwen2.5-VL-7B backbone and report per-task accuracy. The paper does not evaluate the framework in inference mode on VSI-Bench; if there is no gain, that is direct evidence that reward-driven reflection does not solve spatial reasoning and an external geometric layer is necessary.

---

## 8. Quick Reference

| | |
|---|---|
| **Problem type** | Method / framework (training-free video-QA agent with reward-driven reflection) |
| **Domain** | General video understanding + video reasoning + VLA alignment |
| **Models involved** | LLaVA-Video-7B/72B, Qwen2.5-VL-7B/72B, Qwen2-VL-7B, OpenVLA-7B; baselines GPT-4o, Gemini-1.5-Pro, VideoAgent, VideoMemAgent, GRAPE |
| **Data** | LongBench, NextQA, EgoSchema, LVBench, MLVU, VideoMME; VSI-Bench, VideoMMMU, MMVU, MVBench, TempCompass; Video-R1-260k (curation source); SIMPLER (VLA) |
| **Open-source code?** | Yes — [github.com/aiming-lab/ReAgent-V](https://github.com/aiming-lab/ReAgent-V) |
| **Spatial repr. used** | Scene-graph text triples, bounding-box tags in prompt — no geometry |
| **Agent/decomposition?** | Yes — three-stage pipeline (ECRS → tool-augmented reasoning → critic + multi-persona reflection) |
| **Key number** | +6.9% video understanding / +2.1% video reasoning / +9.8% VLA on SIMPLER; VSI-Bench 27.7 → 33.1 via data curation on Qwen2.5-VL-7B |
