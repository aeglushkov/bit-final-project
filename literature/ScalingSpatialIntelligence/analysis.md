# Paper Analysis: Scaling Spatial Intelligence with Multimodal Foundation Models

**Authors:** Zhongang Cai, Ruisi Wang, Chenyang Gu, … Ziwei Liu, Quan Wang, Dahua Lin, Lei Yang (SenseTime Research + NTU) | **Venue:** arXiv:2511.13719v4, 28 March 2026 (technical report) | **Date read:** 2026-05-14

**Links:** [Paper](https://arxiv.org/abs/2511.13719) | [Code](https://github.com/OpenSenseNova/SenseNova-SI) | [Models](https://huggingface.co/collections/sensenova/sensenova-si)

---

## 1. Core Contribution (2-3 sentences)

A large-scale, data-centric study showing that careful curation of 8M spatial QA samples — organized under the five-capability EASI taxonomy with deliberate emphasis on the long-overlooked **Perspective-Taking** axis — lifts three diverse open-source multimodal foundations (Qwen3-VL-8B, InternVL3-{2B,8B}, Bagel-7B-MoT) to state-of-the-art on every major spatial benchmark while preserving general multimodal competence. The single most important claim is that **scaling spatial data, not architectural innovation, is what currently moves the needle** — but the same study also documents *where that approach stops working*: text-based chain-of-thought yields no consistent gains, RL fails to help, and capability gains saturate well below human level.

---

## 2. Relevance to Our Research

### 2.1 Relevance Rating: **High**

The paper is a contemporaneous, well-resourced negative result for naive text-CoT scaling of spatial reasoning — precisely the gap our agent-based hypothesis is designed to fill. SenseNova-SI is also a strong open-source baseline we should benchmark against, and their InternVL3-8B variant is architecturally compatible with the bf16 stack we already serve in `experiments/eva-eval`.

### 2.2 Key Takeaways for Our Direction

- **Direct validation that text-CoT is not the right primitive for 3D reasoning.** Three engineered CoT recipes (GPT-5 free-form, MindCube CogMap JSON, and the authors' own procedural continuous-coordinate CGMap) deliver only ~2% over plain QA-SFT on VSI-Bench Rel.Dir, and the best variant *hurts* when wrapped in GRPO RL (Table 4: 49.2 → 43.1). Section 5.6 explicitly concludes "multimodal RL for spatial reasoning remains largely underexplored… may signal the need for a broader paradigm shift beyond conventional CoT" — this is essentially an open invitation for agent-style externalized reasoning.
- **Perspective-Taking is a meta-skill.** The capability-transfer matrix (Table 7) shows training on PT data alone yields +46.1% on Mental Reconstruction and +10.9% on Comprehensive Reasoning, while hurting only Spatial Relations by 0.2%. If we externalize *just* allocentric transformation we may unlock most of the downstream capability.
- **Scaling saturates fast.** Table 8: gains from 5M→8M samples are within 1–3 points on every benchmark; SITE actually peaks at 1M (47.7→44.5 from 0M→1M, then never returns). The data-only ceiling is in sight.
- **InternVL3-8B is the right base for our agent.** SenseNova-SI-InternVL3-8B is the top family-mate on PT (54.7 ViewSpatial, 85.7 MindCube) yet they could not push VSI beyond 68.8 — our agent should be measured *against* this not against vanilla InternVL3.
- **Their procedural CGMap is a near-miss of our idea.** It builds a continuous-coordinate top-down map step-by-step inside the LM (keyframe → relative camera estimate → global CogMap → answer). The failure mode (Fig. 8) — error accumulation across coordinate transforms — is exactly the case where externalizing the math to code would help.

### 2.3 Perception vs. Reasoning Evidence

Our core hypothesis: VLMs excel at perception but fail at spatial reasoning. The paper provides strong indirect evidence on both sides.

| Aspect | Evidence from paper |
|--------|-------------------|
| Perception capability | BLINK 63.9 for SenseNova-SI-InternVL3-8B, with near-SoTA on visual correspondence, semantic similarity, and counting subtasks — confirms VLMs can be made strong at single-frame perception with enough data. Object-recognition errors in the EmbodiedBench rollouts (Fig. 6 c/d) remain a failure mode even after 8M samples. |
| Reasoning capability | MMSI-Bench overall only 43.3 (vs. human 97.2); Comprehensive Reasoning subtasks like Route Plan on VSI hover at 48.5. Despite scaling, reasoning lags perception by 30–50 points. |
| Ego-allo transformation | Perspective-Taking subtasks are where SenseNova-SI dominates (e.g., MindCube 85.7 vs. GPT-5 56.3; VSI Rel.Dir 80.8 vs. GPT-5 48.7), but only after explicit, scaled-up curation of PT data — and even then ViewSpatial Sec.Sim is only 64.5. The transformation skill is *learnable from data* but does not emerge from general training. |
| Error attribution | Section 5.5: SenseNova-SI drops 85.6 → 52.5 on MindCube without vision, while the prior SoTA MindCube-RawQA-SFT drops only 51.7 → 50.7 — i.e. earlier "spatial intelligence" models were largely text-shortcut artifacts. The authors validate that their gains are truly visually grounded. |

---

## 3. Method

### 3.1 Approach Summary

Three commitments: (1) **architecture-preserving** continued training on three diverse foundations (Qwen3-VL-8B, InternVL3-2B/8B, Bagel-7B-MoT — the last being a unified understanding+generation model); (2) **principled data curation** under the five-capability EASI taxonomy; (3) **controlled empiricism** — they run scaling curves, capability-transfer matrices, text-only ablations, and circular tests rather than just reporting one headline number.

### 3.2 Architecture / Pipeline

No architectural change. Data pipeline (Appendix B):

1. **Unified annotation.** Standardize ScanNet, ScanNet++, SUN RGB-D, CA-1M, MessyTable, Ego-Exo4D, Matterport3D into a common schema with 3D camera poses, 3D object poses (boxes + orientation), 2D point/object visibility, and semantic descriptions.
2. **Object selection.** Drop floor/ceiling/wall; filter by minimum visible-area ratio and visibility ratio (visible-bbox / projected-bbox).
3. **Image selection.** Pose-filter degenerate yaw/pitch; build multi-view sets via Algorithm 1 (greedy frame addition keeping per-pair overlap ρ ∈ [0.03, 0.20]); pad uniformly to ≥16 frames.
4. **QA selection.** Ambiguity reduction (require unique instance of queried category, drop angularly ambiguous directions); balanced sampling across paraphrases and object combinations.
5. **CoT augmentation (optional).** GPT-5 step-wise annotation, MindCube grid CogMap, or the authors' procedural continuous-coordinate CGMap.

Training: one epoch · 128 GPUs · batch 2048 · AdamW · LR 5e-6 · max 16 frames per video sample · ~3 days per backbone.

### 3.3 Spatial Representation

| Representation type | Description | Strengths | Weaknesses |
|-------------------|-------------|-----------|------------|
| Natural language QA pairs | The primary signal — 8M parametric QAs across MM/SR/MR/PT/CR | Bypasses tokenizer / encoder mismatch; uniformly applicable across backbones | All geometry is implicit in weights; no inspectable intermediate state |
| Discretized 2D CogMap (MindCube-style baseline) | JSON dict mapping objects/cameras to a 10×10 grid + 4-way orientations | Compact, easy for LM to emit | Coarse; loses metric precision; no notion of camera intrinsics |
| Procedural continuous-coordinate CGMap (their CoT) | Per-keyframe metric coordinates of objects + relative camera rotations/translations, accumulated into a global frame, then rotated to the query's egocentric anchor | Most expressive text-based representation; can answer arbitrary allocentric queries | Errors accumulate across the per-frame chain (Fig. 8); the LM is doing all SE(3) math token-by-token — exactly the wrong place for it |
| Implicit 3D structure in the VLM | Encoder features (Qwen3-VL / InternVL3 / Bagel) | High-bandwidth visual evidence | Opaque; no external querying possible |

---

## 4. Agent / Decomposition Aspects

*This section is specific to our research interest in agent-based approaches.*

### 4.1 Does the paper use any form of task decomposition?

Only inside the CoT recipes:

- **VLM-generated CoT:** single GPT-5 pass that emits step-wise reasoning per (question, gold answer) pair — used as training labels, not at inference.
- **MindCube CogMap CoT:** two steps — (1) emit a discretized CogMap JSON, (2) free-form reasoning over it.
- **Procedural CGMap CoT (theirs):** four explicit steps — (1) Keyframe Localization, (2) Incremental Relative Camera Estimation, (3) CogMap Construction in a continuous-coordinate global frame, (4) Answer Derivation with optional egocentric rotation.

Critically, all decomposition is *inside the LM's text output*; there are no tool calls, no external solvers, and no feedback loops.

### 4.2 Multi-turn VLM interaction

None. Single forward pass per sample at both training and inference time. The 16-frame cap is enforced uniformly.

### 4.3 External reasoning modules

None at inference. The *training data construction* leverages 3D ground-truth (ScanNet point clouds, Matterport3D poses, Ego-Exo4D paired views) to label the questions, but no external geometry engine is invoked at evaluation.

The procedural CGMap is the closest the paper comes to externalizing geometry — but it externalizes only the *format* (continuous coordinates, allocentric rotation), not the *computation* (the LM still does the SE(3) accumulation in tokens, and Fig. 8 shows it drifting).

### 4.4 Implications for our agent design

1. **Adopt their data and taxonomy as the training prior.** SenseNova-SI-8M is publicly released; the EASI five-capability decomposition matches our error taxonomy. We should build an agent on top of an SI-trained VLM rather than vanilla InternVL3 — the perception baseline is much stronger.
2. **Externalize exactly what their CoT fails at: SE(3) accumulation across views.** Their procedural CGMap (Sec. B.5 / Appendix J Figs. 7–8) does what we'd want to do — build a global metric scene representation from a chain of relative camera estimates — but it does it in tokens and accumulates error. A clean win: ask the VLM only for "what objects in what relative direction at what distance per frame", then accumulate frames externally with a real SE(3) library.
3. **Treat Perspective-Taking as the primary tool boundary.** The capability-transfer matrix (Table 7) shows PT data lifts MR and CR; conversely, if PT is the bottleneck in the *base* model, externalizing PT should unlock MR/CR without further training.
4. **Avoid GRPO on long spatial CoT.** Table 4 shows GRPO on CGMap CoT *regresses* from 49.2 → 43.1. RL over noisy long-form spatial CoT is unlikely to work; if we want learning, it should be over tool-use trajectories, not over verbal CoT.
5. **Benchmark target.** Their 68.8 on VSI-Bench is the open-source ceiling for data-centric SFT on InternVL3-8B at 16 frames. That is the number our agent has to beat to be a real contribution.

---

## 5. Evaluation

### 5.1 Benchmarks & Metrics

| Benchmark | Tasks | Metrics | Relevant to us? |
|-----------|-------|---------|-----------------|
| VSI-Bench [64] | 8 video-based 3D-layout tasks (NA: Obj.Count, Abs.Dist, Obj.Size, Room.Size; MCA: Rel.Dist, Rel.Dir, Route.Plan, Appear.Order) | MRA + Acc | **Primary** — this is our main eval |
| MMSI-Bench [68] | Multi-image positional / attribute / motion / multi-step reasoning | Acc | Yes — multi-frame PT |
| MindCube [70] | Mental modeling from limited views | Acc | Yes — PT meta-task |
| ViewSpatial [31] | Multi-perspective ego/allo localization | Acc | Yes — direct allocentric eval |
| SITE [57] | 30+ datasets unified; abstract spatial reasoning | CAA | Useful for generalization claims |
| BLINK [20] | Low-level perception (depth, correspondence, multi-view) | Acc | Yes — to attribute perception vs. reasoning |
| 3DSR-Bench [39] | Natural-image 3D spatial reasoning with circular eval | Acc | Yes — debiased eval |
| EmbSpatial [16] | Embodied egocentric spatial understanding | Acc | Yes — downstream proxy |
| MMBench-En [35] et al. | General multimodal | Acc | For retention checks only |

### 5.2 Key Results

Headline (Table 1 row "SenseNova-SI InternVL3-8B"): VSI **68.8 (+26.7 vs base)**, MMSI **43.3 (+15.3)**, MindCube **85.7 (+44.2)**, ViewSpatial **54.7 (+16.0)**, SITE **47.7 (+6.6)**, BLINK **63.9 (+10.4)**, 3DSR **55.5 (+11.2)**, EmbSpatial **72.0 (-4.3)**.

Perception-vs-reasoning split:
- BLINK 63.9 with strong sub-scores on Counting (76.7), SpatR (95.9), SemCr (93.5), VisCr (86.0) — perception is solid.
- MMSI overall still only 43.3 vs human 97.2 — reasoning gap is enormous despite massive data scale.

Without-vision drop (Table 3 + Table 10): 85.6 → 52.5 on MindCube without images proves visual grounding; text-only training on the same 10% of corpus only lifts VSI 42.1 → 50.2 (vs. 60.9 with vision), confirming a real but not exclusive role for textual cues.

Frame extrapolation (Table 2): trained on ≤16 frames, generalizes to 32 (68.8) and 64 (62.8) at inference. Suggests an internal representation that survives temporal extension.

### 5.3 Baselines & Comparisons

Proprietary: GPT-5-2025-08-07, Gemini-2.5-Pro-2025-06, Gemini-3-Pro-Preview, Grok-4, Seed-1.6. SenseNova-SI beats Gemini-2.5-Pro on average, matches GPT-5, and is beaten only by Gemini-3-Pro-Preview on the aggregate.

Open-source SI: MindCube-RawQA-SFT, SpatialLadder-3B, Spatial-MLLM-4B, SpaceR-7B, ViLaSR-7B, VST-3B/7B-SFT, Cambrian-S-3B/7B. SenseNova-SI tops all of them across the spatial suite.

**No agent-based baseline is reported** — this is the gap our line of work occupies. SpatialAgent (from SpatialScore) is not compared against.

### 5.4 Failure Analysis

EmbodiedBench rollouts (Fig. 6 c, d) and the spatial-CoT case studies (Fig. 7 vs Fig. 8) reveal:

- **Perception errors** — visible in Fig. 6c: model misidentifies "red cylinder" vs "maroon cylinder" → wrong action plan. Still occurs after 8M samples.
- **Egocentric-allocentric transformation errors** — Fig. 8 (the failure CoT): per-frame camera-rotation estimates drift (e.g. right-50° at one step compounds with left-100° at next), final transferred coordinates put the refrigerator at the wrong quadrant. **This is exactly the failure mode our agent is designed to remove.**
- **Relational reasoning errors** — VSI Route.Plan stays at 48.5 even after training; multi-step planning is the weakest VSI subtask.
- **Integration errors** — MMSI Multi-Step Reasoning (MSR) only 27.8 for SenseNova-SI-InternVL3-8B; combining cues across many frames is fragile.

The authors' own diagnosis of CoT failure (Appendix J): "early local errors in the reasoning may accumulate along the sequence and become increasingly difficult to correct; tasks that require a globally coherent 3D structure may be particularly sensitive to this discrepancy." This is a direct argument for externalizing geometry.

---

## 6. Connections & Gaps

### 6.1 Related work this paper cites that we should read

| Paper | Why relevant |
|-------|-------------|
| EASI [Cai et al. 2025, arXiv:2508.13142] | The taxonomy SenseNova-SI is built on — "Has GPT-5 achieved spatial intelligence? An empirical study". Likely defines the eval protocol used throughout. |
| Brown et al. 2025 [5] "Benchmark designers should train on the test set to expose exploitable non-visual shortcuts" | The VSI-Debiased protocol used in Table 2; we should adopt this for any claim we make. |
| MindCube [Yin et al. 2025] | The CogMap representation we should compare our scene-graph against. |
| Cambrian-S [Xu et al. 2025] | The other strong video-spatial scaling competitor; 64/128-frame training compared to SenseNova-SI's 16-frame training. |
| SpatialReasoner [40] | Authors flag it as related work on explicit 3D spatial reasoning — directly aligned with externalized geometry. |
| EmbodiedBench [65] | The downstream eval they used; useful target for our agent's downstream claim. |

### 6.2 What this paper does NOT address (gaps we could fill)

- **No external tool use at inference.** They externalize *labels* via 3D datasets but never call a solver, depth model, or pose estimator at test time.
- **No multi-turn or iterative VLM interaction.** Single forward pass; no self-correction over their own CoT.
- **No agentic baseline reported.** SpatialAgent / VAdar / SpatialRGPT are absent from the comparison.
- **Saturating returns from data — they explicitly say data scaling alone is insufficient.** The paper ends with a call for "fundamentally different reasoning mechanisms."
- **Per-frame geometric output is not exposed.** Our agent could leverage their model as a strong per-frame perception primitive and supply the geometry externally.

### 6.3 How this paper relates to others we've read

- **vs. `literature/thinking-in-space`:** Yang et al. argued cognitive maps help; SenseNova-SI tries it (MindCube CogMap and their procedural CGMap) and finds only marginal gains over QA-SFT. The conclusion is nuanced — cognitive maps as a *representation* are fine, but expecting the LM to *construct* them in tokens is brittle.
- **vs. `literature/spatialscore`:** SpatialScore proposed SpatialAgent as the tool-augmented complement to SpatialCorpus SFT. SenseNova-SI confirms the SFT side of that story (data does scale) but does *not* test the agent side — leaving SpatialAgent's claim un-replicated and our agent direction unchallenged.
- **vs. `literature/videothinker`:** VideoThinker shows that *training* a VLM to interleave tool calls works for long-video temporal retrieval. SenseNova-SI is the natural baseline for what one can do *without* tools on the spatial axis — and shows it isn't enough.
- **vs. our `notes/idea-agent-architecture`:** strong external validation. Their procedural CGMap is essentially "what would happen if we tried to do our agent purely in text" — and it underperforms. The case for moving the geometry out of the LM is stronger after this paper.

---

## 7. Concrete Ideas Sparked

1. **Use SenseNova-SI-InternVL3-8B (HuggingFace) as the perception backbone for our agent.** Replaces vanilla InternVL3 in `experiments/eva-eval` — would also need a parallel bf16 server entry alongside the current `internvl2-8b-bf16` setup. Likely a big lift to the perception ceiling at no architectural cost.
2. **Reproduce their procedural CGMap failure with our agent doing the SE(3) externally.** Sample VSI Rel.Dir items where their model fails (Fig. 8 style); replace step 3 ("Construct Scene Layout") with a programmatic accumulation using actual camera-pose deltas estimated by VGGT or a depth+pose tool. Direct head-to-head on the same questions.
3. **Adopt the EASI capability split for our error taxonomy.** Map our perception/transformation/relational/integration buckets onto MM/SR/PT/MR/CR so results are comparable to this paper and the EASI benchmarking line.
4. **Run the VSI-Debiased + circular-eval protocols.** Table 3 / Table 10 show how dramatic the shortcut effect is in prior SoTA. We should adopt both to defend any agent gains we claim.
5. **Use EmbodiedBench Spatial subset as our downstream proxy.** Table 5 numbers are the most concrete embodied benefit shown — beating SenseNova-SI's 33.3 SIP would be a strong claim.
6. **Consider a hybrid: SFT the agent's "tool selection / output formatting" head on a subset of SenseNova-SI-8M's PT data.** Avoids learning perception from scratch and gives the agent a clean PT prior.

---

## 8. Quick Reference

| | |
|---|---|
| **Problem type** | method (data-centric scaling study) + analysis |
| **Domain** | general 3D spatial reasoning, indoor scenes, multi-view |
| **Models involved** | Qwen3-VL-8B, InternVL3-2B/8B, Bagel-7B-MoT (continued SFT only) |
| **Data** | SenseNova-SI-8M — 3.3M open-source + 4.5M newly synthesized from ScanNet/ScanNet++/SUN RGB-D/CA-1M/Ego-Exo4D/MessyTable/Matterport3D |
| **Open-source code?** | Yes — https://github.com/OpenSenseNova/SenseNova-SI; models on HuggingFace |
| **Spatial repr. used** | Implicit (in weights) + ablations with discretized CogMap and procedural continuous-coordinate CGMap |
| **Agent/decomposition?** | No external tool use; CoT-style internal decomposition tested and found marginal |
| **Key number** | VSI-Bench 68.8 / MindCube 85.7 / ViewSpatial 54.7 on InternVL3-8B base (open-source SoTA, surpasses GPT-5 on PT-heavy tasks) |
