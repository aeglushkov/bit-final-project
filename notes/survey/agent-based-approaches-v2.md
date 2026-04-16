# Agent-Based Approaches for Video Understanding (v2)

## Navigation

- [VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](#videoseek-long-horizon-video-agent-with-tool-guided-seeking)
- [GCA: Geometrically-Constrained Agent for Spatial Reasoning](#gca-geometrically-constrained-agent-for-spatial-reasoning)
- [RieMind: Geometry-Grounded Spatial Agent](#riemind-geometry-grounded-spatial-agent)
- [VADAR: Visual Agentic AI with a Dynamic API](#vadar-visual-agentic-ai-with-a-dynamic-api)
- [VideoThinker: Agentic VideoLLMs with Tool Reasoning](#videothinker-agentic-videollms-with-tool-reasoning)
- [AoTD: Agent-of-Thoughts Distillation](#aotd-agent-of-thoughts-distillation)
- [LVAgent: Multi-Round Dynamical Collaboration of MLLM Agents](#lvagent-multi-round-dynamical-collaboration-of-mllm-agents)
- [ReAgent-V: Reward-Driven Multi-Agent Framework](#reagent-v-reward-driven-multi-agent-framework)
- [LongVideoAgent: Multi-Agent Reasoning with Long Videos](#longvideoagent-multi-agent-reasoning-with-long-videos)
- [Thinking in Space: How MLLMs See, Remember and Recall Spaces](#thinking-in-space-how-mllms-see-remember-and-recall-spaces)
- [SpatialScore: Comprehensive Evaluation for Spatial Intelligence](#spatialscore-comprehensive-evaluation-for-spatial-intelligence)
- [TRACE: Textual Representation of Allocentric Context from Egocentric video](#trace-textual-representation-of-allocentric-context-from-egocentric-video)

---

## Agent-Based Papers

### VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking

`Arxiv 2026` · 🏛️ AMD · 🏛️ University of Rochester

[📄 Paper](https://arxiv.org/abs/2603.20185) · [💻 Code](https://github.com/jylins/videoseek)

🏷️ **SUBJECT:** ReAct-style long-video QA agent that actively seeks evidence instead of densely parsing frames.

❓ **PROBLEM:**

- Existing video agents densely parse at 0.2–2 FPS, building expensive text indexes that scale badly with video length.
- 80% of LVBench questions can be answered from <5% of frames — exhaustive parsing is wasted compute.
- No principled way for a model to decide *where to look next*.

💡 **IDEA:** Give a reasoning LLM a **three-granularity seeking toolkit** (overview / skim / focus) and let it run a think–act–observe loop that navigates the video's logic flow instead of consuming every frame.

🛠️ **SOLUTION:**

- **`<overview>`:** uniformly sample 16α frames across the whole video (2×4 grids) to recover the global storyline.
- **`<skim>`:** coarse scan of a candidate segment (>4α s) with 4α uniform samples at low reasoning effort.
- **`<focus>`:** dense ~1 FPS inspection of a short clip (≤4α s) with full-resolution frames.
- **Think–act–observe loop:** GPT-5 plans the next tool call over accumulated observations and exits via `<answer>` (max 20 turns).

🏆 **RESULTS:** SOTA on long-video benchmarks while using **76–96% fewer frames** — LVBench 68.4%, VideoMME-long 70.1%, LongVideoBench 73.5% (29.6 frames vs. 384). Swapping GPT-5 for GPT-4.1 drops accuracy to 53%, confirming reasoning strength gates tool use.

---

### GCA: Geometrically-Constrained Agent for Spatial Reasoning

`Arxiv 2025` · 🏛️ Beihang · 🏛️ Shanghai AI Lab · 🏛️ SJTU · 🏛️ ZJU

[📄 Paper](https://arxiv.org/abs/2511.22659) · [💻 Code](https://github.com/Zx55) · [🚀 Project](https://gca-spatial-reasoning.github.io/)

🏷️ **SUBJECT:** Agent framework that constrains VLM spatial reasoning with an explicit task formalization before tool use.

❓ **PROBLEM:**

- VLMs reason in a lossy semantic space misaligned with high-fidelity geometry.
- Training-based methods inherit flawed spatial logic from imperfect GPT-4o-generated data.
- Tool-integrated baselines (SpatialAgent, TIGeR) route only the final computation through tools — the VLM still hallucinates *what* to solve.

💡 **IDEA:** Force the VLM to first emit an explicit **formal task constraint** `C_task = (C_R, C_O)` — a reference frame and an objective — then solve it as a ReAct agent strictly governed by that constraint.

🛠️ **SOLUTION:**

- **Task Formalization:** VLM as semantic analyst generates `C_R` (object/camera/direction-based 3D frame) and `C_O` (what to measure) in parallel.
- **Constrained Computation:** VLM as task solver issues tool calls governed by `C_task` with closed-loop ambiguity resolution.
- **8-tool toolbox:** VGGT reconstruction, open-vocab detection, 6-DoF pose (Orient Anything), metric-scale estimation, OCR, optical flow, Python sandbox.
- **Knowledge-Augmented Code Generation (KACG):** inject verified geometric formulas via RAG so the VLM composes code instead of hallucinating math.

🏆 **RESULTS:** **64.8% average across 5 spatial benchmarks (new SOTA)** — +12% over Gemini-2.5-Pro, +27% over SpatialLadder, +38% over TIGeR; generalizes across GPT-4o, GLM-4.5V, Qwen3-VL, Gemini (+37% avg when the framework is applied).

---

### RieMind: Geometry-Grounded Spatial Agent

`Arxiv 2026` · 🏛️ Riemann Lab, Huawei

[📄 Paper](https://arxiv.org/abs/2603.15386)

🏷️ **SUBJECT:** LLM agent for 3D indoor scene understanding via a geometry-grounded tool interface over a 3D scene graph.

❓ **PROBLEM:**

- End-to-end VLMs conflate perception and reasoning and fail on compositional 3D spatial queries.
- Fine-tuning on spatial QA just memorizes patterns instead of fixing the reasoning chain.
- No clean way to measure whether reasoning or perception is the real bottleneck.

💡 **IDEA:** Decouple perception from reasoning entirely — materialize the scene as a **3D scene graph (3DSG)** and let an LLM answer queries purely through geometry tools over node IDs, never seeing pixels.

🛠️ **SOLUTION:**

- **Hierarchical 3DSG:** Building → Floor → Room → Object nodes with ground-truth bboxes, volumes, orientations (built from ScanNet/ScanNet++/ARKitScenes).
- **MCP tool surface (4 namespaces):** `mem_*` scene context, `sg_*` graph traversal, `geom_*` volume/area/distance primitives, `loc_*` position and frame construction.
- **Node-ID grounding:** every tool takes node IDs — deterministic outputs, no free-text ambiguity.
- **Constrained prompt:** 7-section system prompt enforcing "search → resolve → tool-call" flow and delegating all computation to tools.

🏆 **RESULTS:** On VSI-Bench static, **RieMind + GPT-4.1 hits 89.5% avg vs. 73.6% for the best fine-tuned model** (SpaceMind) — a +16 pt gain; base VLMs improve by 33–50 pts when wrapped in the agent. Caveat: the whole framework assumes a ground-truth 3DSG.

---

### VADAR: Visual Agentic AI with a Dynamic API

`Arxiv 2025` · 🏛️ Caltech

[📄 Paper](https://arxiv.org/abs/2502.06787) · [🚀 Project](https://glab-caltech.github.io/vadar/)

🏷️ **SUBJECT:** Training-free agentic program synthesis for 3D visual-spatial reasoning.

❓ **PROBLEM:**

- Prior visual program synthesis (ViperGPT, VisProg) relies on a fixed, human-authored DSL that caps reachable reasoning.
- End-to-end VLMs struggle on compositional 3D queries (distance, size, relative position).
- Training a neuro-symbolic spatial reasoner requires large supervision (LEFT needs 10K+ samples).

💡 **IDEA:** Let LLM agents **dynamically grow a Pythonic API** — proposing, implementing, and testing new reusable spatial functions on the fly — then synthesize programs against that evolving API.

🛠️ **SOLUTION:**

- **API Generation:** a Signature Agent proposes reusable method signatures; an Implementation Agent writes each via DFS dependency resolution; a Test Agent validates and retries (up to 5×).
- **Program Synthesis:** a Program Agent writes a CoT-planned Python program against the grown API; an Execution Agent runs it line-by-line and feeds errors back.
- **Vision specialists:** Molmo + GroundingDINO, SAM2, UniDepth, GPT-4o VQA seed the API (`loc`, `depth`, `vqa`, `same_object`, `get_2D_object_size`).
- **Prompting scaffolds:** weak-ICL usage hints + pseudo-ICL implementation tips (each contributes independently in ablation).

🏆 **RESULTS:** On Omni3D-Bench, within ~2 pts of GPT-4o and ahead of all other VLMs; on CLEVR 53.6% vs. ViperGPT 42.6%. **Oracle-vision runs reach 83% / 94%** — the remaining gap is perception, not program logic. Matches or beats LEFT with **zero training data**.

---

### VideoThinker: Agentic VideoLLMs with Tool Reasoning

`Arxiv 2026` · 🏛️ ZJU · 🏛️ Fudan · 🏛️ Wuhan · 🏛️ Shanghai AI Lab

[📄 Paper](https://arxiv.org/abs/2601.15724)

🏷️ **SUBJECT:** Training an agentic VideoLLM that natively interleaves tool use with video perception.

❓ **PROBLEM:**

- Uniform frame sampling loses information and breaks temporal localization on long videos.
- Existing video agents reduce the VideoLLM to a passive captioner — only the LLM plans, and it cannot actually see frames.
- Training multi-step video tool use requires a VideoLLM that already understands long video — a chicken-and-egg problem.

💡 **IDEA:** Synthesize tool-use trajectories in **caption space** with a strong agentic LLM, then **swap captions for real frames** at training time — bypassing the chicken-and-egg problem entirely.

🛠️ **SOLUTION:**

- **6-tool toolkit:** temporal retrieval (ClipRetrieval, SubtitleRetrieval, SubtitleSummary) + temporal zoom (FrameZoom, SubtitleZoom, CaptionZoom — the bridge tool used during synthesis).
- **Trajectory synthesis:** Qwen3-235B-MoE reasons over CG-Bench captions; 5 trajectories per sample are filtered to those matching the gold answer.
- **Caption → video substitution:** CaptionZoom outputs replaced with `<video>` tokens over actual frames, producing video-interleaved CoTs.
- **LoRA fine-tuning:** Qwen2.5-VL-7B trained on 10K interleaved samples, ViT frozen, max-seq-len 200K.

🏆 **RESULTS:** Qwen2.5-VL-7B + VideoThinker **matches GPT-4o on MLVU (54.8) and LVBench (48.9)** and beats the 72B base on LVBench (+1.5 pts); gains grow with video length (+3.7 on VideoMME long) and dominate all agentic LLM baselines (VideoAgent, VideoTree, VideoExplorer).

---

### AoTD: Agent-of-Thoughts Distillation

`CVPR 2025` · 🏛️ SJTU · 🏛️ Coop. Medianet Innovation Center

[📄 Paper](https://arxiv.org/abs/2412.01694) · [💻 Code](https://github.com/zhengrongz/AoTD) · [🚀 Project](https://zhengrongz.github.io/AoTD/)

🏷️ **SUBJECT:** Instruction-tuning Video-LLMs with distilled agent reasoning traces for VideoQA.

❓ **PROBLEM:**

- Video-LLMs trained on raw (Q, A) pairs lack explainability and struggle with spatial-temporal grounding.
- Agent-based VideoQA systems offer interpretable reasoning but are too slow and memory-heavy for practical use.
- No established way to transfer the reasoning skill of an agent pipeline back into a single fast model.

💡 **IDEA:** Distill multi-step reasoning traces from a slow **agent-of-thoughts** pipeline (specialist vision tools orchestrated by an LLM) into a single Video-LLM via Chain-of-Thought instruction tuning.

🛠️ **SOLUTION:**

- **Specialist Selection:** benchmark off-the-shelf models on atomic sub-tasks (detection, temporal grounding, QA) and pick the best per slot.
- **Program Generation:** DeepSeek-Coder decomposes each question into a Python program that invokes specialists and records an execution trace.
- **CoT Conversion & Filtering:** LLaMA-3.1-8B rewrites traces into natural-language CoTs; traces with wrong answers or incoherent reasoning are dropped.
- **Distillation:** fine-tune LLaVA-NeXT-Video 7B with joint loss `L = L_label + λ·L_rationale`.

🏆 **RESULTS:** AoTD-tuned LLaVA-NeXT-Video beats baselines on STAR (74.3), NExT-QA (77.6), MVBench (55.6), and VSI-Bench (28.8) while running ~5× faster and using ~3.5× less memory than the agent pipeline it was distilled from.

---

### LVAgent: Multi-Round Dynamical Collaboration of MLLM Agents

`Arxiv 2024` · 🏛️ SIAT (CAS) · 🏛️ Tsinghua AIR · 🏛️ Shanghai AI Lab · 🏛️ SJTU

[📄 Paper](https://arxiv.org/abs/2503.10200) · [💻 Code](https://github.com/64327069/LVAgent)

🏷️ **SUBJECT:** Multi-agent MLLM collaboration framework for long-video question answering.

❓ **PROBLEM:**

- Feeding many frames into a single MLLM is expensive and drowns the model in redundant content.
- Single-MLLM agent pipelines inherit that one model's blind spots.
- Off-the-shelf CLIP retrieval has a domain gap on long videos, so relevant chunks are missed.

💡 **IDEA:** Replace the single MLLM with a **dynamically-curated team** of MLLM agents that iteratively perceive, vote, and expel the weakest member across multiple rounds of discussion.

🛠️ **SOLUTION:**

- **Selection:** rank MLLMs in an Agent Library via pseudo-label voting on 150 samples; top-3 form the team.
- **Perception:** 3-stage retrieval — random peek → 50-word "key info" generation → ASP-CLIP scores 6 chunks and keeps those above 0.8.
- **Action:** each agent answers independently; if >50% agree, early-stop.
- **Reflection:** agents score each other's reasoning, expel the lowest, summarize history, loop back (up to 3 rounds).

🏆 **RESULTS:** First agent method to exceed **80% on all four long-video benchmarks** — EgoSchema 82.9, LongVideoBench 80.0 (+13.3 over GPT-4o), MLVU 83.9, VideoMME 81.7/86.6 — beating 72B-scale non-agent baselines.

> *Orchestration is training-free; only the ASP-CLIP retriever is fine-tuned (on LongVR). The team-of-MLLMs logic uses frozen off-the-shelf models.*
> 

---

### ReAgent-V: Reward-Driven Multi-Agent Framework

`NeurIPS 2025` · 🏛️ UNC-Chapel Hill · 🏛️ University of Washington

[📄 Paper](https://arxiv.org/abs/2506.01300) · [💻 Code](https://github.com/aiming-lab/ReAgent-V)

🏷️ **SUBJECT:** Agentic video-QA framework that couples entropy-guided frame selection, tool-augmented inference, and multi-perspective reflective refinement driven by inference-time reward signals.

❓ **PROBLEM:**

- Single-pass LVLM video QA has no mechanism to self-correct or integrate dynamic feedback.
- Offline reward models / template rewards cannot capture real-time reasoning state during inference.
- Prior multi-agent / tool-agent frameworks are slow, lack reward signals, and overprocess frames.

💡 **IDEA:** Bolt a reward-generating critic and a **multi-perspective reflection** loop (conservative / neutral / aggressive) onto tool-augmented VLM inference, so the same reward trace both refines the current answer and curates high-value samples for SFT / DPO / GRPO.

🛠️ **SOLUTION:**

- **Entropy-Calibrated Relevance Scoring (ECRS):** jointly scores frames by CLIP query-similarity × per-channel RGB histogram entropy; iterative threshold keeps only relevant and information-rich frames.
- **Tool-augmented reasoning:** target agent picks from a tool factory (OCR, ASR, Grounding-DINO, scene graph, CLIP, caption models) per query.
- **Critic agent:** emits an evaluation report with scalar reward + five scored dimensions (visual alignment, temporal accuracy, option disambiguation, reasoning specificity, linguistic precision) and re-invokes tools on unsatisfactory answers.
- **Multi-perspective reflection:** three persona prompts (conservative / neutral / aggressive) regenerate answers with confidence scores; a meta-agent fuses them when `min` confidence > 0.6.
- **Reward-aware data curation:** the same critic report drives SFT filtering, DPO preference pairs, and GRPO sample selection.

🏆 **RESULTS:** **+6.9%** on video understanding (LLaVA-Video-72B / Qwen2.5-VL-72B across LongBench, NextQA, EgoSchema, LVBench, MLVU, VideoMME), **+2.1%** on video reasoning via GRPO data curation (52k vs 260k samples), and **+9.8%** on VLA alignment (OpenVLA+TPO on SIMPLER) over GRAPE.

> *VSI-Bench is evaluated (27.7 → 33.1) but the framework has no spatial primitives — the gain comes from data curation, not a spatial reasoning mechanism.*
> 

---

### LongVideoAgent: Multi-Agent Reasoning with Long Videos

`ACL 2026 Main` · 🏛️ HKUST

[📄 Paper](https://arxiv.org/abs/2512.20618) · [💻 Code](https://github.com/longvideoagent) · [📊 LongTVQA](https://huggingface.co/datasets/longvideoagent/LongTVQA) · [🚀 Project](https://longvideoagent.github.io/)

🏷️ **SUBJECT:** Tool-augmented multi-agent framework for hour-scale video question answering, with the master policy trained by GRPO.

❓ **PROBLEM:**

- Single-pass MLLMs compress or heavily downsample long video into one context window, losing the fine-grained evidence needed for sparse-cue questions.
- Prior agentic systems (VideoAgent) rely on weak, generic perception tools — insufficient for subtle object, action, and OCR cues in TV-episode footage.
- Existing frameworks underuse the LLM's planning ability and lack an RL signal for learning *when* to invoke which tool.

💡 **IDEA:** A **multi-agent pipeline** in which a MasterAgent iteratively decides when to ground, when to look, and when to answer — coordinating a dedicated GroundingAgent and VisionAgent over a bounded action loop, and trained with GRPO on rule-based structural-plus-correctness rewards.

🛠️ **SOLUTION:**

- **MasterAgent:** policy LLM that emits exactly one structured action per turn — `<visual_query>`, `<request_grounding>`, or `<answer>` — for up to K rounds.
- **GroundingAgent:** returns a `<clip_X>` tag localizing question-relevant segments (default: Grok-4-fast-reasoning, frozen).
- **VisionAgent:** reads the grounded clip and returns textual facts about objects, actions, OCR, and scene cues (default: GPT-4o, frozen).
- **AgenticRL (GRPO):** per-step structural reward `r^fmt ∈ {0,1}` + terminal answer-correctness reward `r^ans ∈ [0,1]`; **only the master is fine-tuned**.
- **LongTVQA / LongTVQA+:** new episode-level benchmarks aggregating all TVQA / TVQA+ clips from the same TV episode into hour-scale sequences with preserved timestamps and bounding boxes.

🏆 **RESULTS:** AgenticRL-Qwen2.5-7B reaches **60.20 / 70.80** on LongTVQA / LongTVQA+ — +14.10 / +10.50 over the non-agentic 7B baseline and on par with closed-source GPT-5-mini; Agentic-Grok tops the leaderboard at **82.65 / 85.60**.

---

## Benchmarks

### Thinking in Space: How MLLMs See, Remember and Recall Spaces

`CVPR 2025 Oral` · 🏛️ NYU · 🏛️ Yale · 🏛️ Stanford

[📄 Paper](https://arxiv.org/abs/2412.14171) · [💻 Code](https://github.com/vision-x-nyu/thinking-in-space) · [📊 VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) · [🚀 Project](https://vision-x-nyu.github.io/thinking-in-space.github.io/)

🏷️ **SUBJECT:** Benchmark and analysis of visual-spatial intelligence in video MLLMs.

❓ **PROBLEM:**

- No high-fidelity benchmark measures 3D spatial understanding from egocentric indoor videos in MLLMs.
- Unclear whether MLLM failures come from perception, language, or spatial reasoning.
- Standard prompting tricks (CoT, ToT) are assumed to help but have never been tested on spatial tasks.

💡 **IDEA:** Build a carefully annotated benchmark from 3D reconstructions (**VSI-Bench**) and probe *where* MLLMs actually fail on visual-spatial intelligence — then test whether explicit **cognitive maps** can unblock them.

🛠️ **SOLUTION:**

- **VSI-Bench:** 5K+ QA pairs from 288 egocentric indoor videos (ScanNet, ScanNet++, ARKitScenes), 8 tasks across configurational and measurement categories.
- **Broad evaluation:** 15 open- and closed-source MLLMs benchmarked head-to-head against humans.
- **Linguistic analysis:** separate spatial-reasoning signal from general language capability.
- **Cognitive-map probing:** elicit spatial memory and test whether it improves downstream answers.

🏆 **RESULTS:** **Spatial reasoning — not linguistic capability — is the bottleneck**; CoT/ToT prompting actively hurts on spatial tasks; explicit cognitive-map generation improves distance estimation. 71% of errors come from egocentric-allocentric transformation failures, only 8% from perception.

---

### SpatialScore: Comprehensive Evaluation for Spatial Intelligence

`Arxiv 2025` · 🏛️ SJTU · 🏛️ Shanghai AI Lab

[📄 Paper](https://arxiv.org/abs/2505.17012) · [💻 Code](https://github.com/haoningwu3639/SpatialScore/) · [📊 Dataset](https://huggingface.co/datasets/haoningwu/SpatialScore) · [🚀 Project](https://haoningwu3639.github.io/SpatialScore/)

🏷️ **SUBJECT:** Holistic benchmark, training corpus, and agent system for evaluating and improving spatial intelligence in MLLMs.

❓ **PROBLEM:**

- Existing spatial benchmarks cover narrow slices (single modality, single task type) and under-report where MLLMs actually fail.
- No training corpus targets the specific sub-skills that spatial reasoning requires.
- No standard tool-augmented baseline exists to measure how far agent wrappers can close the gap without retraining.

💡 **IDEA:** Ship three artifacts together — a broad **SpatialScore** benchmark, a matching **SpatialCorpus** for SFT, and a plug-in **SpatialAgent** — so data-centric and agent-centric improvements can be compared head-to-head.

🛠️ **SOLUTION:**

- **SpatialScore:** 5K samples · 30 tasks · 10 categories spanning real-world, simulated, and AIGC imagery across single-image, multi-frame, and video inputs.
- **SpatialCorpus:** 331K supervised QA pairs for spatial SFT (perception, relations, measurement).
- **SpatialAgent:** multi-agent system with 12 perception tools driven by Plan-Execute and ReAct, zero training required.
- **Broad evaluation:** 40 representative MLLMs benchmarked to expose systematic failure modes.

🏆 **RESULTS:** Best MLLM still trails human performance (86.60) by **26.48 points**; SFT on SpatialCorpus gives Qwen3-VL-4B a **+10.47** gain, and SpatialAgent improves reasoning without any training.

---

## Other

### TRACE: Textual Representation of Allocentric Context from Egocentric video

`Arxiv 2026` · 🏛️ Tsinghua · 🏛️ Shanghai AI Lab · 🏛️ University of Tokyo

[📄 Paper](https://arxiv.org/abs/2603.23404)

🏷️ **SUBJECT:** Prompting method that builds an allocentric text scaffold for spatial QA in MLLMs.

❓ **PROBLEM:**

- MLLMs reason egocentrically from raw frames and lose the global 3D layout needed for spatial QA.
- Existing prompting baselines (CoT, ToT, LtM, Cognitive Map) give small or inconsistent gains.
- Fine-tuning or bolting on geometric modules is expensive and does not transfer across backbones.

💡 **IDEA:** Have the MLLM first generate a **Textual Representation of Allocentric Context from Egocentric video (TRACE)** — room topology, coordinate frame, camera trajectory, and an entity registry — and answer the question in the *same* forward pass.

🛠️ **SOLUTION:**

- **Meta-context block:** room topology + global coordinate system committed to before listing entities.
- **Camera trajectory log:** per-frame ego pose grounds observations allocentrically.
- **Entity registry:** object list with estimated 3D positions and pairwise spatial relations.
- **One-stage inference:** the scaffold and the answer are produced in a single pass — two-stage variants are worse (the generation *process* is the reasoning).

🏆 **RESULTS:** TRACE consistently beats CoT, ToT, LtM, and Cognitive Map prompting on **VSI-Bench** and **OST-Bench** across Gemini 3 Pro, Qwen2.5-VL-72B, and MiMo-VL-7B — gains of ~7.5% that hold across backbone scale.

---

## Results Comparison

### VSI-Bench (video spatial QA, 5K samples, human: 79.2%)

| Method | Type | Overall |
| --- | --- | --- |
| RieMind + GPT-4.1 | Agent + GT 3D scene graph | **89.5%** |
| RieMind + GPT-4o | Agent + GT 3D scene graph | 85.2% |
| RieMind + Qwen2.5-VL-7B | Agent + GT 3D scene graph | 64.1% |
| TRACE + Gemini 3 Pro | Prompting | 60.15% |
| VADAR | Agent (image-only subset) | 50.1% |
| Gemini-1.5 Pro | Base VLM | 49.1% |
| GPT-4o | Base VLM | 34.9% |
| AoTD (7B) | Distilled | 28.8% |

### SpatialScore (30 spatial tasks, 5K samples, human: 86.6%)

| Method | Type | Overall |
| --- | --- | --- |
| Gemini-3-Pro | Base VLM | 60.12% |
| GPT-5 | Base VLM | 58.13% |
| Qwen3-VL-8B + SpatialCorpus | Fine-tuned | 54.71% |
| Qwen3-VL-8B + SpatialAgent-ReAct | Agent (training-free) | 53.81% |
| Qwen3-VL-4B + SpatialCorpus | Fine-tuned | 52.99% |
| Qwen3-VL-4B + SpatialAgent-ReAct | Agent (training-free) | 50.30% |
| Qwen3-VL-8B | Base VLM | 45.48% |

**Takeaway:** Agent-based approaches (training-free) achieve +8 points over base VLMs, competitive with fine-tuning (+9–10 points).