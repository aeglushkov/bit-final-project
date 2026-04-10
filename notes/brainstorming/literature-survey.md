# Literature Survey: Agent-Based Approaches for Spatial Video Understanding

**Date:** 2026-04-08
**Scope:** 10 papers stored in `literature/` — the primary reading list for this research project.
Papers 1–3 recommended by advisor Diwei Su (confirmed relevant). Papers 4–10 self-found (relevance assessed in §5).
For the broader landscape of agent frameworks beyond these 10 papers, see [agent-landscape-survey.md](agent-landscape-survey.md).

### Diagrams (FigJam)
- [Timeline: Field Evolution Dec 2024 → Mar 2026](https://www.figma.com/online-whiteboard/create-diagram/a29dfdeb-6732-41e9-81bb-871d87b086b8?utm_source=claude&utm_content=edit_in_figjam)
- [Externalization Spectrum: How Much Reasoning is Outside the VLM?](https://www.figma.com/online-whiteboard/create-diagram/21cc84c4-c84d-416d-bc2f-642a940c0845?utm_source=claude&utm_content=edit_in_figjam)
- [Research Gap: Where Our Work Fits](https://www.figma.com/online-whiteboard/create-diagram/1f0a6a1a-ae98-4d2e-a4a1-d7642a4ef2ea?utm_source=claude&utm_content=edit_in_figjam)

---

## 1. Chronological Overview

| # | Paper | Authors | Affiliation | Venue | Date | Contribution |
|---|-------|---------|-------------|-------|------|-------------|
| 1 | **Thinking in Space** | Yang, Yang, Gupta, Han, Fei-Fei, Xie | NYU, Yale, Stanford | **CVPR 2025 Oral** | Dec 2024 | VSI-Bench: first systematic spatial intelligence benchmark for MLLMs |
| 2 | **SpatialScore** | Wu et al. | Shanghai Jiao Tong, Shanghai AI Lab | arXiv (2025) | May 2025 | Holistic spatial benchmark (5K samples, 30 tasks) + SpatialAgent with 12 tools |
| 3 | **Agent-of-Thoughts Distillation** | Shi et al. | — | **CVPR 2025** | 2025 | Distills multi-step agent reasoning into a single Video-LLM via CoT tuning |
| 4 | **GCA** | Chen et al. | Beihang University | **CVPR 2026** (accepted) | Nov 2025 | Formal task constraints (C_task) decoupling semantics from geometry; +27% over SOTA |
| 5 | **VideoSeek** | Lin, Wu, Liu et al. | AMD, Univ. of Rochester | arXiv (Mar 2026) | Mar 2026 | Long-horizon video agent with overview/skim/focus tools; outperforms GPT-5 with 93% fewer frames |
| 6 | **RieMind** | Ropero et al. | Huawei Riemann Lab | arXiv (Mar 2026) | Mar 2026 | 3D scene graph agent with geometry tools; +16% on VSI-Bench over fine-tuned models |
| 7 | **TRACE** | Hua, Yin, Wu et al. | Tsinghua, Shanghai AI Lab, Univ. of Tokyo | arXiv (Mar 2026) | Mar 2026 | Prompting method for textual allocentric representations; consistent gains across models |
| 8 | **LVAgent** | Chen, Yue, Chen et al. | SIAT (CAS), Tsinghua, Shanghai AI Lab, SJTU | arXiv (Dec 2024) | Dec 2024 | Multi-agent collaboration (3+ MLLMs) with retrieval, debate, and agent expulsion for long video |
| 9 | **VADAR** | Marsili, Agrawal, Yue, Gkioxari | Caltech | arXiv (Mar 2025) | Mar 2025 | Training-free agentic program synthesis with dynamically generated API for 3D spatial reasoning |
| 10 | **VideoThinker** | Li, Chen, Han et al. | Zhejiang Univ., Fudan, Wuhan Univ., Shanghai AI Lab | arXiv (Jan 2026) | Jan 2026 | Agentic VideoLLM trained via caption-proxy data synthesis to internalize temporal tool use |

**Chronological pattern:** Problem identification + first multi-agent video systems (Dec 2024) → benchmarks & early spatial agents (2025, incl. VADAR's program synthesis) → wave of spatial agent solutions + agentic VideoLLMs (2026, multiple independent groups simultaneously).

---

## 2. Benchmarks — Why Spatial Reasoning Fails

### Thinking in Space (Yang et al., CVPR 2025 Oral)

**Credibility:** Top venue (CVPR Oral — top ~3% of submissions). Senior authors include Li Fei-Fei (Stanford, co-inventor of ImageNet) and Saining Xie (NYU, author of ResNeXt, ConvNeXt).

VSI-Bench: 5,000+ spatial QA pairs from 288 indoor egocentric videos (ScanNet, ScanNet++, ARKitScenes). 8 task types across configurational and measurement categories.

**Key findings that motivate our research:**
- Best model (Gemini-1.5 Pro) ~50% overall vs ~80% human
- **Error decomposition: 71% of failures are egocentric-allocentric transformation errors**, only 8% perception errors — VLMs *see* correctly but can't *reason* spatially
- CoT/ToT/Self-Consistency *hurt* spatial task performance (−4% avg) — spatial reasoning is not linguistic reasoning
- Cognitive maps (explicit spatial representations) improve distance estimation (+6%)
- Models build local but not global spatial maps — accuracy drops from ~64% (within 1 grid unit) to ~40% (4+ units)

**Implication:** The bottleneck is spatial reasoning, not perception. External reasoning scaffolding is the right direction.

### SpatialScore (Wu et al., 2025)

**Credibility:** Shanghai AI Lab is a leading Chinese AI research institution. Large-scale effort: 40 models evaluated, 5K manually verified samples.

Broadens the scope from VSI-Bench's 8 tasks to 30 tasks across 10 categories (mental animation, counting, depth, distance, motion, camera pose, temporal reasoning, view reasoning, size, localization). Includes both 2D and 3D spatial understanding.

**Key findings:**
- 26.48-point gap between best model (Gemini-3-Pro, 60.12) and human (86.60)
- Model scale correlates with spatial performance
- Existing spatial fine-tuning often causes catastrophic forgetting on other tasks
- SpatialAgent (12 tools, ReAct paradigm) achieves +7.78 without any training — competitive with fine-tuning (+10.47) while being training-free and more robust

**Implication:** The problem is broad and deep; training-free agent approaches are viable and avoid the forgetting problem.

For detailed VSI-Bench vs SpatialScore comparison, see [vsibench-vs-spatialscore.md](vsibench-vs-spatialscore.md).

---

## 3. Agent Approaches — Ordered by Externalization Degree

The central question: *how much spatial reasoning should be externalized from the VLM?* These 8 papers span a spectrum from prompting-only to full geometric externalization.

*See diagram: [Externalization Spectrum](https://www.figma.com/online-whiteboard/create-diagram/21cc84c4-c84d-416d-bc2f-642a940c0845?utm_source=claude&utm_content=edit_in_figjam)*

### 3.1 TRACE — Prompting Only (Least Externalized)

**What it externalizes:** The *format* of spatial reasoning — forces the VLM to generate a structured allocentric representation (meta-context + camera trajectory + entity registry) before answering.

**What VLM still does:** Everything — perception, coordinate estimation, trajectory reconstruction, and final reasoning. No external tools or modules.

**Credibility:** Tsinghua University (top Chinese university), Shanghai AI Lab, University of Tokyo. Not yet peer-reviewed (arXiv March 2026).

**Method:** TRACE = ⟨Meta Context M, Camera Trajectory T, Entity Registry E⟩. The VLM generates this structured text representation from egocentric video, then uses it as a "spatial cache" in the same context window to answer spatial questions.

**Key results:**
- +7.54% over Direct on VSI-Bench (Gemini 3 Pro), +5.40% (MiMo-VL-7B)
- Only method that consistently improves across all tested model families (unlike CoT/ToT which sometimes hurt)
- **Critical finding: one-stage generation > two-stage** — pre-generating TRACE and feeding as context is worse than letting the model generate it inline. The *process* of constructing the representation matters, not just having it.

**What breaks:** Self-generated representations are bounded by VLM's own perception quality. No systematic evaluation of coordinate estimation accuracy. Static representation — doesn't handle dynamic scenes.

### 3.2 Agent-of-Thoughts Distillation (AoTD) — Agent as Teacher

**What it externalizes:** Reasoning at training time — a multi-tool agent system generates reasoning traces, which are distilled into a single Video-LLM via CoT instruction tuning. At inference time, the model runs end-to-end (no agent).

**What VLM still does:** At inference: everything. The agent is only used to generate training data.

**Credibility:** CVPR 2025 (top-tier venue).

**Method:** (1) Benchmark specialist models on atomic sub-tasks → (2) Decompose VideoQA into Python programs calling specialist tools → (3) Convert execution traces to natural language CoTs with LLM-based verification → (4) Fine-tune Video-LLM on verified CoTs.

**Key results:**
- 74.3% on STAR, 77.6% on NExT-QA (temporal/compositional VideoQA)
- ~5x faster, ~3.5x less memory than the agent system at inference
- But: only 28.8% on VSI-Bench — spatial reasoning remains the weakest category
- CoT yield only ~20% (32.3K verified from 158.6K pairs)

**What breaks:** Only 2D bounding-box-level spatial reasoning. No true 3D or allocentric understanding. Performance depends heavily on specialist model quality. The distillation approach compresses but doesn't solve the spatial reasoning gap.

### 3.3 VADAR — Program Synthesis with Dynamic API

**What it externalizes:** All spatial reasoning logic — the VLM only handles perception (object detection, depth estimation, VQA). Spatial reasoning is compiled into generated Python programs that call vision specialists.

**What VLM still does:** Perception only (via specialists: Molmo + GroundingDINO for detection, SAM2 for segmentation, UniDepth for depth, GPT-4o for VQA). The LLM (GPT-4o) generates programs but never directly reasons about spatial relationships.

**Credibility:** Caltech (Georgia Gkioxari, Yisong Yue — well-known CV/ML researchers). arXiv March 2025, not yet peer-reviewed.

**Method:** Two-phase architecture:
- **Phase 1 (API Generation):** Signature Agent sees 15 queries → proposes reusable function signatures (e.g., `_is_behind`, `_find_closest_object_3D`) → Implementation Agent writes them using base modules (`loc()`, `depth()`, `vqa()`, `same_object()`, `get_2D_object_size()`) → Test Agent validates via Python interpreter
- **Phase 2 (Program Synthesis):** Program Agent generates a Python program per query using the generated API → Execution Agent runs it against vision specialists with error retry (up to 5 attempts)

**Key results:**
- CLEVR: 53.6% (vs GPT-4o 58.4%, ViperGPT 42.6%, VisProg 39.9%)
- Omni3D-Bench: 40.4% (vs GPT-4o 42.9%) — competitive without any training
- **Oracle accuracy: 83.0% CLEVR, 94.4% Omni3D-Bench** — proves program logic is correct; vision specialists are the bottleneck
- Outperforms LEFT (neuro-symbolic, needs 10K+ training samples) with zero training
- VSI-Bench-img subset: 50.1% vs Gemini 1.5 Pro's 49.5%

**What breaks:** **Single-image only** — no video, no temporal reasoning. Depth is monocular (UniDepth), not metric 3D — spatial relations like "behind" are approximated via depth comparison, not geometric reasoning. No ego-allo transformation. API generation is question-aware (sees 15 queries), so somewhat tailored to benchmark distribution. Expensive (~35.7s per question on A100).

**Key insight for our work:** VADAR proves the externalization thesis quantitatively — the 30-40% gap between execution and oracle accuracy shows program logic works but perception fails. The dynamic API generation pattern (agents creating reusable spatial functions on-the-fly) is more flexible than hand-crafted DSLs. The depth-as-3D approximation has a clear ceiling; true 3D reconstruction with ego-allo transformation would unlock the next level.

### 3.4 LVAgent — Multi-Agent Collaboration

**What it externalizes:** *Which model to use* and *where to look* in time. Multiple MLLMs collaborate through debate, with underperforming agents dynamically expelled.

**What VLM still does:** All perception and reasoning — agents are black-box answer generators that debate textually. No external geometric or spatial tools.

**Credibility:** SIAT (CAS), Tsinghua AIR, Shanghai AI Lab, SJTU. arXiv December 2024.

**Method:** Four-step loop (up to 3 rounds):
1. **Selection** — pseudo-label voting on 150 samples to rank agents per task domain; top 3 form the team
2. **Perception** — 3-stage retrieval: agents decide if full viewing needed → generate key information → ASP-CLIP (finetuned on 82K LongVR dataset) scores 6 video chunks
3. **Action** — each agent answers; if >50% agree, early stop; else trigger Reflection
4. **Reflection** — agents score each other's reasoning; lowest-scored agent expelled; loop back with updated context

**Key results:**
- First agent method to exceed 80% on all four major long-video benchmarks
- EgoSchema: 82.9% (vs Qwen2-VL-72B 77.9%), LongVideoBench: 80.0% (vs LLaVA-Video-72B 64.9%)
- +13.3% over GPT-4o on LongVideoBench
- Efficient: 71.2 frames and 33.6s per video on average (vs 568 frames / 90.5s for Qwen2-VL alone)

**What breaks:** **MCQ-only** — entire pipeline assumes discrete answer space. Heavy infrastructure (3+ large models, 8x A800 GPUs). No spatial reasoning at all — purely temporal retrieval and debate. Fixed 6-chunk video segmentation ignores content structure. Task-specific agent selection requires 6.58 hours of preselection per new dataset.

**Relevance to our work:** Low for spatial reasoning specifically, but the **multi-agent collaboration pattern** (selection → debate → expulsion) and **retrieval-then-reason paradigm** are transferable design patterns. Demonstrates that no single MLLM is best at everything — dynamic agent teams outperform any individual model by 5-19%.

### 3.5 VideoThinker — Agentic VideoLLM with Internalized Tool Use

**What it externalizes:** *Where to look* in time — temporal retrieval and zoom tools. But unlike external agent pipelines, the tool-use capability is **trained into a single 7B VideoLLM** via synthetic data.

**What VLM still does:** Everything at inference — perception, tool invocation decisions, and reasoning. The model has learned to autonomously trigger retrieval/zoom when uncertain.

**Credibility:** Zhejiang University, Fudan, Wuhan University, Shanghai AI Lab. arXiv January 2026.

**Method:**
- **Data synthesis (caption-proxy trick):** VideoLLM generates captions → agentic LLM (Qwen3-235B) reasons over captions using 6 tools (ClipRetrieval, SubtitleRetrieval, SubtitleSummary, FrameZoom, SubtitleZoom, CaptionZoom) → captions swapped back to actual video frames → yields video-interleaved reasoning traces
- **Training:** LoRA fine-tune Qwen2.5-VL-7B on 10K samples (CG-Bench)
- **Inference:** Confidence-gated — if γ = exp(avg log-prob) > τ (0.7), answer directly; otherwise trigger multi-turn tool reasoning

**Key results:**
- MLVU: 54.8% (+6.8 over base 7B), matching GPT-4o (54.9%)
- LVBench: 48.9% (+10.6), matching GPT-4o (48.9%)
- Outperforms all agentic LLM baselines (VideoAgent, VideoTree, VideoExplorer) by 15-20%
- Gains scale with video duration: +0.2% on short (<2min), +3.7% on long (>15min)

**What breaks:** MCQ-only evaluation. VideoMME gap remains large (53.7% vs GPT-4o 72.1%). Caption-proxy may lose spatial layout information during synthesis. All 6 tools are purely temporal — no spatial tools. Fixed 10s clip segmentation. Frame budget capped at 64.

**Relevance to our work:** The **caption-proxy data synthesis** pattern is transferable — generate spatial reasoning trajectories over scene descriptions, then swap to actual images at training time to bootstrap spatial tool-use. The **confidence-gated tool invocation** (easy questions answered directly, hard ones triggering tools) is an efficient inference pattern. However, all tools are temporal; adapting this for spatial reasoning would require replacing temporal retrieval/zoom with spatial tools (depth estimation, viewpoint transformation, spatial graph construction).

### 3.6 VideoSeek — Temporal Navigation Agent

**What it externalizes:** *Where to look* in time — the agent reasons about which video segments to inspect, using a coarse-to-fine toolkit. Perception and spatial reasoning remain in the VLM.

**What VLM still does:** All spatial/visual reasoning. The agent only navigates temporally.

**Credibility:** AMD Research and University of Rochester. arXiv March 2026, not yet peer-reviewed. Strong empirical results on established benchmarks.

**Method:** ReAct-style think-act-observe loop (max 20 turns) with three tools:
- **`<overview>`** — 16α frames across full video, packed into 2×4 grids
- **`<skim>`** — 4α frames in a candidate segment (coarse scan)
- **`<focus>`** — ~1 FPS on a short clip (full resolution)

Two-call design: thinking LLM generates free-text reasoning, separate LLM call parses into tool calls.

**Key results:**
- LVBench: 68.4% (vs GPT-5 base 60.1%) using 92 frames (vs ~384 baseline) — **76% fewer frames**
- With subtitles: 76.7% using only 27 frames — **93% fewer frames**
- Outperforms DVD (prior best agent, 8,074 frames) while using ~1% of its frames
- Ablation: removing `<overview>` has largest impact (−13.3 pts)

**What breaks:** Purely temporal — no spatial tools. Relies entirely on GPT-5 API. ~13 LLM calls per question. Not suited for anomaly detection (needs logic flow to navigate).

**Relevance to our work:** Not a spatial reasoning system, but the **multi-granularity toolkit pattern** (overview → skim → focus) and the **decoupled thinking + action parsing** are directly transferable design patterns. Adapt for spatial: scene_overview → spatial_scan → precise_measurement.

### 3.7 GCA — Formal Geometric Constraints

**What it externalizes:** Geometric computation — the VLM formalizes the problem into constraints, then external tools (3D reconstruction, detection, projection, pose estimation) compute the answer.

**What VLM still does:** Task formalization (translating natural language query into formal constraints) and agent planning (choosing which tools to call).

**Credibility:** Beihang University (strong in aerospace/robotics). **Accepted at CVPR 2026** — peer-reviewed at top venue.

**Method:** Two-stage architecture with formal task constraint C_task = (C_R, C_O):
- **Stage 1 (Task Formalization):** VLM translates query into Reference Frame Constraint C_R (which coordinate system?) + Objective Constraint C_O (what to compute?)
- **Stage 2 (Constrained Computation):** ReAct-style agent uses 8 specialized tools (reconstruct, detect, project_box_to_3d_points, predict_obj_pose, estimate_scale, ocr, analyze_motion, code execution with KACG)

**Key results:**
- 64.8% average across 5 benchmarks (new SOTA)
- +12% over Gemini-2.5-Pro, +27% over training-based methods, +38% over TIGeR agent
- **Ablation: removing C_R (reference frame): −6.6 pts; removing C_O (objective): −1.2 pts** — reference frame identification is 5× more important. This is the ego-allo problem.
- Generalizable across VLMs (works with different backbones)

**What breaks:** 30% of errors from incorrect formalization (VLM still responsible for the hardest step). **Image-only evaluation** — no video benchmarks, no VSI-Bench. Heavy infrastructure (2 A100 GPUs, Ray + LangGraph). Abstract spatial concepts (e.g., "south of") use fragile proxies.

**Key insight for our work:** The C_task formalism — especially C_R — directly addresses the ego-allo transformation problem. The ablation proving C_R >> C_O validates that reference frame identification is the core challenge. Video extension is the natural next step.

### 3.8 RieMind — Full Geometric Externalization (Most Externalized)

**What it externalizes:** All geometry and spatial reasoning — the LLM never sees images. A persistent 3D Scene Graph (3DSG) provides complete geometric information; the LLM reasons purely via tool calls.

**What VLM still does:** In the ideal version, perception (building the 3DSG from video). In the paper's experiments: nothing — ground-truth 3DSG is used.

**Credibility:** Huawei Riemann Lab (well-funded industry research). arXiv March 2026, not yet peer-reviewed. Strong results but with the ground-truth caveat.

**Method:** Two-layer architecture:
- **Perception layer:** 3D Scene Graph with 4-node hierarchy (building → floor → room → object), storing geometry, dimensions, orientation, location
- **Reasoning layer:** LLM with tool access via MCP servers across 4 namespaces:
  - Memory tools (scene context)
  - Scene tools (~15 graph traversal tools)
  - Geometry tools (volume, dimensions, surface area, distance)
  - Location/orientation tools (position, frame construction, projection, look-at)

**Key results:**
- Qwen2.5-VL-7B: 31.2% → 64.1% (+32.9 pts with agent)
- GPT-4o: 35.3% → 85.2% (+49.9 pts with agent)
- **GPT-4.1: 89.5% — vs 73.6% best fine-tuned model** (new SOTA on VSI-Bench static)
- Absolute questions (count, distance, size) see massive improvements
- Relative direction remains hardest — Qwen actually degrades (−3.8)

**What breaks:** **Ground-truth 3DSG is the critical limitation** — constructing it from RGB-D is acknowledged as future work. Cost/latency unreported. Even with perfect tools + data, base model reasoning capability matters hugely (Qwen 34.7% vs GPT-4.1 87.3% on relative direction). Static scenes only.

**Key insight for our work:** Proves the externalization hypothesis — when you provide ground-truth geometry and let the LLM reason with tools, the approach crushes fine-tuned models. The gap is the perception pipeline. This is exactly where our research fits: building a practical perception-to-3DSG pipeline that doesn't require ground truth.

---

## 4. Cross-Paper Comparison

| | Spatial Repr. | Agent Paradigm | Ego-Allo Handling | Training? | Input | Benchmark | Key Result | Code? |
|---|---|---|---|---|---|---|---|---|
| **Thinking in Space** | Cognitive map (10×10 grid) | — (benchmark) | Diagnosed as bottleneck | No | Video | VSI-Bench | ~50% best model vs ~80% human | Yes |
| **SpatialScore** | Per-frame tool outputs | Plan-Execute / ReAct | Implicit (via tools) | Optional | Image/Video | SpatialScore | +7.78 (agent, training-free) | Yes |
| **AoTD** | 2D bounding boxes | Distillation (agent→model) | Not addressed | Yes (distillation) | Video | STAR, NExT-QA | 74.3% STAR; 28.8% VSI-Bench | Yes |
| **VADAR** | Pixel coords + monocular depth | Program synthesis (dynamic API) | **None** (depth comparison only) | No | **Image only** | CLEVR, Omni3D-Bench | 94.4% oracle; 50.1% VSI-Bench-img | Yes |
| **LVAgent** | None (temporal only) | Multi-agent debate + expulsion | Not addressed | Partial (CLIP finetune) | Video | EgoSchema, LongVideoBench | 82.9% EgoSchema, >80% all 4 benchmarks | Yes |
| **VideoThinker** | None (temporal only) | Internalized tool use (trained) | Not addressed | Yes (LoRA, 10K) | Video | MLVU, LVBench | 48.9% LVBench (= GPT-4o) at 7B | No |
| **GCA** | 3D coordinates via tools | Formal constraints + ReAct | **Formalized as C_R** | No | **Image only** | 5 spatial benchmarks | 64.8% avg (+27% over SOTA) | No |
| **VideoSeek** | None (temporal only) | ReAct (think-act-observe) | Not addressed | No | Video | LVBench, VideoMME | +8.3 on LVBench, 93% fewer frames | Yes |
| **RieMind** | **3D Scene Graph** | Tool-augmented LLM | **Geometric tools** | No | **Ground-truth 3DSG** | VSI-Bench | **89.5%** (GPT-4.1) | No |
| **TRACE** | Allocentric text repr. | Prompting only | Prompted generation | No | Video | VSI-Bench, OST-Bench | +7.54% (Gemini 3 Pro) | No |

---

## 5. Relevance Assessment of Self-Found Papers

### GCA — **HIGH relevance**
Closest existing work to our research thesis. The C_task = (C_R, C_O) formalism directly addresses the ego-allo transformation problem, and the ablation proves reference frame identification (C_R) is 5× more important than the objective (C_O). The main gap — image-only, no video extension — is exactly our opportunity. Accepted at CVPR 2026, so it's credible and will be a key paper to position against.

### RieMind — **HIGH relevance**
Strongest empirical validation of the externalization hypothesis. 89.5% on VSI-Bench (vs 73.6% fine-tuned SOTA) proves the approach works in principle. The ground-truth 3DSG assumption is both the paper's limitation and our research opportunity: building a practical perception pipeline to construct the scene graph from RGB(-D) video. Huawei Riemann Lab is a serious research group.

### VADAR — **HIGH relevance**
Directly validates our core thesis: VLMs are bottlenecked by spatial reasoning, not perception. The oracle experiments (83% CLEVR, 94.4% Omni3D-Bench with perfect vision specialists) quantitatively prove that externalizing spatial logic into programs works — the gap comes from vision specialist errors, not reasoning failures. The dynamic API generation pattern (agents creating reusable spatial functions on-the-fly vs. hand-crafted DSLs) is a compelling design choice. However, VADAR's "3D reasoning" is really "2D + depth ordering" — no true ego-allo transformation, no metric 3D, no video. Caltech (Gkioxari, Yue) lends strong credibility.

### TRACE — **MEDIUM-HIGH relevance**
Validates that even prompting-only externalization of allocentric representations helps. The one-stage > two-stage finding is architecturally important for us — it suggests our agent should interleave perception and reasoning rather than pre-computing a full scene graph then reasoning over it. The entity registry design (temporal stamping, visual signature, metric estimation, spatial relations) is a useful template. Less relevant because it doesn't use external geometric tools — but the findings inform agent design.

### VideoThinker — **MEDIUM relevance**
The architecture pattern — synthesize reasoning data via caption-proxy, train a single model to internalize tool use, gate invocation on confidence — is elegant and transferable. The caption-proxy data synthesis trick could bootstrap spatial tool-use without needing a model that already handles spatial reasoning. However, all 6 tools are purely temporal (retrieval/zoom in time), with no spatial awareness. More relevant as a methodological template (how to train agentic tool-use into a VideoLLM) than as a spatial reasoning approach.

### AoTD — **MEDIUM relevance**
The agent→distillation pattern is interesting for future work (if our agent works, we could distill it into a faster model). But the paper itself doesn't address spatial reasoning in any sophisticated way — 2D bounding boxes only, 28.8% on VSI-Bench. More relevant as a methodological reference than a spatial reasoning approach. CVPR 2025 publication confirms credibility.

### LVAgent — **LOW relevance**
Demonstrates that multi-agent collaboration (+13-19% over single models) works for long video understanding, but the approach is entirely temporal — no spatial primitives, no geometric reasoning, no ego-allo handling. MCQ-only evaluation. The multi-agent debate/expulsion pattern is interesting but tangential to our spatial reasoning focus. Heavy infrastructure requirements (3+ 72B models simultaneously).

---

## 6. Synthesis — Mapping to Our Research Idea

Our proposed architecture (from [idea-agent-architecture.md](../idea-agent-architecture.md)) has 6 core capabilities. Here's how each paper informs them:

| Capability | Best Reference | What We Learn |
|---|---|---|
| **1. Decompose spatial tasks** | GCA (C_task formalism), VADAR (dynamic API signatures) | Formalize query as reference frame + objective constraint *before* any computation; consider dynamic function generation for novel spatial subproblems |
| **2. Step-by-step external reasoning** | RieMind (5-step pipeline), GCA (formalize→compute), VADAR (program synthesis) | Separate formalization from execution; use deterministic geometric tools; VADAR's oracle results (83–94%) prove program logic works |
| **3. Multi-turn VLM interaction** | VideoSeek (ReAct loop), SpatialAgent (ReAct/PE), VideoThinker (confidence-gated) | Think-act-observe with full history; decouple thinking from action parsing; gate tool invocation on model confidence |
| **4. Cross-frame integration** | TRACE (camera trajectory), VideoSeek (progressive evidence), VideoThinker (retrieval+zoom) | Track trajectory + entity registry across frames; accumulate evidence iteratively; temporal retrieval before spatial reasoning |
| **5. Guide attention to spatial structures** | VideoSeek (coarse-to-fine), GCA (formalize-before-compute) | Multi-granularity toolkit: overview → spatial scan → precise measurement |
| **6. Build consistent spatial representation** | RieMind (3DSG), TRACE (allocentric text), GCA (workspace) | Persistent world-space representation with geometric grounding |

### The Gap

No existing paper combines all four requirements:

| Requirement | Thinking in Space | SpatialScore | AoTD | VADAR | LVAgent | VideoThinker | GCA | VideoSeek | RieMind | TRACE |
|---|---|---|---|---|---|---|---|---|---|---|
| **Video input** | Benchmark | Partial | Yes | **No** | Yes | Yes | **No** | Yes | **No (GT 3DSG)** | Yes |
| **Externalized geometric reasoning** | No | Partial | No | **Yes (programs)** | No | No | **Yes** | No | **Yes** | No |
| **Practical perception pipeline** | N/A | Yes | Yes | Yes | Yes | Yes | Yes | Yes | **No (GT)** | Yes |
| **Explicit ego-allo handling** | Diagnosed | Implicit | No | **No** | No | No | **Yes (C_R)** | No | **Yes (tools)** | Prompted |

The uncovered cell: **video input + externalized geometric reasoning + practical perception + explicit ego-allo handling**. VADAR comes closest on externalized reasoning but lacks video and ego-allo. This is where our research fits.

*See diagram: [Research Gap](https://www.figma.com/online-whiteboard/create-diagram/1f0a6a1a-ae98-4d2e-a4a1-d7642a4ef2ea?utm_source=claude&utm_content=edit_in_figjam)*

---

## 7. Implications for Adjusting Research Direction

Based on this survey, concrete recommendations:

1. **Adopt GCA's C_task formalism for video.** The C_R/C_O decomposition is the cleanest formalization of the ego-allo problem. Extend it to handle dynamic/time-varying reference frames C_R(t) for video.

2. **RieMind proves the ceiling.** 89.5% on VSI-Bench with ground-truth 3DSG is the upper bound. Our contribution is the practical perception pipeline that approaches this ceiling without ground truth.

3. **Interleave perception and reasoning (TRACE's lesson).** Don't just pre-compute a scene graph and reason over it. TRACE's one-stage > two-stage finding suggests the agent should iteratively build its spatial understanding, querying the VLM for specific visual details as geometric reasoning demands them.

4. **Borrow VideoSeek's multi-granularity toolkit pattern.** Adapt overview/skim/focus for spatial exploration: scene_overview (what objects, rough layout) → spatial_scan (specific regions, depth estimation) → precise_measurement (geometric tool calls for exact values).

5. **Evaluate on VSI-Bench (primary) + SpatialScore (breadth).** VSI-Bench is where RieMind sets the bar; SpatialScore provides breadth across 30 task types. Both benchmarks have code available for evaluation.

6. **AoTD distillation is a future direction, not the first step.** If the agent works, distilling it into a faster model is a natural follow-up. But first: build the agent that actually solves spatial reasoning.

7. **Address the reference frame bottleneck.** GCA's ablation (C_R 5× more important than C_O) and Thinking in Space's error analysis (71% ego-allo errors) both point to the same thing: identifying and transforming between reference frames is the core unsolved problem.

---

## References

1. Yang et al. "Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces." CVPR 2025 Oral. [arXiv:2412.14171](https://arxiv.org/abs/2412.14171)
2. Wu et al. "SpatialScore: Towards a Unified Evaluation for Multimodal Spatial Understanding." arXiv 2025. [arXiv:2505.17012](https://arxiv.org/abs/2505.17012)
3. Shi et al. "Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation." CVPR 2025.
4. Chen et al. "Geometrically-Constrained Agent." CVPR 2026. [arXiv:2511.22659](https://arxiv.org/abs/2511.22659)
5. Lin et al. "VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking." arXiv 2026. [arXiv:2603.20185](https://arxiv.org/abs/2603.20185)
6. Ropero et al. "RieMind: Towards Grounded Spatial Reasoning in LLMs through Riemannian Geometric Tools." arXiv 2026. [arXiv:2603.15386](https://arxiv.org/abs/2603.15386)
7. Hua et al. "TRACE: Unleashing Spatial Reasoning in MLLMs via Textual Representation Guided Reasoning." arXiv 2026. [arXiv:2603.23404](https://arxiv.org/abs/2603.23404)
8. Chen et al. "LVAgent: Long Video Understanding by Multi-Round Dynamical Collaboration of MLLM Agents." arXiv 2024. [arXiv:2503.10200](https://arxiv.org/abs/2503.10200)
9. Marsili et al. "VADAR: Visual Agentic AI for Spatial Reasoning with a Dynamic API." arXiv 2025. [arXiv:2502.06787](https://arxiv.org/abs/2502.06787)
10. Li et al. "VideoThinker: Building Agentic VideoLLMs with LLM-Guided Tool Reasoning." arXiv 2026. [arXiv:2601.15724](https://arxiv.org/abs/2601.15724)
