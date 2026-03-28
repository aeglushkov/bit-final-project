# "SpatialScore" — Paper Analysis

## 1. Paper Summary

**"SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence"** (Wu et al., SJTU / Shanghai AI Lab, arXiv 2505.17012) tackles the fragmented state of spatial intelligence evaluation in MLLMs. The authors make four contributions: (1) **SpatialScore**, a benchmark of ~5K manually verified samples across 30 tasks and 10 categories; (2) extensive evaluation of 40 MLLMs revealing persistent challenges; (3) **SpatialCorpus**, a 331K-sample training resource for spatial fine-tuning; and (4) **SpatialAgent**, a multi-agent framework with 12 specialized spatial perception tools that improves spatial reasoning without additional training.

---

## 2. What Problem Does This Paper Solve?

Existing spatial intelligence benchmarks suffer from two key limitations:
- **Over-simplistic tasks:** Focus on superficial spatial queries (object presence, coarse position relations) while neglecting rigorous visual geometry perception (camera pose, depth, homography).
- **Narrow evaluation scope:** Fragmented assessments rely on naive questions (Yes/No), single-modality inputs (static images), or isolated skills (size estimation only).

SpatialScore aims to be the first **holistic** spatial intelligence benchmark covering diverse data types, modalities, QA formats, and 30 distinct task types.

---

## 3. SpatialScore: The Benchmark

### 3.1 Data Sources & Construction

Two-pronged construction approach:

**A. 3D Data Repurposing (2.3K new samples):**
- Sample 500 scenes from ScanNet++, Omni3D, WildRGB-D, PointOdyssey, and CA-1M
- Leverage precise 3D annotations (depth, 3D bounding boxes) to generate QA pairs
- Use templates + LLM rewriting (DeepSeek-v3) for linguistic diversity
- Convert to judgment, multi-choice, and open-ended formats
- Distractors generated via: same-category sampling, small perturbations, LLM-synthesized confusers

**B. Integration of 23 Existing Datasets (curated):**
- SRBench, SpatialEval, SpatialViz-Bench, SpatialSense, VSR, VSI-Bench, Space-10, MIRAGE, 3DSRBench, QSpatialBench, STI-Bench, VLM4D, SITE-Bench, SPAR-Bench, MMSI-Bench, OmniSpatial, RoboSpatial, CV-Bench, MMVP, BLINK, MMIU, RealWorldQA
- Initial pool: 63,857 candidates
- GPT-OSS-120B filtering removes text-answerable questions → 40,238
- Manual verification → final **5,025 samples**

### 3.2 Task Taxonomy (10 Categories, 30 Tasks)

| Category | Tasks | QA Format |
|----------|-------|-----------|
| **Mental Animation** | Spatial Map, Maze Navigation, Multi-view Projection, Spatial Folding, 2D/3D Rotation | MC, Judgment |
| **Counting** | Object Counting, Value Counting, Count with Relation | Open-ended, MC |
| **Depth Estimation** | Relative Depth, Absolute Depth | MC, Open-ended |
| **Object Distance** | Absolute Distance, Relative Distance, Camera Distance | Open-ended, MC |
| **Object Motion** | Object Motion | MC |
| **Camera Pose & Motion** | Camera Motion, Camera Intrinsics, Camera Extrinsics, Homography Matrix | MC |
| **Temporal Reasoning** | Navigation Route, Appearance Order | MC |
| **View Reasoning** | Orientation, Object Existence | MC, Judgment |
| **Object Size** | Relative Size, Absolute Size, Size Compatibility | MC, Open-ended |
| **Object Localization** | 2D Localization, 3D Object Detection, Spatial Position | MC, Open-ended |

### 3.3 Key Design Advantages over Prior Benchmarks

| Feature | SpatialScore | Prior Benchmarks |
|---------|-------------|-----------------|
| Data types | Real + Simulated + AIGC | Usually real only |
| Input modalities | Image + Multi-frame + Video | Usually image only |
| QA formats | Judgment + MC + Open-ended | Usually 1-2 formats |
| Tasks | 30 | Typically 2-12 |
| Samples | 5,025 | 271-8,068 |
| Manual verification | Yes | Varies |

### 3.4 Evaluation Metrics

- **Judgment & MC:** Exact-match accuracy against ground truth
- **Open-ended numerical:** Mean Relative Accuracy (MRA) — same as VSI-Bench
- **Complex open-ended:** Average of (i) rule-based parsing + (ii) LLM scoring (GPT-OSS-20B)

---

## 4. Main Evaluation Results (40 Models)

### 4.1 Overall Rankings

| Model | Overall Score | Category |
|-------|-------------|----------|
| Human-level | **86.60** | — |
| Gemini-3-Pro | **60.12** | Proprietary (best overall) |
| GPT-5 | 58.13 | Proprietary |
| Claude-4.5-Sonnet | 45.68 | Proprietary |
| Qwen3-VL-235B-A22B | **56.63** | Open-source (best) |
| Qwen3-VL-30B-A3B | 50.71 | Open-source |
| InternVL3-14B | 44.89 | Open-source |
| Qwen3-VL-8B | 45.48 | Open-source |
| InternVL2.5-8B | 36.92 | Open-source (small) |

### 4.2 Key Findings

#### Finding 1: Large gap to human performance (26.48 points)
- Best model (Gemini-3-Pro) achieves 60.12 vs. human 86.60
- Gap is especially pronounced on: **view reasoning, camera pose, motion analysis, real-world 3D perception**
- Some tasks approach human level: mental animation, object localization

#### Finding 2: Model scale correlates with performance
- Consistent within InternVL and Qwen-VL families
- Larger models generally perform better across all categories
- But even the largest models have significant weaknesses

#### Finding 3: Existing spatial fine-tuning often hurts
- SpaceThinker, SpaceR, and similar spatially fine-tuned models deliver only marginal gains or **underperform their base models**
- This indicates current fine-tuning strategies and datasets are partial and insufficient
- Underscores the diversity and difficulty of SpatialScore

#### Finding 4: Fundamental 3D understanding remains weak
- Near-human performance on some tasks (mental animation, object localization)
- But persistent struggles with: view reasoning, camera pose, motion analysis, real-world 3D perception
- This exposes a "pronounced deficiency in realistic 3D understanding"

---

## 5. SpatialCorpus: Training Data (331K samples)

### 5.1 Construction
- Same 3D repurposing pipeline as SpatialScore (no test set overlap)
- 273K real-world & simulation QA pairs across 16 tasks in 7 categories
- 58K synthetic rendered data for mental animation (spatial maps, 2D/3D rotation)
- Rule-based templates (cheaper than LLM rewriting at scale)
- Excludes CA-1M (class-agnostic, not useful for training)

### 5.2 Fine-tuning Results

| Model | Base Score | + SpatialCorpus | Improvement |
|-------|-----------|-----------------|-------------|
| Qwen3-VL-4B | 42.52 | 52.99 | **+10.47** |
| Qwen3-VL-8B | 45.48 | 54.71 | **+9.23** |

**However**, gains concentrate on already-optimized tasks (mental animation, camera analysis). Tasks with limited data scalability (view reasoning) may suffer from catastrophic forgetting. This motivates the agent-based approach.

---

## 6. SpatialAgent: Multi-Agent System

### 6.1 Architecture Overview

SpatialAgent uses a VLM (e.g., Qwen3-VL) as the "agent core" and orchestrates 12 specialized spatial perception tools. Two reasoning paradigms are supported:

### 6.2 Plan-Execute (PE) Paradigm

Three components: **Planner** → **Executor** → **Summarizer**

```
1. Planner (Φ_plan): Given question q, visual input v, and toolbox T:
   p = {(t_1, args_1), ..., (t_k, args_k)}

2. Executor (Φ_exec): Sequentially executes tools:
   Y = {y_1, ..., y_k}

3. Summarizer (Φ_sum): Produces final answer from tool outputs + original inputs:
   r_PE = Φ_sum(Y, q, v)
```

**Strengths:** Efficient plan formulation and execution
**Weaknesses:** Predetermined plan may sacrifice precision in complex scenarios

### 6.3 ReAct Paradigm

Three components: **Observer** → **Executor** → **Summarizer** (with memory M)

```
At each step i:
  1. Observer decides next action: o_i = Φ_obs(M_i, q, v)
  2. Executor processes: y_i = Φ_exec(o_i)
  3. Memory updates: M_{i+1} = M_i ∪ {(o_i, y_i)}
  4. Loop until Observer outputs "Terminate"
  5. Summarizer consolidates: r_ReAct = Φ_sum(M, q, v)
```

**Strengths:** Better flexibility through dynamic planning, adapts to intermediate outputs
**Weaknesses:** Reduced efficiency due to iterative nature

### 6.4 Toolbox (12 Tools in 4 Categories)

| Category | Tools | Models Used |
|----------|-------|-------------|
| **General Perception** | LocalizeObjects, GetObjectMask, 3D Detection | Rex-Omni (detection), SAM2 (segmentation), DetAny3D (3D bbox) |
| **Motion & Transform** | OpticalFlow, PointMatching, Extrinsics, Homography | RAFT (optical flow), VGGT (extrinsics), SIFT (matching/homography) |
| **Pose & Geometry** | Intrinsics, Orientation, Depth, 3D Distance | VGGT (camera params), Depth-Anything-V2 (depth), OrientAnything (orientation), MapAnything (3D reconstruction) |
| **Auxiliary** | Terminate, SelfThink | — |

Key design: All tools use **open-source models only**, ensuring reproducibility and allowing continuous improvement as underlying tools evolve.

### 6.5 Agent Results

| Configuration (Qwen3-VL-4B) | Overall |
|------------------------------|---------|
| Zero-shot baseline | 42.52 |
| + SpatialCorpus (fine-tuning) | 52.99 |
| + SpatialAgent-PE | 48.93 |
| + SpatialAgent-ReAct | **50.30** |

| Configuration (Qwen3-VL-8B) | Overall |
|------------------------------|---------|
| Zero-shot baseline | 45.48 |
| + SpatialCorpus (fine-tuning) | 54.71 |
| + SpatialAgent-PE | 52.75 |
| + SpatialAgent-ReAct | **53.81** |

**Key insight:** SpatialAgent yields slightly smaller absolute improvements than fine-tuning (+6.41/+7.27 for PE, +7.78/+8.33 for ReAct) BUT:
- Requires **no additional training**
- Consistently enhances performance across **nearly all tasks**
- More practical and robust as a drop-in improvement

---

## 7. Code Architecture

### 7.1 Repository Structure

```
SpatialScore/
├── README.md
├── dataset/
│   ├── SpatialScore.json          # Full benchmark (5K samples)
│   └── SpatialScore-Hard.json     # Hard subset
├── SpatialAgent/
│   ├── agent.py                   # UserAgent class (AutoGen-based)
│   ├── autogen/                   # Modified AutoGen framework
│   ├── utils/
│   │   ├── prompt.py              # CoTAPrompt, DirectAnswerPrompt, FeedbackPrompt
│   │   ├── parser.py              # JSON response parsing
│   │   ├── executor.py            # Tool execution engine
│   │   ├── observation.py         # BaseObservation for tool outputs
│   │   └── action_utils.py        # Image encoding utilities
│   ├── DepthAnythingV2/           # Metric depth estimation tool
│   ├── OrientAnything/            # 3D object orientation tool
│   └── RAFT/                      # Optical flow estimation tool
├── test_qwen.py                   # Evaluation script (Qwen/InternVL/all models)
└── assets/
```

### 7.2 Agent Architecture (Code Level)

The agent is built on a **modified AutoGen** framework:

1. **UserAgent** (`agent.py`): Extends `UserProxyAgent`. Manages the conversation loop:
   - `generate_init_message()`: Creates the initial prompt with tool specs + demos
   - `receive()`: Parses model responses, executes tools, sends feedback
   - Tracks `called_tools`, `new_image_paths`, `step_id`

2. **Prompt System** (`prompt.py`):
   - `CoTAPrompt`: Generates the full system prompt with goal, action metadata, format instructions, and 7 few-shot demos (optical flow, orientation, object detection, depth, segmentation, homography, camera params)
   - `DirectAnswerPrompt`: Simple direct-answer baseline
   - `FeedbackPrompt`: Generates observation feedback or error correction prompts

3. **Executor** (`executor.py`):
   - Resolves image paths (`image-0` → actual file path)
   - Dispatches to registered tool functions via `action_registry`
   - Handles image outputs (saves to result folder, updates image IDs)
   - Returns `BaseObservation` objects with results

4. **Conversation Loop:**
```
UserAgent → sends init prompt with question + tool specs
  → Assistant (VLM) generates JSON: {thought, actions: [{name, arguments}]}
    → UserAgent.receive() parses JSON
      → If action == "Terminate": extract final answer, stop
      → Else: Executor runs tool, returns observation
        → FeedbackPrompt formats observation
          → Send back to Assistant for next step
            → Loop until Terminate or max_consecutive_auto_reply
```

---

## 8. Critical Analysis

### 8.1 Strengths

1. **Most comprehensive spatial benchmark to date** — 30 tasks across 10 categories, integrating 23 existing datasets + new 3D-repurposed data
2. **Rigorous quality control** — GPT filtering + manual verification reduces 63K candidates to 5K high-quality samples
3. **Dual solution pathways** — data-driven (SpatialCorpus) and training-free (SpatialAgent) approaches are complementary and practical
4. **Agent design is well-motivated** — fine-tuning's catastrophic forgetting on underrepresented tasks justifies the agent approach
5. **Open-source toolbox** — all perception tools use open-source models, enabling reproducibility
6. **Scale of evaluation** — 40 models benchmarked, including latest proprietary (GPT-5, Gemini-3)
7. **Practical impact** — SpatialCorpus provides 331K training samples that could benefit the community

### 8.2 Limitations & Weaknesses

1. **RGB-only inputs** — no point clouds, depth maps, or surface normals as input modalities. This limits evaluation of models that could leverage richer 3D input.

2. **SpatialCorpus causes biased gains** — fine-tuning improvements concentrate on well-represented tasks; underrepresented tasks (view reasoning) may degrade. The authors acknowledge but don't fully resolve this.

3. **SpatialAgent toolbox is "relatively rudimentary"** — the authors themselves note this. Tool errors propagate (e.g., confusing depth with object distance in qualitative examples). Tool accuracy bottlenecks overall agent performance.

4. **AutoGen dependency** — the agent framework builds on a modified version of AutoGen, which adds complexity and may create maintenance burden.

5. **Limited agent paradigm comparison** — only PE and ReAct are tested. Other agent paradigms (Reflexion, tree-based planning) are not explored.

6. **No compositional analysis** — which tool combinations are most effective? Which tools fail most? The paper lacks granular analysis of tool-level performance.

7. **Prompt sensitivity not studied** — the 7 few-shot demos in CoTAPrompt are fixed. How sensitive is agent performance to demo selection?

### 8.3 Relevance to Our Research

**This paper is directly relevant to our research direction** of building an agent layer on top of VLMs that externalizes spatial reasoning:

| Aspect | SpatialScore Paper | Our Research Direction |
|--------|-------------------|----------------------|
| **Core idea** | Use tools to externalize spatial perception | Use tools to externalize spatial *reasoning* |
| **VLM role** | Agent core (planning + summarization) + perception via tools | Perception only — VLMs are good at seeing, bad at spatial reasoning |
| **Ego-allo transform** | Handled implicitly by tools (depth, pose) | Should be handled *explicitly* as a separate reasoning step |
| **Spatial reasoning** | Still delegated to the VLM (planner/summarizer) | Externalized to geometric computation |
| **Key difference** | Tools replace VLM *perception* (depth, flow, etc.) | We want tools to replace VLM *reasoning* (transformations, distance computation from perceived data) |

**Key takeaway for us:** SpatialAgent validates that tool-augmented VLMs improve spatial performance. But their approach still relies on the VLM for spatial *reasoning* (the planner decides what to do, the summarizer interprets results). Our hypothesis — that VLMs should only do perception while reasoning is fully externalized — is complementary and potentially more powerful. The SpatialScore benchmark itself would be an excellent evaluation target for our approach.

**Specific opportunities:**
- SpatialScore's 30 tasks provide a comprehensive test suite for our agent
- The finding that fine-tuning causes catastrophic forgetting supports our training-free approach
- Their toolbox (DepthAnythingV2, RAFT, VGGT, OrientAnything) could be reused in our pipeline
- The gap between SpatialAgent results and human performance (86.60 vs ~53) shows room for improvement through better reasoning externalization

---

## 9. Comparison with "Thinking in Space" (Yang et al.)

| Dimension | Thinking in Space (VSI-Bench) | SpatialScore |
|-----------|------------------------------|--------------|
| **Scope** | Video-based spatial QA, 8 tasks | Multi-modal spatial QA, 30 tasks |
| **Input** | Video only (egocentric walkthroughs) | Image + multi-frame + video |
| **Data types** | Real indoor scenes only | Real + simulated + AIGC |
| **Scale** | 5K QA pairs, 288 videos | 5K QA pairs from 23 datasets |
| **Models tested** | 15 | 40 |
| **Solutions proposed** | Cognitive maps (prompting) | SpatialCorpus (fine-tuning) + SpatialAgent (tools) |
| **Key finding** | CoT hurts spatial tasks; egocentric-allocentric transform is the bottleneck | Spatial fine-tuning can cause catastrophic forgetting; agents help |
| **Venue** | CVPR 2025 (Oral) | arXiv 2025 |
| **Complementarity** | VSI-Bench is *included* as one of SpatialScore's 23 source datasets | SpatialScore subsumes and extends VSI-Bench |

**Both papers converge on:** Current MLLMs have significant spatial reasoning deficits, and the gap to human performance is substantial. Standard prompting doesn't solve the problem. External spatial scaffolding (cognitive maps, tool-augmented agents) is needed.

**SpatialScore advances beyond Thinking in Space by:**
- Covering far more task types and modalities
- Proposing concrete solutions (training data + agent system)
- Evaluating at much larger scale (40 models)
- Including geometric perception tasks (homography, camera intrinsics/extrinsics) that VSI-Bench doesn't cover

---

## 10. Quick Reference

| | |
|---|---|
| **Title** | SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence |
| **Authors** | Haoning Wu*, Xiao Huang*, Yaohui Chen, Ya Zhang, Yanfeng Wang, Weidi Xie |
| **Affiliations** | SJTU, Shanghai AI Lab |
| **Venue** | arXiv 2505.17012 (Dec 2025) |
| **Benchmark** | SpatialScore: ~5K samples, 30 tasks, 10 categories, 23 source datasets |
| **Training data** | SpatialCorpus: 331K multimodal QA pairs |
| **Agent** | SpatialAgent: 12 tools, Plan-Execute + ReAct paradigms |
| **Best model** | Gemini-3-Pro (60.12) |
| **Human performance** | 86.60 |
| **Key insight** | Tools improve spatial reasoning without training; fine-tuning helps but causes forgetting; 26-point gap to human level remains |
| **Code** | github.com/haoningwu3639/SpatialScore |
| **Dataset** | huggingface.co/datasets/haoningwu/SpatialScore |
