# TRACE — Paper Analysis

## 1. Paper Summary

**"Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning"** (Hua et al., Tsinghua/Shanghai AI Lab/UTokyo, March 2026) proposes **TRACE** — a prompting method that guides MLLMs to generate a structured text-based allocentric representation of a 3D environment from egocentric video, before answering spatial questions. Drawing on cognitive science theories of allocentric spatial reasoning (Marr & Nishihara 1978, Klatzky 1998), TRACE forces the model to externalize its spatial understanding as a structured intermediate representation containing room layout, camera trajectory, and object registry.

---

## 2. What Problem Does This Paper Solve?

MLLMs struggle with 3D spatial reasoning because they:
1. Over-fixate on 2D visual signals and learn shortcut correlations
2. Fail to build hierarchical abstractions of 3D scene structure
3. Cannot reliably transform between egocentric and allocentric reference frames

Prior approaches address this via (a) large-scale supervised fine-tuning (limited scalability) or (b) adding geometric/stereo modalities (restricts applicability to off-the-shelf MLLMs). TRACE takes a third path: **prompting-only**, requiring no additional training or modules.

**Central question:** Can MLLMs be guided to explicitly construct and reason over structured allocentric representations of 3D environments from 2D visual observations?

---

## 3. Method: TRACE Components

TRACE is formally defined as a tuple: G = <M, T, E>

### 3.1 Meta Context (M)

Establishes a **Room Aligned Coordinate System**:
- Room topology (e.g., "Rectangular Hotel Room")
- Origin [0,0] at starting position of observer
- Y-axis aligned with the most salient straight line (characterized by large static objects), not camera heading
- Grid direction and initial heading relative to this grid
- Wall labels with notable objects along each wall

**Why this matters:** A common failure mode is losing track of camera initialization and coordinate system. By anchoring the coordinate system to the room geometry (not the camera), TRACE provides a stable allocentric frame.

### 3.2 Camera Trajectory (T)

A discrete sequence of steps: {(t_k, p_k, phi_k)} recording:
- Timestamp
- Estimated position [x, y] in meters relative to grid origin
- Facing direction (8 cardinal/ordinal directions, y-axis = north)
- Action description (egocentric camera-centric motion context)

**Design choices:**
- 8 discrete directions rather than continuous angles (accurate numerical angle estimation is too hard for current MLLMs)
- Action property encodes camera-centric motion context, enabling navigation/route-planning questions
- Effectively reconstructs the surveyor's path as a traversable spatial map

### 3.3 Entity Registry (E)

A registry of observed objects with structured attributes per entity:
- **Temporal Stamping:** First-seen timestamp (aids object tracking)
- **Visual Signature:** Brief appearance description (disambiguates visually similar instances)
- **Metric Estimation:** Plausible 2D coordinates [x, y] in meters relative to grid origin — forces geometric commitment
- **Spatial Relations:** Natural language relations to nearby entities (complements absolute coordinates)
- **Strict Serialization:** Individual listing (chair_01, chair_02) not grouped — ensures granular counting

**Key insight vs. Cognitive Map (Yang et al. 2025b):** Instead of a coarse 10x10 grid with fixed object categories, TRACE uses continuous metric coordinates, open-vocabulary entities, and rich per-entity attributes.

### 3.4 Inference Mechanism

**Single-pass generation:** The model first generates the TRACE representation G, then uses it as a "spatial cache" in the same context window to answer the question. Formalized as:

```
A_hat, G_hat = argmax P(A | G, V, Q) * P(G | V, Q)
                        [Reasoning Parser]  [Spatial Descriptor]
```

The model can compute Euclidean distances between object coordinates in E or traverse trajectory nodes in T to answer spatial questions.

---

## 4. Experimental Setup

### 4.1 Benchmarks

| Benchmark | Type | Size | Key Properties |
|-----------|------|------|----------------|
| **VSI-Bench** | Egocentric indoor video | 5,130 QA, 288 videos | 8 tasks: configurational + measurement |
| **OST-Bench** | Online spatio-temporal | 1,386 scenes, 10,165 QA | Multi-round dialogue, incremental observation |

For OST-Bench, they use a reproducible 200-scene subset (1,396 QA pairs).

### 4.2 Models Tested

| Model | Type | Role |
|-------|------|------|
| **Gemini 3 Pro** | Proprietary | Primary backbone |
| **Qwen2.5-VL-72B** | Open-source | VSI-Bench open-source |
| **MiMo-VL-7B-SFT** | Open-source (compact) | Both benchmarks |

Additional models in supplementary: o3, GLM-4.5V.

### 4.3 Baselines Compared

| Method | Description |
|--------|-------------|
| Direct | No explicit reasoning |
| CoT | Step-by-step natural language reasoning |
| ToT | 3 reasoning branches, evaluate, select best |
| LtM | Decompose into subproblems, solve sequentially |
| CM | Cognitive Map — 10x10 grid with object positions |
| **TRACE (Ours)** | Structured allocentric representation |

All methods share the same base system prompt and differ only in the user prompt's reasoning protocol.

---

## 5. Key Results

### 5.1 VSI-Bench (Table 1)

| Method | Gemini 3 Pro | Qwen2.5-VL-72B | MiMo-VL-7B |
|--------|-------------|----------------|-------------|
| Direct | 52.61 | 36.28 | 36.02 |
| CoT | 53.65 | 29.78 | 34.27 |
| ToT | 58.88 | 38.06 | 29.45 |
| LtM | 59.52 | 38.01 | 30.44 |
| CM | 59.72 | 35.47 | 54.26 |
| **TRACE** | **60.15** | **39.38** | **41.42** |

**Improvements over Direct:** +7.54% (Gemini), +3.10% (Qwen), +5.40% (MiMo).

Notable patterns:
- TRACE is the **only method that consistently improves across all three model families**
- CoT sometimes *hurts* performance (especially Qwen: 36.28 -> 29.78)
- CM is strong on MiMo but inconsistent across models
- TRACE achieves best or near-best on nearly every individual task

### 5.2 OST-Bench (Table 2)

| Method | Gemini 3 Pro | MiMo-VL-7B |
|--------|-------------|-------------|
| Direct | 59.22 | 62.65 |
| CoT | 59.26 | 61.69 |
| CM | 59.04 | 64.00 |
| **TRACE** | **60.42** | **65.04** |

Gains are more modest on OST (+1.2% Gemini, +2.4% MiMo). This is because OST is multi-turn and TRACE is currently a static representation — dynamic egocentric agent state tracking suffers.

### 5.3 Prediction Settings (Table 3)

| Setting | Gemini 3 Pro | Qwen2.5-VL-72B |
|---------|-------------|----------------|
| Video Direct | 52.61 | 37.58 |
| **One-Stage** | **60.15** | **38.92** |
| Two-Stage | 58.52 | 32.85 |
| Text-Only | 52.27 | 31.11 |

**Critical finding:** One-stage > Two-stage. The *reasoning process* of generating TRACE is as important as the representation itself. When you pre-generate TRACE and feed it as context, performance drops. This suggests the structured generation acts as a form of "thinking" that primes the model's spatial reasoning.

**Text-only matches video direct for Gemini.** TRACE captures enough spatial information that the LLM can answer without seeing the video again — a strong validation of the representation's informativeness.

### 5.4 Comparison with Other Spatial Representations (Table 4)

Using Qwen2.5-VL-72B, text-only inference:

| Representation | Avg |
|----------------|-----|
| Cognitive Map | 21.41 |
| Spatial Caption | 27.58 |
| **TRACE** | **31.11** |
| TRACE w/o Trajectory | 29.19 |
| TRACE w/o Entity Registry | 25.87 |

TRACE outperforms Cognitive Map by +9.7% and Spatial Caption by +3.53%.

### 5.5 Ablation Studies

- **Removing Trajectory:** -1.92% overall. Mainly affects distance and order tasks.
- **Removing Entity Registry:** -5.24% overall. Large drop on object-related tasks.
- Removing trajectory *improves* room size and relative direction — current MLLMs can't reliably estimate camera motion, which confuses alignment-based reasoning.

---

## 6. Decompositional Analysis (Fig. 4)

The paper decomposes 3D spatial understanding into two roles:
- **Spatial Descriptor:** Generates TRACE from video (3D visual perception)
- **Reasoning Parser:** Uses TRACE to answer the question (language-based spatial reasoning)

Key findings when mixing models across roles:
- Replacing either Descriptor or Parser with weaker models degrades performance significantly
- Qwen 72B and 7B have **comparable 3D visual perception** (similar Descriptor performance)
- But 72B has **much stronger reasoning capacity** (large gap when used as Parser)
- **Bottleneck identification:** For current open-source models, spatial *reasoning* is more limiting than spatial *perception*

---

## 7. Critical Analysis

### 7.1 Strengths

1. **Pure prompting approach** — no fine-tuning, no additional modules, applicable to any off-the-shelf MLLM. This is the first prompting method specifically designed for structured spatial reasoning.

2. **Cognitively grounded design** — draws explicitly from allocentric representation theory (Marr, Klatzky). The meta-context/trajectory/entity decomposition maps to known cognitive constructs.

3. **Consistent cross-model gains** — unlike CoT/ToT/LtM which are inconsistent or harmful across models, TRACE helps every tested backbone. This suggests the approach addresses a fundamental capability gap.

4. **One-stage > Two-stage finding** — a non-obvious insight that the act of generating structured spatial descriptions is itself beneficial, beyond just having the representation available. This has implications for how we think about CoT-style methods for spatial tasks.

5. **Thorough experimental design** — fair comparison with 5 baseline prompting methods using shared scaffolding, ablations, decompositional analysis, cross-environment generalization, and token efficiency analysis.

6. **Practical insight about bottlenecks** — the Descriptor/Parser decomposition reveals that open-source models are more limited by reasoning than perception, directing future research efforts.

### 7.2 Limitations & Questions

1. **Static representation problem:**
   - TRACE is a single global allocentric representation generated once
   - Doesn't handle dynamic scenes or multi-turn updates well (acknowledged drop on OST agent state tasks)
   - Real embodied agents need incrementally updated representations

2. **Self-generated representations:**
   - The MLLM generates TRACE itself — so the representation quality is bounded by the model's own perception
   - Authors note that specialized visual expert models could enhance accuracy
   - No ground-truth comparison of TRACE quality vs. actual 3D scene annotations

3. **Metric estimation quality unknown:**
   - Entities get estimated [x,y] coordinates in meters, but how accurate are these?
   - No systematic evaluation of coordinate estimation error
   - The paper argues the *act* of estimating forces geometric commitment, even if estimates are noisy

4. **Limited model coverage:**
   - Only 3 primary models (Gemini 3 Pro, Qwen2.5-VL-72B, MiMo-VL-7B)
   - No Claude, no GPT-4o/o3 as primary (o3 only in supplementary)
   - Qwen excluded from OST due to multi-turn limitations

5. **Prompt sensitivity not explored:**
   - TRACE prompts are highly structured with specific formatting requirements
   - How sensitive is performance to prompt wording variations?
   - Would the approach work with less rigid schema?

6. **Token cost:**
   - TRACE generates substantial intermediate text (meta-context + trajectory + entity registry)
   - Token efficiency varies significantly across models
   - For latency-sensitive embodied applications, this overhead matters

7. **No comparison with non-prompting approaches:**
   - Fine-tuned spatial models (Mm-spatial, SAT) not compared
   - Geometric module approaches (SpatialRGPT, LLaVA-3D) not compared
   - Hard to assess where TRACE stands relative to the entire landscape

### 7.3 Relationship to "Thinking in Space" (Yang et al. 2025b)

TRACE directly builds on the Thinking in Space paper's findings:

| Thinking in Space Finding | TRACE Response |
|--------------------------|----------------|
| CoT/ToT/SC hurts spatial tasks | TRACE provides domain-specific structured reasoning instead of generic linguistic reasoning |
| Cognitive Map (10x10 grid) helps distance estimation | TRACE extends this: continuous coordinates, open-vocabulary entities, trajectory, richer attributes |
| Egocentric-allocentric transformation is the bottleneck (~71% of errors) | TRACE explicitly constructs an allocentric representation, externalizing this transformation |
| Models build local but not global maps | TRACE's room-aligned coordinate system + trajectory aims to provide global context |
| Spatial reasoning > linguistic reasoning as bottleneck | TRACE's design separates spatial grounding from linguistic reasoning |

**TRACE can be seen as the "next step" after Thinking in Space** — taking the diagnostic findings and translating them into an actionable prompting strategy.

---

## 8. Relevance to Our Research

### Direct connections to our project direction

Our research direction: **build an agent layer on top of VLMs that externalizes spatial reasoning, using VLMs only for perception while handling ego-allo transformations externally.**

TRACE is highly relevant because:

1. **Validates the externalization hypothesis:** TRACE shows that making spatial reasoning *explicit* via structured text consistently improves performance. This supports our thesis that VLMs shouldn't do spatial reasoning internally.

2. **But TRACE still relies on the VLM for everything:** The VLM generates both the spatial representation AND reasons over it. Our approach would go further — use the VLM only for perception (object detection, appearance description) and handle coordinate estimation, trajectory tracking, and spatial reasoning with external modules.

3. **One-stage > Two-stage is a key insight for our architecture:** If the reasoning *process* matters, not just the representation, then our agent can't simply pre-compute a scene graph and hand it to an LLM. The agent needs to interleave perception and reasoning.

4. **Entity Registry design is a useful reference:** The structured per-entity schema (temporal stamping, visual signature, metric estimation, spatial relations, strict serialization) is a practical template for our agent's object representation.

5. **Metric estimation is the weak link to improve:** TRACE relies on the VLM to estimate metric positions. Our agent could replace this with depth estimation models, visual odometry, or other geometric methods for more accurate coordinates.

6. **Trajectory generation could be replaced with visual odometry:** Instead of asking the VLM to estimate camera poses (which the paper shows is unreliable — removing trajectory sometimes helps), our agent could use SLAM or visual odometry for ground-truth trajectory.

### How TRACE informs our agent design

| TRACE Component | Our Agent Equivalent |
|----------------|---------------------|
| Meta Context (room topology, coordinate system) | Scene initialization module with depth/geometry estimation |
| Camera Trajectory (VLM-estimated poses) | Visual odometry / SLAM module (more accurate) |
| Entity Registry (VLM-estimated positions) | Object detection + depth-based localization (more accurate) |
| Reasoning Parser (VLM reasons over TRACE) | LLM reasoning over externally-computed spatial representation |
| Single-pass generation | Iterative perception-reasoning loop |

---

## 9. Quick Reference

| | |
|---|---|
| **Title** | Unleashing Spatial Reasoning in MLLMs via Textual Representation Guided Reasoning |
| **Method** | TRACE: Textual Representation of Allocentric Context from Egocentric Video |
| **Authors** | Jiacheng Hua, Yishu Yin, Yuhang Wu, Tai Wang, Yifei Huang, Miao Liu |
| **Affiliations** | Tsinghua, Shanghai AI Lab, University of Tokyo |
| **Date** | March 2026 (preprint) |
| **Approach** | Prompting-only (no fine-tuning, no extra modules) |
| **Components** | Meta Context + Camera Trajectory + Entity Registry |
| **Benchmarks** | VSI-Bench, OST-Bench |
| **Best result** | +7.54% over Direct on VSI-Bench (Gemini 3 Pro) |
| **Key insight** | One-stage generation of structured allocentric representation > two-stage or generic CoT |
| **Limitation** | Static representation; VLM generates its own spatial estimates |
