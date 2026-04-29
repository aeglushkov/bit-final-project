# "Embodied VideoAgent" — Paper Analysis

## 1. Paper Summary

**"Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding"** (Fan, Ma, Su, Guo, Wu, Chen, Li; BIGAI / USTC / Tsinghua / PKU; arXiv 2501.00358v2, 9 Jan 2025) extends the VideoAgent line of tool-using multimodal agents from long-form RGB-only video to **dynamic, embodied 3D scenes**. The core contributions are: (1) a **persistent object memory** jointly built from egocentric video and embodied sensor input (depth maps + camera 6D poses), with each entry carrying ID, state, related-object relations, 3D bounding box, and CLIP visual + context features; (2) a **VLM-based memory update** mechanism that detects actions in the stream, associates them with the right memory entries via visual prompting on rendered 3D bounding boxes, and rewrites the entries in place; (3) a downstream LLM-driven tool-use agent that queries this memory and additionally executes embodied action primitives. Embodied VideoAgent beats end-to-end MLLMs and the original VideoAgent across three benchmarks for dynamic scene understanding — Ego4D-VQ3D (+4.9 Succ%), OpenEQA subset (+5.8 ALL on hard subset; +10.7 over VideoAgent at the same backbone), and EnvQA (+11.7 average) — with the largest gain on EnvQA Events questions (5.54 → 25.91), where the VLM-based action-to-object association is the dominant factor. The authors also use the agent to (a) bootstrap synthetic user-assistant interaction data in AI-Habitat / HSSD and (b) drive a Franka robot through an occlusion-robust pick-and-place demo.

---

## 2. What Problem Does This Paper Solve?

Two complementary failure modes motivate the paper:

1. **End-to-end MLLMs do not scale to long egocentric video + dynamic 3D scenes.** Sophisticated MLLMs trained for long-form video work for static-scene description but their compute and accuracy degrade quickly when the scene is volatile (objects get moved, opened, picked) and when fine spatial-temporal dependencies must be resolved. Capability is bottlenecked by context compression, not by reasoning per se.
2. **Existing tool-using video agents (VideoAgent, et al.) build memory from RGB only.** Their memory is a textual + frame-feature index; it has no native 3D structure, no notion of object identity across time under occlusion, and no mechanism to update memory when actions change scene state. The authors' early ablation in Section 3 shows that simply pointing VideoAgent at dynamic-scene benchmarks underperforms — the failure is in the *memory*, not in the reasoning loop.

The thesis is that an embodied agent for dynamic 3D scenes needs (a) a **persistent**, **3D-grounded**, **object-centric** memory that fuses RGB with depth and pose, and (b) an **active update** mechanism that re-runs perception when actions are observed.

---

## 3. Method

### 3.1 System Architecture

The pipeline (Figure 2) ingests egocentric RGB + per-frame depth + per-frame 6D pose, and produces and maintains:

```
Egocentric RGB + Depth + Pose
        │
        ▼
   ┌────────────────────────┐
   │ Per-frame perception   │   YOLO-World (open-vocab)  +  SAM-2 masks
   │   (every frame)        │   2D→3D lifting via depth + camera pose
   └─────────────┬──────────┘
                 ▼
   ┌────────────────────────────────────────────────────────────┐
   │ Persistent Object Memory  M_O                              │
   │   per object: ID · STATE · RO · 3D Bbox · OBJ Feat · CTX Feat │
   └─────────────┬──────────────────────────────────────────────┘
                 ▼
   ┌────────────────────────┐
   │ Action stream (every 2s)│  LaViLa →  "#C C catches the can"
   │ + VLM-based update     │  ─►  visual prompting on rendered 3D bbox
   │                        │  ─►  rewrite STATE / OBJ Feat / 3D Bbox
   └─────────────┬──────────┘
                 ▼
   ┌────────────────────────────────────────────────────────────┐
   │ History buffers: ActionBuffer (T, action, object, frame)   │
   │                  VisibleObjectBuffer (T, object, 3D bbox)  │
   └─────────────┬──────────────────────────────────────────────┘
                 ▼
   ┌────────────────────────┐
   │ LLM Agent (ReAct loop) │   Tools: query_db, temporal_loc,
   │   GPT-4o / InternVL2-8B│          spatial_loc, vqa
   │                        │   Embodied primitives: chat, search,
   │                        │          goto, open, close, pick, place
   └────────────────────────┘
```

The original VideoAgent's temporal memory `M_T` (segment-level captions + features) is also retained — `M_O` and the history buffers are net additions.

### 3.2 Persistent Object Memory M_O

**Entry schema (Appendix A):**

| Field | Description |
|---|---|
| `ID` | unique object id with detected category |
| `STATE` | one of `normal`, `open`, `close`, `in-hand` (updated by VLM) |
| `RO` (Related Objects) | list of `(relation, target_id)` from `{on, uphold, in, contain}`, derived from 3D bbox geometry |
| `3D Bbox` | axis-aligned bbox in world coords; moving-average update |
| `OBJ Feat` | CLIP feature of the cropped object image; moving-average update |
| `CTX Feat` | CLIP feature of the frame containing the object; moving-average update |

`OBJ Feat` is used for cross-frame visual matching during re-identification. `CTX Feat` is *not* used during re-ID — it exists so that downstream `query_db` can do environment-conditioned retrieval ("the cup in the kitchen" vs "the cup at the desk").

**Construction pipeline (Algorithm 3):** for each frame `(RGB^t, Depth^t, Pose^t)`:

1. Detect 2D objects with **YOLO-World** (open-vocabulary).
2. Get masks with **SAM-2**, lift each detection to a 3D bbox via depth + extrinsics; filter foreground/background outlier points by depth ordering (drop top/bottom 10% along ray).
3. Compute `OBJ Feat` (CLIP) and a **DINOv2** feature for re-ID (DINOv2 is used in re-ID only, not stored).
4. Re-identify each detection against memory via two algorithms — static-object re-ID (Algorithm 1) and dynamic-object re-ID (Algorithm 2). Match → moving-average merge; no match → new entry.
5. Recompute `RO` from current bboxes (purely geometric: altitude + contact + horizontal-surface check yields `on/uphold/in/contain`).
6. Run VLM-based update for any newly-recognized actions (Section 3.3 below).

**Static vs dynamic split (Appendix C.3).** Before re-identifying a new detection, existing entries are split into:

- **Static** `S`: object's stored 3D bbox is currently in the camera frustum *and* visible (no depth occlusion) *and* the cropped patch at that screen location has visual similarity to `OBJ Feat` ≥ 0.45.
- **Dynamic** `D`: visible-but-different-looking → "object isn't where it should be" → it has been moved.

This split is important because the matching criteria differ: static re-ID can rely on **spatial overlap** (Spatial_IoU > 0.2 or Spatial_MaxIoS > 0.2 with same category), while dynamic re-ID must rely on **visual similarity** because the bbox has moved (Visual > 0.45 *and* Spatial_VolSim > 0.7).

**Moving-average windows differ by class:** static merges use window 10 (slow drift), dynamic merges use window 2 (catch up to fast motion).

**Visual similarity score (from VideoAgent):** `0.15·CLIP + 0.85·DINOv2`. Spatial scores: `IoU = V_inter / V_union`, `MaxIoS = max(V_inter/V_1, V_inter/V_2)`, `VolSim = min(V_1,V_2)/max(V_1,V_2)`.

### 3.3 VLM-based Memory Update

This is the paper's most distinctive contribution. The pipeline is run every 2 s on each LaViLa-extracted action ("#C C catches the can"):

1. **Object extraction.** Prompt GPT-4o with the action text; extract candidate object categories (e.g. `bottle`, `fridge`).
2. **Candidate retrieval.** Match those categories against detections in the current frame to get a candidate set in `M_O`.
3. **Visual prompting (Figure 4).** For each candidate, render its 3D bbox onto the current frame and ask the VLM: *"Is the object in the bounding box the target object of action 'C catches the can'? Only output Yes or No."*
4. **State update.** For matches, log the action in `ActionBuffer` and update `STATE` on the matched entry (e.g. `→ in-hand` for `catches`, `→ open` for `opens`).

This re-uses the set-of-mark / 3D-bbox-as-visual-prompt idea from prior work but applies it to *memory association*, not perception per se. The benefit is large on event/order/state questions (see EnvQA results below).

When the agent serves as an **active planner** in Habitat (not as a passive observer of an egocentric video), the action is already known to the LLM — no LaViLa is needed; instead the VLM acts as an **action validator** confirming success and refreshing `STATE`.

### 3.4 Tools and Embodied Action Primitives

**Perception tools (Appendix F)** consumed by the agent in a LangChain-style ReAct loop:

| Tool | Behavior |
|---|---|
| `query_db` | SQL-style + similarity retrieval over the object DB. Sub-modes: `retrieve_objects_by_appearance` (text↔OBJ Feat) and `retrieve_objects_by_environment` (text↔CTX Feat). Returns up to 10 candidates. |
| `temporal_loc` | Top-5 frame IDs by text↔frame-feature similarity over `M_T` (inherited from VideoAgent). |
| `spatial_loc` | Top-3 3D positions: average centers of object clusters whose `CTX Feat` matches the description (used for "bedroom", "kitchen"). |
| `vqa` | InternVL2 / GPT-4o on a frame, optionally with a 3D bbox overlay as visual prompt. |

**Embodied action primitives (Appendix G)** for Habitat experiments: `chat`, `search` (Frontier-Based Exploration), `goto` (A* over navmesh), `open`, `close`, `pick`, `place`.

The system prompt (Appendix F) is a vanilla ReAct format (Thought / Action / Action Input / Observation) with two non-trivial guideline lines:

- *Prioritize* `retrieve_objects_by_appearance`, `retrieve_objects_by_environment`, and `frame_localization` over raw `query_db`.
- For **"where" questions**, interpret them as spatial location queries via `vqa` *on a localized frame*, not as identity queries.

### 3.5 Two-Agent Synthetic Data Pipeline

A separate use case (Section 2.3, Appendix G.1): use Embodied VideoAgent as the assistant agent, with another LLM playing the user role. The user is given a *randomly trimmed* scene graph (curiosity prompt) and asked to invent tasks; the assistant fulfills them in 118 HSSD scenes × 20 layouts. The output is episodes of natural-language user requests + tool-use traces — usable as training data for embodied foundation models.

---

## 4. Camera Poses and Sensor Assumptions

The paper is careful (Section 2.2 "Note on camera poses") about the fact that "egocentric video + 6D pose + depth" looks like a strong sensor assumption. The actual experimental setup:

| Benchmark | Pose source | Depth source |
|---|---|---|
| Ego4D-VQ3D | **COLMAP** estimated | (provided) |
| EnvQA | **DUSt3R** estimated | (provided) |
| OpenEQA | Habitat ground-truth | Habitat ground-truth |

Appendix Table 4 shows that swapping ground-truth Habitat poses for DUSt3R-estimated ones costs only ~1.2 ALL on OpenEQA (41.2 → 40.0). The authors argue this is robustness evidence: their tool/memory redundancy bypasses individually-flawed memory entries.

**Caveat:** all three benchmarks are essentially indoor / room-scale. Outdoor or very-long-trajectory pose estimation has different failure modes (drift, scale ambiguity) that this experiment doesn't probe.

---

## 5. Experiments

### 5.1 Ego4D-VQ3D — 3D object localization in dynamic scenes (Table 1)

| Method | Succ% ↑ | Succ*% ↑ | L2 ↓ | QwP% ↑ |
|---|---:|---:|---:|---:|
| EgoLoc [31] | 80.49 | **98.14** | **1.45** | 82.32 |
| Ego4D* [11] | 73.78 | 91.45 | 2.05 | 80.49 |
| Ego4D [11] | 1.22 | 30.77 | 5.98 | 1.83 |
| **E-VideoAgent (text)** | 53.05 | 94.57 | 2.00 | 56.10 |
| **E-VideoAgent (image)** | **85.37** | 92.72 | 1.86 | **92.07** |

`Succ*%` is the success rate restricted to questions the model chose to answer; `L2` is mean distance error on those answered queries; `QwP%` is the answer rate. Two findings:

1. **Open-vocabulary detection lifts QwP%** — YOLO-World gives Embodied VideoAgent a much wider object inventory than EgoLoc's closed-set Ego4D detector, so it tries to answer 92.07% of queries vs EgoLoc's 82.32%.
2. **Visual re-ID is critical in dynamic scenes** — `E-VideoAgent (image)` retrieves objects by visual score against the up-to-date memory; `E-VideoAgent (text)` only uses category. The 32-point gap (53.05 → 85.37) is the value of visual similarity in re-ID.

### 5.2 OpenEQA — embodied question answering (Table 2)

Full validation set (top half) and a 1/5 hard subset (bottom half). Tool stack: GPT-4o LLM + InternVL2-8B or GPT-4o VLM.

| | ScanNet | HM3D | ALL |
|---|---:|---:|---:|
| **Full set** | | | |
| GPT-4 w/ LLaVA-1.5 | 45.4 | 40.0 | 43.6 |
| GPT-4 w/ Concept Graphs | 37.8 | 34.0 | 36.5 |
| Video-LLaVA | 41.5 | 34.6 | 39.2 |
| LLaMA-VID | 33.4 | 34.0 | 33.6 |
| **OpenEQA subset (harder)** | | | |
| Video-LLaVA | 32.9 | 27.8 | 30.6 |
| LLaMA-VID | 31.2 | 28.0 | 29.4 |
| VideoAgent | 37.6 | 34.6 | 36.3 |
| **E-VideoAgent (InternVL2-8B)** | 39.7 | 43.0 | 41.2 |
| **E-VideoAgent (GPT-4o)** | **46.0** | **48.2** | **47.0** |

**Takeaways:**

1. **Agentic > end-to-end on the hard subset.** All three agentic methods (VideoAgent, E-VA-InternVL2, E-VA-GPT-4o) beat both Video-LLaVA and LLaMA-VID, vindicating the multi-step retrieval-then-reason pattern.
2. **Embodied VideoAgent (GPT-4o) beats VideoAgent by +10.7 ALL** with the same backbone — direct evidence that the persistent object memory + VLM update adds value on top of VideoAgent's text-only memory.
3. **Temporal_loc + vqa beats GPT-4 + scene-graph (CG)** on the full set (47.0 vs 36.5 on subset; +10.5). The authors note GPT-4 + frame captions (LLaVA) does well at 43.6 on the full set, suggesting that for many OpenEQA questions, *frame captions are already enough* — Embodied VideoAgent's gain comes mostly on relational/state questions where memory matters.

The abstract reports +5.8% on OpenEQA — that's the gain over the **best baseline** (GPT-4 w/ LLaVA-1.5 at 43.6 → 47.0 with Embodied VideoAgent).

### 5.3 EnvQA — open-ended QA over embodied interactions (Table 3)

EnvQA tests three question types over simulated embodied trajectories. 200 questions per type.

| Method | Events | Orders | States |
|---|---:|---:|---:|
| Video-LLaVA | 10.19 | 39.00 | 18.50 |
| LLaMA-VID | 9.98 | 54.00 | 5.50 |
| VideoAgent | 5.54 | 65.5 | 12.5 |
| **Embodied VideoAgent** | **25.91** | **68.00** | **35.50** |

This is the most consequential table:

1. **Events: 5.54 → 25.91 (+20.37).** The action buffer + VLM-based action↔object association is the only thing in the pipeline that can reliably answer "what happened when X did Y?" questions. End-to-end models lose this in compression.
2. **States: 12.5 → 35.50 (+23.0).** State questions ("where was the book moved?") benefit from `RO` (related objects via 3D bbox) — once relations are tracked, the final receptacle is recoverable from the memory entry.
3. **Orders is already saturated** (54–68% across baselines) — temporal localization helps but the headroom is smaller.

Notably, the abstract claim of "+11.7% on EnvQA" is the average across the three columns over the *VideoAgent* baseline, which is the natural agentic counterfactual.

### 5.4 Real-World Robot Demo (Figure 7)

A Franka arm is told to pick up an apple. Mid-task, a box is placed in front of the apple, occluding it. With persistent memory, the agent recalls the apple's 3D position from `M_O`, moves the box, and completes the pick. This is a one-trial qualitative demo, not a benchmarked result, but it concretely illustrates what "persistent object memory" buys in a real manipulation loop.

### 5.5 Ablation: Noisy Camera Poses (Appendix Table 4)

| | ScanNet | HM3D | ALL |
|---|---:|---:|---:|
| Video-LLaVA | 32.9 | 27.8 | 30.6 |
| VideoAgent | 37.6 | 34.6 | 36.3 |
| **E-VideoAgent (GT poses)** | **39.7** | **43.0** | **41.2** |
| E-VideoAgent (DUSt3R noisy poses) | 38.2 | 42.2 | 40.0 |

A 1.2-point drop ALL when swapping in DUSt3R-estimated poses is the only quantitative robustness study on sensor noise. The authors note this opens the door to **RGB-only** deployment (depth + pose from monocular reconstruction), which is the practical setting for a robot in the wild.

---

## 6. Critical Analysis

### 6.1 Strengths

1. **Clean separation of concerns.** Perception (YOLO-World, SAM-2, CLIP, DINOv2) is geometric/visual. Spatial reasoning (3D bboxes, IoU, containment) is programmatic. The VLM is *only* asked perceptual questions — "is this the target?", "what color?". This is exactly the agent-on-top-of-VLM architecture the project research direction hypothesizes.
2. **Memory is updated, not just constructed.** Most prior 3D-scene-graph work (ConceptGraphs, OpenScene) treats the scene as static and rebuilds memory from scratch. The VLM-based update is what makes this work for *dynamic* scenes — and the EnvQA Events delta (5.54 → 25.91) shows the update mechanism alone explains a large fraction of the headline gains.
3. **Single coherent system across three benchmarks** (Ego4D-VQ3D, OpenEQA, EnvQA) without per-benchmark re-engineering. Same memory, same tools, same prompts.
4. **Robust to estimated poses.** The DUSt3R ablation is small (1.2 ALL drop) but pointed in the right direction — it argues the pipeline is not narrowly dependent on Habitat-quality sensor data.
5. **Tool-orthogonal scaling.** Switching VLM (InternVL2-8B → GPT-4o) on OpenEQA gives +5.8 ALL (41.2 → 47.0) without touching the memory — perception quality is the bottleneck, not the memory representation.

### 6.2 Limitations and Open Questions

1. **No head-to-head agentic baselines on Ego4D-VQ3D or EnvQA.** Comparison is to specialized methods (EgoLoc) or to VideoAgent only. No comparison to ConceptGraphs / Chat-Scene / 3D-LLM in the agentic regime. So we know "Embodied VideoAgent > VideoAgent" but not "Embodied VideoAgent > strongest 3D-aware agent baseline".
2. **Hand-tuned thresholds.** Dynamic re-ID uses `Visual > 0.45 ∧ VolSim > 0.7`; static re-ID uses `IoU > 0.2 ∨ MaxIoS > 0.2`. No ablation on these. Sensitivity to scene density and detector noise is unclear.
3. **OpenEQA: GPT-4 + frame captions is competitive on the full set (43.6).** That suggests for many EQA questions, captions plus a strong LLM are already enough — Embodied VideoAgent's headline win is on the harder subset where memory and relations matter. The general claim that 3D memory is required for embodied QA is over-stated relative to the data.
4. **Memory is object-centric only.** No room-level / topological structure beyond `RO`. `spatial_loc` falls back to averaging detection centers whose `CTX Feat` matches "bedroom" — a soft heuristic, not a real spatial graph. Ego↔allo transformations beyond per-object world coordinates are not first-class.
5. **VLM update has no quantitative ablation.** Section 3.2 claims VLM-based update is "a key role" in EnvQA Events; the ablation is implicit (Embodied VideoAgent vs VideoAgent), not isolated. We don't know how much of the Events gain is VLM-update vs how much is just having a 3D memory at all.
6. **Re-ID failure modes not characterized.** What happens when two visually-similar objects (two identical cans) sit on the same shelf? Nothing in Algorithms 1/2 disambiguates them beyond category + position. This is exactly the case where action-based update should help (the can the agent picked up *is* the one in their hand) but failure stats are not reported.
7. **Reliance on closed-source perception.** GPT-4o is the default VLM and the InternVL2-8B variant lags by ~6 ALL on OpenEQA. Memory + InternVL2 (41.2) is *roughly* on par with VideoAgent + GPT-4o (36.3) — close but not above; the framework still benefits significantly from a strong VLM.
8. **No code or checkpoints in the literature folder** at time of writing — only PDF. Project page promises a public release.

### 6.3 Mapping to Project Research Hypothesis

The project's working thesis is "VLMs excel at perception, fail at spatial reasoning; build an agent layer that externalizes spatial reasoning." Embodied VideoAgent is essentially a worked example of this:

| Aspect | Inside the VLM | Outside the VLM |
|---|---|---|
| Object detection | ✓ (YOLO-World is a vision-only detector, but visual) | |
| 2D→3D lifting | | ✓ deterministic (depth + pose) |
| Object identity (re-ID) | partly (CLIP/DINOv2 features used) | mostly: spatial IoU + thresholds |
| Containment / on-top relations | | ✓ purely geometric on bboxes |
| Ego↔allo transform | | ✓ camera extrinsics |
| Action↔object association | ✓ "is the target in this bbox?" | scaffolded by 3D bbox visual prompt |
| Multi-hop reasoning | | ✓ ReAct loop in LLM |
| State tracking over time | | ✓ memory + action buffer |

The papers most directly on this same axis from `literature/`:

- **VideoAgent (Fan et al., ECCV 2025):** the predecessor; same agent loop, no 3D memory.
- **LongVideoAgent / LVAgent / VideoSeek:** sibling agentic-video work but on RGB-only long-form video.
- **Thinking-in-Space (Yang et al., CVPR 2025):** the benchmark side — isolates VLM perception vs reasoning failures. Embodied VideoAgent does *not* report VSI-Bench numbers but its memory architecture is the kind of external scaffolding TiS argues for.
- **SAVVY / Feature4X / LIRA:** sibling 3D-scene representation work in the same literature folder, none yet analyzed; worth contrasting on memory granularity.

---

## 7. Concrete Ideas Sparked

1. **Reproduce Table 4's noisy-pose ablation on VSI-Bench.** Plug Embodied VideoAgent's memory pipeline (DUSt3R poses + open-vocab detection) into a VSI-Bench-style task. Hypothesis: most VSI-Bench errors that TiS attributes to "egocentric ↔ allocentric failure" disappear once the agent has a world-frame memory.
2. **Isolate the VLM-update contribution** with a clean ablation: same memory, same agent, but disable VLM update (no STATE tracking, no action buffer). Quantify how much of EnvQA Events / OpenEQA State gains come from the update mechanism specifically.
3. **3D-bbox-as-visual-prompt as a perception primitive.** Section 3.3's pattern — render 3D bbox onto frame, ask VLM yes/no — is reusable as a general spatial-grounding probe. It could replace tool calls of the form "is X to the left of Y?" by rendering both bboxes and asking the VLM directly.
4. **Threshold sensitivity sweep.** Run Algorithms 1 & 2 with `Visual ∈ [0.3, 0.6]` and `IoU ∈ [0.1, 0.4]`. Report a heatmap of re-ID precision/recall. This is a low-cost contribution to the open-source release.
5. **Topological extension.** Embodied VideoAgent's object memory has `RO` for fine-grained relations but no room/region structure. Add a coarse Voronoi-style room graph indexed by frame `CTX Feat` clusters; test on OpenEQA "which room"-style questions.
6. **Connect to robot manipulation benchmark.** The Franka demo (Figure 7) is qualitative. A small reproducible ablation on a public benchmark (e.g. ManiSkill or Habitat 3) comparing "memory off / static memory / persistent memory with update" would make the manipulation claim quantitative.

---

## 8. Quick Reference

| | |
|---|---|
| **Title** | Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding |
| **Authors** | Yue Fan*, Xiaojian Ma*†, Rongpeng Su, Jun Guo, Rujie Wu, Xi Chen, Qing Li† |
| **Affiliation** | BIGAI · USTC · Tsinghua · PKU |
| **Venue** | arXiv 2501.00358v2 (9 Jan 2025) |
| **Builds on** | VideoAgent [Fan et al., ECCV 2025] — extends RGB-only memory to 3D + dynamic |
| **Memory** | Persistent Object Memory M_O (per-object: ID/STATE/RO/3D Bbox/OBJ Feat/CTX Feat) + temporal memory M_T + ActionBuffer + VisibleObjectBuffer |
| **Perception stack** | YOLO-World (open-vocab detection) · SAM-2 (masks) · CLIP + DINOv2 (visual features) · LaViLa (action annotation) · GPT-4o / InternVL2-8B (VLM) |
| **Agent loop** | LangChain-style ReAct; 4 tools (`query_db`, `temporal_loc`, `spatial_loc`, `vqa`) + 7 embodied primitives (`chat`, `search`, `goto`, `open`, `close`, `pick`, `place`) |
| **Sensor inputs** | RGB + per-frame depth + 6D camera pose (ground-truth, COLMAP, or DUSt3R-estimated) |
| **Benchmarks** | Ego4D-VQ3D · OpenEQA (full + 1/5 subset) · EnvQA |
| **Headline numbers** | Ego4D-VQ3D Succ% **85.37** (+4.9 vs EgoLoc) · OpenEQA subset ALL **47.0** (+10.7 vs VideoAgent same backbone) · EnvQA Events / Orders / States **25.91 / 68.00 / 35.50** |
| **Robustness** | GT poses → DUSt3R noisy poses on OpenEQA: 41.2 → 40.0 ALL (-1.2) |
| **Code** | Project page: embodied-videoagent.github.io ("code and demo will be made public") |
