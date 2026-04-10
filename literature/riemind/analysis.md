# RieMind — Paper Analysis

## 1. Paper Summary

**"RieMind: Geometry-Grounded Spatial Agent for Scene Understanding"** (Ropero et al., Huawei Riemann Lab, March 2026) proposes an agentic framework for 3D indoor spatial understanding that decouples perception from reasoning. Instead of feeding video into a VLM and hoping it can reason spatially, they build a **3D scene graph (3DSG)** from the scene and give an LLM access to geometry-grounded tools that query this graph. The LLM never sees images — it reasons entirely over structured geometric data via tool calls. Evaluated on the static portion of VSI-Bench (4,185 questions across 6 task types), the framework achieves state-of-the-art results, surpassing all proprietary, open-source, and fine-tuned models.

---

## 2. Core Architecture

### 2.1 Perception Layer: 3D Scene Graph (3DSG)

The 3DSG is a hierarchical graph G = (N, E) with four node layers:

| Layer | Node Type | What It Stores |
|-------|-----------|---------------|
| Building | N_B | Entire scene, class attributes |
| Floor | N_F | Groups rooms on the same floor, areas, bounding box geometry |
| Room | N_R | Bounded spatial region, bounding box and derived geometry |
| Object | N_O | Semantic class (from closed set), bounding box dimensions, volume, surface area, orientation, location |

Edges encode hierarchical containment (building→floor→room→object) and inter-object relations (restricted to `{near}` — the agent derives other relations via tools).

**Critical design choice:** In this paper, the 3DSG is built from **ground-truth annotations** (ScanNet, ScanNet++, ARKitScenes), not from a perception pipeline. This isolates the spatial reasoning capability of the LLM and provides an upper bound.

### 2.2 Reasoning Layer: Agentic LLM with Tools

The architecture uses **Model Context Protocol (MCP)** servers to expose tools across four semantic namespaces:

#### Memory Tools (`mem_*`)
- `mem_get_scene_context` — structured summary of the 3DSG (hierarchy, objects, counts)

#### Scene Tools (`sg_*`)
- `sg_describe_scene_graph_schema` — schema description
- `sg_search_nodes_by_name/class/type` — entity search and disambiguation
- `sg_get_node_attributes` — attribute lookup by node ID
- `sg_get_objects_in_rooms/floors/buildings` — scope restriction
- `sg_get_nearest_neighbors_in_room` — proximity search
- `sg_get_spatial_relation` — semantic relation between nodes
- ~15 tools total for graph traversal

#### Geometry Tools (`geom_*`)
- `geom_convex_volume` / `geom_concave_volume` — volume computation
- `geom_dimensions` — bounding box size
- `geom_convex_surface_area` / `geom_concave_surface_area` — surface area
- `geom_convex_footprint_area` / `geom_concave_footprint_area` — floor footprint
- `geom_diagonal_length` — diagonal of bounding box
- `geom_euclid_dist_by_nodes` — Euclidean distance between nodes

#### Location and Orientation Tools (`loc_*`)
- `loc_get_node_position` — 3D position of a node
- `loc_build_frame` — construct a reference frame from position + orientation
- `loc_project_into_frame` — project a point into a reference frame
- `loc_position_look_at_position` — compute orientation vector
- `loc_position_from_interpolation` — interpolate between positions
- `loc_orientation_look_at_position` — orientation at a point looking at another

### 2.3 Tool Design Principles

1. **Minimal geometric primitives** — each tool does one atomic operation (no composite shortcuts)
2. **Explicit grounding** — all geometry/orientation tools operate on **node IDs**, not free-form text; only scene search tools accept text
3. **Determinism** — tool outputs depend solely on 3DSG state, not on the LLM

### 2.4 Agent Prompt Structure

The system prompt has 7 sections:
1. Role definition (spatial reasoning system, must delegate all computation to tools)
2. Available tools list
3. Scene context cache (from `mem_get_scene_context`)
4. Core agent constraints (6 rules: resolve terminology, search before query, no duplicate calls, valid args, one tool per step, maintain reasoning plan)
5. Tool data flow (object name → disambiguate → resolve to class → get node ID → call tools)
6. Tool catalog (reference docs for all 4 namespaces)
7. Answer format (JSON with natural language summary, tool evidence, data dictionary)

### 2.5 Reasoning Example: Relative Direction Question

> "Standing by the bed, facing the trash can, is the mouse to my front-left, front-right, back-left, or back-right?"

The agent executes a 5-step pipeline:
1. **Entity grounding** — `sg_search_nodes_by_name("Bed", "Trash Can", "Mouse")` → get node IDs
2. **Metric grounding** — `loc_get_nodes_position([ID_bed, ID_trash, ID_mouse])` → 3D coordinates
3. **Orientation grounding** — `loc_orientation_look_at_point(pos_bed, pos_trash)` → orientation vector
4. **Frame grounding** — `loc_build_frame(pos_bed, orientation, "ZUpYForwardXRight")` → reference frame
5. **Geometry grounding** — `loc_project_into_frame(frame, pos_mouse)` → projected coordinates → interpret signs (negative X = left, positive Y = forward) → "Front-Left"

---

## 3. Experiments & Results

### 3.1 Setup

- **Benchmark:** VSI-Bench static portion — 4,185 questions across 6 types (excluded route planning and appearance order as they are fundamentally dynamic)
- **3DSG source:** Ground-truth annotations from ARKitScenes, ScanNet, ScanNet++
- **LLMs tested as agents:** Qwen2.5-VL-7B, GPT-4o, GPT-4.1

### 3.2 Impact of Agentic Reasoning (Table 2)

Comparison of base VLM (no tools, no fine-tuning) vs. LLM + agent tools:

| Question Type | Qwen2.5-VL-7B Base → Agent | GPT-4o Base → Agent |
|--------------|---------------------------|---------------------|
| Object Count | 40.9 → **89.7** (+48.8) | 46.2 → **85.1** (+38.9) |
| Absolute Distance | 14.8 → **90.3** (+75.5) | 5.3 → **93.2** (+87.9) |
| Object Size | 43.4 → **93.6** (+50.2) | 43.8 → **96.5** (+52.7) |
| Room Size | 10.7 → **31.9** (+21.2) | 38.2 → **83.5** (+45.3) |
| Relative Distance | 38.6 → **44.5** (+5.9) | 37.0 → **85.6** (+48.6) |
| Relative Direction | **38.5** → 34.7 (-3.8) | 41.3 → **67.5** (+26.2) |
| **Average** | 31.2 → **64.1** (+32.9) | 35.3 → **85.2** (+49.9) |

Key observations:
- **Absolute questions** (count, distance, size) see massive improvements — the 3DSG provides exact geometric truth
- **Relative direction** is the hardest — requires 5-6 tool calls in a chain; Qwen2.5-VL-7B actually *degrades* (-3.8) because the small model hallucinates during long reasoning chains
- **GPT-4o benefits far more** than Qwen2.5-VL-7B — stronger reasoning capability enables better tool use
- The improvement pattern confirms that **reasoning capability, not perception, is the bottleneck** once geometric grounding is provided

### 3.3 Score vs. Reasoning Complexity (Table 3)

For Qwen2.5-VL-7B agent:

| Question Type | # Questions | Score | Avg Tools | Median Tools |
|--------------|-------------|-------|-----------|-------------|
| Object Count | 544 | 89.7 | 1.05 | 1 |
| Absolute Distance | 799 | 90.3 | 2.14 | 2 |
| Object Size | 927 | 93.6 | 2.15 | 2 |
| Room Size | 282 | 31.9 | 2.35 | 2 |
| Relative Distance | 703 | 44.5 | 2.28 | 2 |
| Relative Direction (Easy) | 211 | 46.1 | 4.06 | 4 |
| Relative Direction (Med.) | 360 | 31.7 | 3.83 | 4 |
| Relative Direction (Hard) | 359 | 31.1 | 3.89 | 4 |

Simpler questions (1-2 tools) score 89-93%. Relative direction requiring 4+ tool chains scores 31-46%. **Compositional reasoning depth is the bottleneck for smaller models.**

### 3.4 State-of-the-Art Comparison (Table 4)

| Model | Avg | Obj Count | Abs Dist | Obj Size | Room Size | Rel Dist | Rel Dir |
|-------|-----|-----------|----------|----------|-----------|----------|---------|
| *Best fine-tuned (SpaceMind)* | 73.6 | 73.3 | 61.4 | 77.3 | 74.2 | 67.2 | **88.4** |
| *Best fine-tuned (ViCA)* | 63.5 | 68.8 | 57.0 | 79.2 | 75.1 | 58.5 | 42.6 |
| **RieMind + Qwen2.5-VL-7B** | 64.1 | **89.7** | **90.3** | 93.6 | 31.9 | 44.5 | 34.7 |
| **RieMind + GPT-4o** | 85.2 | 85.1 | 93.2 | 96.5 | **83.5** | 85.6 | 67.5 |
| **RieMind + GPT-4.1** | **89.5** | 86.5 | **94.9** | **97.9** | 77.8 | **92.7** | 87.3 |

Key takeaways:
- RieMind + GPT-4.1 achieves **89.5% average** vs. 73.6% for the best fine-tuned model (SpaceMind) — a **16% improvement**
- On absolute questions (distance, size), the agent dominates because it has direct access to geometric truth
- Relative direction remains the hardest — SpaceMind (88.4%) still beats GPT-4o agent (67.5%), but GPT-4.1 (87.3%) nearly matches it
- The gap from GPT-4o to GPT-4.1 on relative direction (+20 points) shows that **model reasoning capability directly translates to better tool use** for compositional tasks
- RieMind + Qwen2.5-VL-7B (64.1%) already outperforms most fine-tuned models without any training

---

## 4. Critical Analysis

### 4.1 Strengths

1. **Clean separation of perception and reasoning** — by using ground-truth 3DSG, the paper isolates what fraction of spatial understanding failure comes from reasoning vs. perception. This is a valuable experimental control.

2. **Tool design is principled** — minimal primitives, explicit grounding via node IDs, deterministic outputs. The 4-namespace organization (memory, scene, geometry, location/orientation) is clean and extensible.

3. **MCP-based architecture** — using Model Context Protocol for tool serving is a practical, standards-based design that could be deployed with any MCP-compatible LLM.

4. **Strong empirical results** — 16% over best fine-tuned models, 33-50% over base VLMs. These are substantial margins.

5. **Reveals reasoning as the bottleneck** — Qwen2.5-VL-7B degrades on relative direction despite having perfect geometric data, proving that the reasoning chain itself fails, not the data.

6. **No training required** — zero-shot agentic approach outperforms models specifically fine-tuned on spatial QA data.

### 4.2 Limitations & Concerns

1. **Ground-truth 3DSG is the elephant in the room** — the entire framework assumes a perfect scene graph. In practice, building a 3DSG from RGB-D data introduces segmentation errors, incorrect labels, missing objects, wrong bounding boxes. The paper acknowledges this but doesn't quantify how much performance would degrade with a noisy 3DSG.

2. **Cost and latency** — each question requires multiple LLM API calls with tool use. Relative direction questions need 5-6 tool calls. At GPT-4.1 prices, this is substantially more expensive than a single VLM inference. The paper doesn't report cost or latency.

3. **Room size is weak** — Qwen2.5-VL-7B agent scores only 31.9% on room size despite having access to geometric tools. This suggests the tools or the agent prompting may not adequately handle room-level geometry queries.

4. **Static scenes only** — the framework targets static 3D indoor scenes. Route planning, temporal questions, and dynamic environments are explicitly excluded.

5. **Limited LLM coverage** — only 3 LLMs tested as agents (Qwen2.5-VL-7B, GPT-4o, GPT-4.1). Claude, Gemini, open-source models with strong tool use (Llama 3.1) are not tested.

6. **The 3DSG representation may overconstrain** — by restricting inter-object edges to `{near}`, the graph may miss useful spatial relations. The paper argues the agent can derive these from tools, but this adds tool-call overhead.

### 4.3 Comparison to SpatialScore's SpatialAgent

Both RieMind and SpatialAgent (from the SpatialScore benchmark) share the idea of tool-augmented spatial reasoning, but differ fundamentally:

| Aspect | SpatialAgent (SpatialScore) | RieMind |
|--------|---------------------------|---------|
| Scene representation | Per-frame perception (depth estimation, detection) | Persistent 3DSG |
| Perception | Estimated (monocular depth, bounding boxes) | Ground-truth annotations |
| Tools | Frame-level (depth at pixel, bbox lookup) | Graph-level (node positions, volumes, orientations) |
| Coordinate system | Image-space + estimated depth | World-space 3D coordinates |
| Frame management | Explicit frame selection | No frames — entirely text/graph based |
| Ego-allo transform | Not explicitly supported | Dedicated `loc_*` tools for frame construction and projection |

RieMind's approach is structurally more sound for 3D reasoning — persistent world-space representations avoid the per-frame estimation noise that plagues SpatialAgent. However, RieMind requires a pre-built 3DSG, while SpatialAgent works from raw video.

---

## 5. Relevance to Our Research

### 5.1 Direct Connections

This paper is **closely aligned with our research direction** — externalizing spatial reasoning from VLMs:

1. **Validates our hypothesis** — the paper empirically confirms that decoupling perception from reasoning improves spatial understanding (33-50% improvement over base VLMs). This is exactly the idea in our agent architecture proposal.

2. **3DSG as scene representation** — using a structured scene graph with explicit geometry is a concrete instantiation of the "external spatial representation" concept. Our approach could adopt a similar representation.

3. **Tool-based reasoning** — the geometry-grounded toolbox is a direct example of the "VLM for perception, tools for reasoning" split we're investigating.

4. **Reasoning capability matters** — the Qwen2.5-VL-7B vs. GPT-4.1 gap on relative direction (34.7% vs. 87.3%) shows that even with perfect tools and perfect data, the base model's reasoning capability is critical. This informs our choice of backbone LLM.

### 5.2 Gaps We Could Address

1. **Close the perception gap** — RieMind uses ground-truth 3DSGs. Building a practical 3DSG from RGB-D or video using VLM perception is an open problem. Our work could focus on this end-to-end pipeline.

2. **Noisy 3DSG robustness** — how much does agent performance degrade when the 3DSG has errors? Characterizing this degradation curve would be valuable.

3. **Dynamic scenes and temporal reasoning** — RieMind explicitly excludes route planning and temporal questions. An agent that handles both static and dynamic spatial reasoning would be more general.

4. **Alternative to MCP** — while MCP is clean, it introduces infrastructure complexity. A simpler tool-calling interface might be more practical for research.

---

## 6. Quick Reference

| | |
|---|---|
| **Title** | RieMind: Geometry-Grounded Spatial Agent for Scene Understanding |
| **Authors** | Fernando Ropero, Erkin Turkoz, Daniel Matos, Junqing Du, Antonio Ruiz, Yanfeng Zhang, Lu Liu, Mingwei Sun, Yongliang Wang |
| **Affiliation** | Riemann Lab, Huawei Technologies |
| **Date** | March 2026 |
| **arXiv** | 2603.15386 |
| **Benchmark** | VSI-Bench (static portion, 4,185 questions, 6 task types) |
| **Key result** | 89.5% avg with GPT-4.1 agent (vs. 73.6% best fine-tuned, vs. 35.3% base GPT-4o) |
| **Core idea** | Decouple perception from reasoning; ground LLM in 3DSG via geometry tools |
| **Main limitation** | Relies on ground-truth 3DSG; constructing from real data is future work |
| **Relation to our work** | Directly validates our approach of externalizing spatial reasoning from VLMs |
