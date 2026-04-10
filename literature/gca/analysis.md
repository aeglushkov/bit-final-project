# GCA -- Detailed Analysis

## Relation to Our Research Direction

Our project's thesis: **build an agent layer on top of VLMs that externalizes spatial reasoning, using VLMs only for perception (where they excel) while handling egocentric-allocentric transformations externally.**

GCA is the closest existing work to this thesis. It explicitly decouples VLM perception from geometric computation and formalizes the ego/allo transformation as a core architectural component (the Reference Frame Constraint). However, there are important differences in approach and scope.

## What GCA Gets Right

### 1. The Semantic-to-Geometric Gap Formulation
GCA correctly identifies that VLMs fail not at understanding spatial concepts semantically, but at grounding those concepts into precise geometry. The "oracle paradox" observation (training on data from flawed oracles) is a strong argument against training-based approaches.

### 2. Constraining the Planning Process, Not Just Execution
The key insight over prior tool-integrated agents (SpatialAgent, TIGeR): it's not enough to offload computation to tools if the VLM's *plan itself* is geometrically flawed. By forcing formalization before computation, GCA prevents the VLM from "hallucinating" spatial plans in lossy semantic space.

### 3. Reference Frame as the Primary Ambiguity
The ablation showing C_R >> C_O in importance validates a deep intuition: the hardest part of spatial reasoning is establishing *from whose perspective* or *in what coordinate system* the answer should be given. This is essentially the egocentric-allocentric transformation problem.

### 4. Training-Free, Generalizable
Works across multiple VLMs with consistent gains. The paradigm doesn't depend on specialized training data, avoiding dataset bias issues that plague SpatialLadder and others.

## Limitations and Gaps

### 1. Still VLM-Dependent for Formalization
The C_task formalization is done by the VLM itself. This means:
- 30% of errors come from incorrect formalization
- The VLM must correctly interpret complex semantics ("down" as gravity vs. camera direction)
- For difficult cases (multi-image, abstract implications), the VLM still fails

**Our opportunity:** Could we design a more structured formalization pipeline that reduces VLM burden? E.g., a classification-based approach (classify reference frame type first, then instantiate) rather than open-ended generation.

### 2. Static Reference Frames Only
GCA explicitly acknowledges it cannot handle:
- Dynamic/time-varying reference frames (video, navigation sequences)
- The C_R(t) problem where the coordinate system changes over time

**Our opportunity:** VSI-Bench includes video-based spatial reasoning. Extending the constraint formalism to temporal sequences is an open problem.

### 3. Abstract Spatial Concepts
"The living room is south of the kitchen" requires reasoning about abstract regions, not detectable objects. GCA uses camera-frame proxies here, which is fragile.

### 4. Heavy Infrastructure
The system requires:
- 2 NVIDIA A100 GPUs for the tool suite
- Ray + LangGraph for orchestration
- Multiple VFMs (VGGT, Orient Anything, SAM-2, GroundingDINO, MoGe-2)
- vLLM for VLM inference on 8 A100s

This limits reproducibility and makes iteration expensive.

### 5. Only Image-Based Benchmarks
Despite spatial reasoning in the real world being inherently 3D and often temporal (video), GCA evaluates only on image-based benchmarks. No evaluation on VSI-Bench or other video spatial benchmarks.

## Architecture Deep Dive (from code)

### LangGraph Workflow
```
[MetaPlanner (optional)] -> SemanticAnalyst -> SolverPlanner <-> SolverExecutor
                                                    ^                  |
                                                    |   (loop until    |
                                                    +--- done/budget)--+
```

- **SemanticAnalyst:** Runs C_R and C_O analysis in parallel (async). Each has up to 3 retries.
- **SolverPlanner:** ReAct-style planner that generates the next tool call(s) given history + C_task
- **SolverExecutor:** Executes tool calls, stores results in workspace, feeds back to planner
- **MetaPlanner:** Optional routing layer (can decide to skip agent pipeline for simple queries)

### State Management
- `AgentState` is a TypedDict with: messages (chat history), workspace (named variables from tools), current_plan (next tool calls), provenance maps (variable -> call_id -> inputs)
- Workspace is the "computable memory" -- all named tool outputs live here

### Tool Deployment
- All tools are Ray Serve deployments with auto-configured GPU/CPU allocation
- Dependency graph resolved via topological sort
- Autoscaling supported for some tools
- Tool-to-tool dependencies injected at init (e.g., `project_box_to_3d_points` depends on SAM-2 and VGGT handles)

### VLM Roles (3 distinct prompts)
1. **Semantic Analyst:** temperature=0.6, top_p=0.95 (needs reasoning flexibility)
2. **Tool Orchestrator:** same sampling params (needs to handle ambiguity)
3. **Coder:** temperature=0.0 (deterministic code generation)

### Knowledge-Augmented Code Generation
- Pre-built library of geometric formulas (coordinate transforms, etc.)
- Selected based on variable types in the workspace
- Injected into the coder's prompt context (RAG-style)
- Prevents the VLM from hallucinating geometric formulas

## Comparison with SpatialAgent (from our experiments)

| Aspect | SpatialAgent | GCA |
|--------|-------------|-----|
| Planning | Unconstrained ReAct | Constrained by C_task |
| Reference frame | Implicit (VLM decides during execution) | Explicit formalization before execution |
| 3D reconstruction | DUSt3R | VGGT |
| Object detection | GroundingDINO | GroundingDINO or VLM-native |
| Orientation | Not handled | Orient Anything (6-DoF) |
| Scale estimation | Not handled | MoGe-2 |
| Code generation | Direct VLM generation | KACG (formula library injection) |
| Evaluation | VSI-Bench (video) | MMSI-Bench, MindCube, OmniSpatial, SPBench, CV-Bench (images) |
| Infrastructure | Simpler | Ray + LangGraph + multiple VFMs |

## Key Takeaways for Our Project

1. **The C_task formalism is the main contribution.** The idea that spatial reasoning should be decomposed into (a) identifying the coordinate frame and (b) defining the objective within that frame is powerful and well-validated.

2. **Reference frame formalization is the bottleneck and the opportunity.** 30% of errors come from formalization, and C_R is far more important than C_O. Improving reference frame identification could yield large gains.

3. **The toolbox is secondary.** The same paradigm works across different VLMs and presumably different tool implementations. What matters is the constraint structure, not the specific tools.

4. **Video/temporal extension is unexplored.** GCA doesn't address dynamic reference frames or video-based spatial reasoning, which is our focus with VSI-Bench.

5. **The KACG approach is worth adopting.** Pre-built verified geometric formulas avoid VLM hallucination in code generation. This is a practical engineering insight that would improve any agent system.
