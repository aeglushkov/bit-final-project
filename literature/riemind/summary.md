# RieMind: Geometry-Grounded Spatial Agent for Scene Understanding

`Arxiv 2026` · 🏛️ Riemann Lab, Huawei Technologies

[📄 Paper](https://arxiv.org/abs/2603.15386)

🏷️ **SUBJECT:** LLM agent for 3D indoor scene understanding via a geometry-grounded tool interface over a 3D scene graph.

❓ **PROBLEM:**
- End-to-end VLMs conflate perception and reasoning and fail on compositional 3D spatial queries.
- Fine-tuning on spatial QA data doesn't fix the underlying reasoning chain; it just memorizes patterns.
- There is no clean way to measure whether the reasoning or the perception stage is the real bottleneck.

💡 **IDEA:** Decouple perception from reasoning entirely — materialize the scene as a **3D scene graph (3DSG)** and let an LLM answer queries purely through geometry tools over node IDs, never seeing pixels.

🛠️ **SOLUTION:**
- **Hierarchical 3DSG:** Building → Floor → Room → Object nodes with ground-truth bboxes, volumes, orientations (built from ScanNet/ScanNet++/ARKitScenes annotations).
- **MCP tool surface (4 namespaces):** `mem_*` scene context, `sg_*` graph traversal/search, `geom_*` volume/area/distance primitives, `loc_*` position, frame construction, and projection.
- **Node-ID grounding:** every geometry/orientation tool takes node IDs — no free-text ambiguity; outputs are deterministic functions of the 3DSG.
- **Constrained agent prompt:** 7-section system prompt forces "search → resolve → tool-call" flow and delegates all computation to tools.

🏆 **RESULTS:** On VSI-Bench static (4,185 Qs), **RieMind + GPT-4.1 hits 89.5% avg vs. 73.6% for the best fine-tuned model** (SpaceMind), a +16 pt gain; base VLMs improve by 33–50 pts when wrapped in the agent, and a Qwen2.5-VL-7B agent (64.1%) already beats most fine-tuned 7B baselines.

💭 **THOUGHTS:**
- **GT 3DSG is the elephant in the room:** the whole framework depends on perfect annotations — quantifying degradation under a noisy RGB-D-built 3DSG is the open problem we care about.
- **Reasoning, not perception, is the bottleneck:** Qwen2.5-VL-7B *degrades* on relative-direction (-3.8 pts) even with perfect geometry, while GPT-4.1 matches fine-tuned SOTA — strong evidence that backbone reasoning capability directly gates tool-use quality.
- **Static scenes only:** route planning and temporal questions are explicitly excluded — extending the 3DSG concept to dynamic scenes is where our own agent direction can differentiate.
