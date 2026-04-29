# Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding

`Arxiv 2025` · 🏛️ BIGAI · 🏛️ USTC · 🏛️ Tsinghua · 🏛️ Peking University

[📄 Paper](https://arxiv.org/abs/2501.00358) · [🚀 Project](https://embodied-videoagent.github.io)

🏷️ **SUBJECT:** LLM tool-use agent that maintains a 3D object memory from egocentric video + depth + camera pose to answer questions and act in dynamic 3D scenes.

❓ **PROBLEM:**
- End-to-end MLLMs struggle with long egocentric video + dynamic 3D scenes — context cost grows fast and fine spatial-temporal cues get compressed away.
- Existing tool-using video agents (e.g. VideoAgent) build textual memory from RGB only, lacking precise 3D structure needed for embodied reasoning and planning.
- No prior memory mechanism keeps the scene representation up-to-date when actions and activities continuously rearrange objects (open/close/move/pick).

💡 **IDEA:** Augment a VideoAgent-style tool-use loop with a **persistent object memory** built jointly from egocentric RGB and embodied sensors (depth + 6D pose), and a **VLM-based memory-update** procedure that watches detected actions and rewrites the relevant object entries (state, 3D bbox, features) on the fly.

🛠️ **SOLUTION:**
- **Persistent Object Memory M_O:** per-object entry with `ID`, `STATE` (normal / open / close / in-hand), `Related Objects` (on / uphold / in / contain), `3D Bbox`, `OBJ Feat` (CLIP of crop), `CTX Feat` (CLIP of frame).
- **Construction pipeline:** YOLO-World 2D detection → SAM-2 masks → 2D-3D lifting via depth + pose → CLIP+DINOv2 visual + IoU/MaxIoS/VolSim spatial re-ID → moving-average updates; static vs dynamic objects re-IDed by separate algorithms.
- **VLM-based memory update:** LaViLa annotates an action every 2 s; for each action, retrieve candidate objects in view, render their 3D bboxes onto the frame, and prompt the VLM ("Is the object in the box the target of action X? Yes/No") to associate action ↔ object and update `STATE`.
- **Tool-use agent (ReAct loop):** four perception tools — `query_db`, `temporal_loc`, `spatial_loc`, `vqa` — plus seven embodied primitives (`chat`, `search`, `goto`, `open`, `close`, `pick`, `place`) for Habitat experiments; GPT-4o or InternVL2-8B as the LLM.
- **Two-agent synthesis:** user-LLM proposes tasks from a trimmed scene graph; assistant (Embodied VideoAgent) explores 118 HSSD scenes × 20 layouts to generate user-assistant interaction episodes.

🏆 **RESULTS:** On three dynamic-scene benchmarks Embodied VideoAgent beats both end-to-end MLLMs and prior tool-use agents: **+4.9 Succ%** on Ego4D-VQ3D (85.37 vs EgoLoc 80.49), **+5.8 ALL%** on the OpenEQA subset (47.0 vs VideoAgent 36.3 with the same GPT-4o backbone — +10.7 against VideoAgent), and **+11.7%** on EnvQA (25.91/68.00/35.50 on Events/Orders/States vs VideoAgent 5.54/65.5/12.5, with the largest jump on event-understanding questions).

💭 **THOUGHTS:**
- **Spatial reasoning is delegated, not generated.** 3D bboxes, containment relations, and ego→world coordinate transforms are computed programmatically — VLM is only asked perceptual questions ("is this the target object?", "what color is X?"). Strong evidence for the "VLMs for perception, agent for spatial reasoning" hypothesis.
- **What if poses are wrong?** Appendix Table 4 shows DUSt3R-estimated poses on OpenEQA (41.2 → 40.0 ALL) — a small drop. Worth checking how this scales to outdoor or moving-camera-only settings without depth sensors.
- **Memory is object-centric only.** No topological/room-level structure beyond Related Objects. For navigation-heavy questions ("which room is closer to the kitchen?") the spatial_loc tool falls back to CTX-feat similarity rather than a real graph.
