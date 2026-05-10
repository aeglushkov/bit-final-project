# 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning

`Arxiv 2025` · 🏛️ UMass Amherst · 🏛️ CUHK · 🏛️ Columbia · 🏛️ MIT · 🏛️ MIT-IBM Watson AI Lab

[👤 Authors](https://yuncongyang.com/) · [📄 Paper](https://arxiv.org/abs/2411.17735) · [🚀 Project](https://umass-embodied-agi.github.io/3D-Mem/)

🏷️ **SUBJECT:** A 3D scene-memory format for embodied agents that exposes the scene to a VLM as a small set of multi-view snapshot images, supporting active exploration plus lifelong reasoning.

❓ **PROBLEM:**
- Object-centric 3D scene graphs (e.g. ConceptGraphs) flatten inter-object geometry into text labels and lose spatial nuance ("is there room *in front of* the armchair?").
- Dense 3D representations (point clouds, neural fields) are heavy and ill-matched to current VLMs, which were never trained on dense 3D inputs.
- Neither family represents *unexplored* regions, so they cannot drive active exploration.
- Memory grows unbounded with exploration, making per-query VLM context-length and latency blow up.

💡 **IDEA:** Represent the scene as a small set of **Memory Snapshots** — RGB views each covering a cluster of co-visible objects with their surrounding context — paired with **Frontier Snapshots** that depict candidate unexplored directions, all directly usable as visual prompts for an off-the-shelf VLM.

🛠️ **SOLUTION:**
- **Memory Snapshot:** image + the set of detected objects co-visible in it; every object is assigned to exactly one snapshot, so the snapshot set is a compact cover of the scene.
- **Co-visibility clustering (Algorithm 1):** greedily pick the frame candidate that covers the most still-unassigned objects; bisecting K-means on 2D positions splits whatever cluster remains uncovered.
- **Frontier Snapshot:** an RGB view captured at the boundary of an unexplored region (DBSCAN over the occupancy map), reusing the Explore-EQA frontier-exploration scaffold.
- **Incremental construction:** at each step only re-cluster objects in newly observed regions and merge with previous snapshots — the rest of memory is reused.
- **Prefiltering retrieval:** the VLM ranks all observed object classes by relevance to the question; keep only snapshots containing the top-K classes (K=10), shrinking ~10.94 → 3.26 snapshots on EM-EQA.
- **Decision step:** the VLM is shown filtered memory + frontier snapshots and outputs either an answer (terminate) or a frontier choice (continue exploring).

🏆 **RESULTS:** With GPT-4o, 3D-Mem reaches **52.6 / 42.0** LLM-Match / SPL on A-EQA (ConceptGraphs+Frontier 47.2 / 33.3; blind GPT-4o 35.9; human 85.1), **57.2** LLM-Match on EM-EQA at only 3.1 avg frames vs 48.1 for the 75-frame Multi-Frame baseline, and **69.1 / 48.9** Success / SPL on the GOAT-Bench Val-Unseen subset (Explore-EQA 55.0 / 37.9).

💭 **THOUGHTS:**
- **Open question 1:** the VLM still sees raw pixels for every spatial query — does externalizing memory help if the egocentric→allocentric transformation (the bottleneck identified by *Thinking-in-Space*) still happens inside the VLM?
- **Open question 2:** the YOLOv8x-World detector with a 200-class ScanNet vocabulary is a hard ceiling — every failure mode in §8 traces back to detection or limited VLM resolution. How much of the gain is from the snapshot format vs. just "show the VLM the right image"?
- **Open question 3:** static-scene assumption + single floor + precise agent pose are real constraints for any "lifelong" claim.
