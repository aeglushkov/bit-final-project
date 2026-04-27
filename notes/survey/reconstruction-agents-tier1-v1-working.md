# Papers Survey

## Navigation

| # | Paper | Venue |
|---|---|---|
| 1 | [SAVVY: Spatial Awareness via Audio-Visual LLMs](#savvy-spatial-awareness-via-audio-visual-llms) | `NeurIPS 2025 Oral` |
| 2 | [Online Reasoning Video Segmentation w/ Just-in-Time Digital Twins](#online-reasoning-video-segmentation-with-just-in-time-digital-twins) | `ICCV 2025` |
| 3 | [LIRA: Reasoning Reconstruction via MLLMs](#lira-reasoning-reconstruction-via-multimodal-large-language-models) | `ICCV 2025` |
| 4 | [Embodied VideoAgent](#embodied-videoagent-persistent-memory-from-egocentric-videos-and-embodied-sensors) | `ICCV 2025` |
| 5 | [Feature4X](#feature4x-monocular-video-to-4d-agentic-ai-with-gaussian-feature-fields) | `CVPR 2025` |

---

## SAVVY: Spatial Awareness via Audio-Visual LLMs

`NeurIPS 2025 Oral` · 🏛️ University of Washington · 👤 Mingfei Chen, Zijun Cui, Xiulong Liu et al.

[📄 Paper](https://arxiv.org/abs/2506.05414) · [🚀 NeurIPS poster](https://neurips.cc/virtual/2025/poster/115001) · [🌐 Project](https://zijuncui02.github.io/SAVVY/)

🏷️ **SUBJECT:** Training-free pipeline that augments an AV-LLM (Gemini-2.5-pro plugin in the paper) with structured 3D spatial reasoning over long egocentric audio-visual streams.

❓ **PROBLEM:**
- Sounding objects often leave the camera view for tens of seconds; AV-LLMs ingest mono-mixed audio and lose 7-mic Aria directional cues.
- Allocentric queries require an ego→external coord transform that AV-LLMs do not perform.

💡 **IDEA:** Stage 1 estimates per-object egocentric trajectories $(t, \theta, r)$ for three roles — *target* (sounding), *reference*, *facing* — from a snapshot AV-LLM call + text-guided segmentation + spatial audio. Stage 2 lifts to global coordinates via SLAM, aggregates into a dynamic global map, then transforms to the query viewpoint.

🛠️ **SOLUTION:**
- **Stage 1:** Snapshot Descriptor (1 AV-LLM call) + Text-Guided Snapshot Segmentation (CLIPSeg + SAM2 + monocular depth) + Spatial Audio (SRP-PHAT for direction, CDR for distance).
- **Stage 2:** SLAM-projected per-frame positions; DBSCAN for static (reference, facing), Kalman for dynamic (target); allocentric aligns reference→facing with +y axis.
- **SAVVY-Bench:** built on Aria Everyday Activities; ego/allo × dir/dist over 30–300s multi-room scenes with 7-channel audio.

🏆 **RESULTS:** Overall QA 58.0% vs. Gemini-2.5-pro 50.9% (**+7.1%**); per-category gains +9.5% ego-dir, +12.3% allo-dir. Human ceiling 78.7%.

💭 **THOUGHTS:**
- Long stream + multi-specialist reconstruction + agentic detect-build-refine.

---

## Online Reasoning Video Segmentation with Just-in-Time Digital Twins

`ICCV 2025` · 🏛️ Johns Hopkins University · 👤 Yiqing Shen, Bohan Liu, Chenjia Li et al.

[📄 Paper](https://arxiv.org/abs/2503.21056)

🏷️ **SUBJECT:** Streaming reasoning segmentation as an LLM-orchestrated agent that builds a query-specific digital twin, no LLM fine-tuning.

❓ **PROBLEM:**
- Compressed video-LLMs (LLaMA-VID, VISA) lose detail needed for compositional implicit queries.
- Always-on tool ensembles waste compute; single-FM methods (LISA / GSVA / V*) fail on multi-step queries.

💡 **IDEA:** **Planner** (LLM) parses query → JSON spec of needed specialists + a DAG of perception/state/reasoning nodes. Specialists build a **just-in-time digital twin** (dynamic scene graph, sliding 6-frame window) with semantic/spatial/temporal attributes per object. **Reasoner** = base LLM (semantic) + LLM-coder (spatial/temporal predicates compiled to executable code).

🛠️ **SOLUTION:**
- Planner + base reasoner: GPT-4o-mini; LLM-coder: GPT-4o.
- Specialists: SAM-2, DepthAnything-2, OWLv2, DINOv2.
- Output: per-frame masks with temporal smoothing.
- Benchmark: 200 videos / 895 implicit queries (DAVIS + SA-V), 3 categories × 3 difficulty levels.

🏆 **RESULTS:** Beats LISA, GSVA, LLM-Seg, V*, VISA at every difficulty level on own bench. **ReVOS** $\mathcal{J}/\mathcal{F}$ = 0.748 / 0.773 vs. VISA 0.488 / 0.529 (+26 pts $\mathcal{J}$).

💭 **THOUGHTS:**
- Detect-plan-reconstruct-refine loop on streaming video: detects gaps → calls specialists → reasons.

---

## LIRA: Reasoning Reconstruction via Multimodal Large Language Models

`ICCV 2025` · 🏛️ Chinese Academy of Sciences, Baidu Inc. · 👤 Zhen Zhou, Tong Wang, Yunkai Ma et al.

[📄 Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_LIRA_Reasoning_Reconstruction_via_Multimodal_Large_Language_Models_ICCV_2025_paper.pdf) · [💻 Code](https://github.com/zhen6618/LIRA)

🏷️ **SUBJECT:** Defines *reasoning reconstruction* — implicit instruction + RGB-D sequence → incremental 3D reconstruction of instruction-relevant instances. Three-stage pipeline with a LoRA-fine-tuned MLLM.

❓ **PROBLEM:**
- Explicit-instruction 3D grounding (VLMaps, ConceptGraphs) breaks on implicit queries.
- Existing 3D instance fusion uses one keyframe at a time and ignores text features.

💡 **IDEA:** Three stages:
1. TSDF reconstruction + LoRA-tuned MLLM emits `[SEG]` tokens + structured attribute text per candidate (LLaVA + SAM mask decoder).
2. **TIFF** (Text-enhanced Instance Fusion within Fragment-Bounding-Volume) — back-projects candidates to a voxel volume, fuses *multiple keyframes simultaneously* with masked cross-attention over voxel + text features.
3. Global LLM (ChatGPT-4o-mini) reads fused candidate attribute table and picks the target.

🛠️ **SOLUTION:**
- Stage I: LLaVA-7B (LoRA) + SAM-ViT-H. Stage II: 4 cm voxels, 9-keyframe FBV, 3-block transformer + confidence head. Stage III: ChatGPT-4o-mini.
- Benchmark: **ReasonRecon** on ScanNetV2, >5k scene-instruction pairs.
- LIRA-Fast: 5.63 KFPS via quantized modules + MobileSAM.

🏆 **RESULTS:** AP / AP$_{50}$ / AP$_{25}$ = 11.57 / 34.39 / **66.24** vs. best baseline BBQ 11.52 / 22.17 / 35.86 — large lead at AP$_{25}$ (~30 pts).

💭 **THOUGHTS:**
- The output is a reconstruction, not just an answer.

---

## Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors

`ICCV 2025` · 🏛️ BIGAI, USTC, Tsinghua, Peking University · 👤 Yue Fan, Xiaojian Ma, Rongpeng Su et al.

[📄 arXiv](https://arxiv.org/abs/2501.00358) · [🌐 Project](https://embodied-videoagent.github.io)

🏷️ **SUBJECT:** LLM agent maintaining a persistent 3D object memory from egocentric video + depth/pose, with VLM-driven action-time updates and a 4-tool query surface; built on top of VideoAgent (Fan et al. 2024).

❓ **PROBLEM:**
- VideoAgent (2024) has no notion of state changes when an embodied agent acts on the scene.
- End-to-end MLLMs are unstable on dynamic 3D scenes; hand-crafted pipelines miss precise object understanding under occlusion.

💡 **IDEA:** Persistent object memory $\mathcal{M}_O$ (per-object {ID, STATE, relations, 3D Bbox, Obj-Feat, CTX-Feat}) + a VLM that detects when an action targets a specific object and patches the corresponding STATE field.

🛠️ **SOLUTION:**
- Object detector: YOLO-World; 2D-3D lifting via depth + pose; re-ID across frames.
- VLM action update: render bboxes, prompt "is this the target of action $A$?" → patch STATE.
- LLM tools: `query_db`, `temporal_loc`, `spatial_loc`, `vqa`.
- Action primitives: `chat`, `search`, `goto`, `open`, `close`, `pick`, `place`.

🏆 **RESULTS:** **+4.9** on Ego4D-VQ3D, **+5.8** on OpenEQA, **+11.7** on EnvQA. Robust under noisy estimated poses (COLMAP / DUSt3R).

💭 **THOUGHTS:**
- Memory-mechanism may be relevant.

---

## Feature4X: Monocular Video to 4D Agentic AI with Gaussian Feature Fields

`CVPR 2025` · 🏛️ UCLA, MIT, Stanford, UT Austin, DEVCOM ARL · 👤 Shijie Zhou, Hui Ren, Yijia Weng et al.

[📄 arXiv](https://arxiv.org/abs/2503.20776) · [🌐 Project](https://feature4x.github.io/)

🏷️ **SUBJECT:** Reconstructs an explicit 4D Gaussian-Splatting scene + unified compact 4D feature field distilled from three FMs from monocular video; GPT-4o agent routes segmentation / editing / VQA through the reconstruction.

❓ **PROBLEM:**
- Direct VQA over monocular video misses spatiotemporal context never observed from the right viewpoint.
- Existing 3D feature fields need multi-view calibration and are limited to one task per field.

💡 **IDEA:** Dynamic 3D Gaussian Splatting (Motion Scaffold from MoSca) + compact $D=32$ unified latent feature field, supervised by SAM2, CLIP-LSeg, and InternVideo2 features via per-FM lightweight decoders. Reasoning runs over rendered novel views and decoded features.

🛠️ **SOLUTION:**
- 4D representation: static Gaussians + scaffold-warped dynamic Gaussians; per-Gaussian feature = linear combination of scaffold-node features.
- Distilled FMs: SAM2, CLIP-LSeg, InternVideo2.
- Tasks: 2D segmentation across novel views, 3D scene editing (GPT-4o picks softmax threshold by inspecting rendered samples), 4D spatiotemporal VQA via local/global novel views.

🏆 **RESULTS:** Spatiotemporal VQA (DAVIS, 400 Qs): **61.32%** overall vs. 49.06% on direct input video (+12.3%); ~3× faster inference. Novel-view segmentation: comparable mIoU at **6.2× lower memory** than Feature 3DGS.

💭 **THOUGHTS:**
- Reconstruction from video
