# Scaling Spatial Intelligence with Multimodal Foundation Models

`Arxiv 2026` · 🏛️ SenseTime Research · 🏛️ Nanyang Technological University

[📄 Paper](https://arxiv.org/abs/2511.13719) · [💻 Code](https://github.com/OpenSenseNova/SenseNova-SI) · [🤗 Models](https://huggingface.co/collections/sensenova/sensenova-si)

🏷️ **SUBJECT:** Data-centric scaling study that builds the **SenseNova-SI** family by training Qwen3-VL, InternVL3, and Bagel on 8M curated spatial QA samples — no architectural change.

❓ **PROBLEM:**
- Spatially-grounded training data is scarce, fragmented, and heavily skewed toward measurement/relations; perspective-taking (PT) is severely under-represented.
- The community lacks a systematic picture of how spatial intelligence scales with data — and what its ceiling looks like.
- Prior spatial-intelligence open-source models either chase narrow capabilities or rely on text-CoT recipes whose benefit is not validated.

💡 **IDEA:** Curate **SenseNova-SI-8M** under the five-capability EASI taxonomy (MM, SR, MR, PT, CR) — with deliberate emphasis on the long-overlooked **Perspective-Taking** axis — and run a controlled scaling study across three foundation-model families to expose data-scaling laws, emergent transfer, and the limits of textual chain-of-thought.

🛠️ **SOLUTION:**
- **SenseNova-SI-8M corpus:** 3.3M reorganized open-source QA + 4.5M newly synthesized from ScanNet/ScanNet++/SUN RGB-D/CA-1M/Ego-Exo4D/MessyTable/Matterport3D, filtered for ambiguity, visibility, and cross-view difficulty.
- **EASI taxonomy:** five spatial capabilities — Metric Measurement, Spatial Relations, Mental Reconstruction, **Perspective-Taking** (View Correspondence → Camera Motion → Allocentric Transformation), Comprehensive Reasoning.
- **One-epoch SFT:** 128 GPUs · batch 2048 · 5e-6 LR · max 16 frames; backbones unchanged.
- **CoT ablations:** GPT-5-annotated CoT, MindCube CogMap CoT, and a procedural continuous-coordinate **CGMap CoT**; also GRPO RL.

🏆 **RESULTS:** SenseNova-SI sets open-source SoTA across spatial benchmarks (**VSI 68.8, MMSI 43.3, MindCube 85.7, ViewSpatial 54.7, SITE 47.7, BLINK 63.9, 3DSR 55.5, EmbSpatial 72.0**) — outperforming GPT-5 on perspective-taking — while preserving MMBench-En 84.9; EmbodiedBench manipulation jumps **+59.6%** over the InternVL3-8B base with no finetuning.

💭 **THOUGHTS:**
- **Text-CoT may be the wrong primitive for 3D reasoning.** All three CoT recipes underperform plain QA scaling on VSI-Bench; the authors suggest "a broader paradigm shift beyond conventional CoT" — directly aligned with our agent direction.
- **Scaling saturates.** Gains diminish past ~5M samples; reaching human-level likely needs different reasoning mechanisms, not more data.
- **PT is a meta-capability.** Training only on PT data improves MR by +46% and CR by +11% (Table 7) — a strong hint that allocentric transformation is the load-bearing skill we should externalize.
