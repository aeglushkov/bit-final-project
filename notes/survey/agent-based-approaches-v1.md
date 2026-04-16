# Agent-Based Approaches for Video Understanding

## Agent-Based Papers

**VideoSeek** ([arXiv](https://arxiv.org/abs/2603.20185), [code](https://github.com/jylins/videoseek), CVPR 2026) — ReAct agent with hierarchical temporal toolkit (overview → skim → focus) for long video understanding. Uses 93% fewer frames than baselines while matching performance. Temporal navigation only, no spatial reasoning. *Separate layer on top of VLM, training-free.*

**GCA** ([arXiv](https://arxiv.org/abs/2511.22659), [project](https://gca-spatial-reasoning.github.io/)) — Geometrically-constrained agent that formalizes spatial queries into reference frame constraints and objective constraints before calling 8 specialized tools. Key finding: reference frame identification (ego-allo) is 5x more important than the objective itself. +27% over SOTA. Image-only, no video support. *Separate layer on top of VLM, training-free.*

**RieMind** ([arXiv](https://arxiv.org/abs/2603.15386)) — Agentic LLM with 30+ geometry-grounded tools operating on a 3D scene graph. Achieves 89.5% on VSI-Bench (vs ~50% for best VLMs) by fully decoupling perception from reasoning — LLM never "sees" images, only queries tools. Limitation: relies on ground-truth 3D scene graph, not practical perception. *Separate layer (LLM + tools), training-free.*

**VADAR** ([arXiv](https://arxiv.org/abs/2502.06787), [project](https://glab-caltech.github.io/vadar/)) — Program synthesis approach where 5 LLM agents dynamically generate a spatial API (Python functions like `_is_behind`, `_find_closest_object_3D`) and compose programs to answer queries. Oracle accuracy reaches 83–94%, proving spatial reasoning logic works — the bottleneck is vision specialist quality, not reasoning. Image-only. *Separate layer (multi-agent pipeline + vision specialists), training-free.*

**VideoThinker** ([arXiv](https://arxiv.org/abs/2601.15724)) — 7B model trained to autonomously invoke temporal tools (retrieval, zoom) via a caption-space data synthesis trick. Matches GPT-4o on long-video benchmarks. Interesting training paradigm, but tools are temporal only. *Tool use is trained into the model via LoRA fine-tuning, requires training.*

**AoTD** ([arXiv](https://arxiv.org/abs/2412.01694), [code](https://github.com/zhengrongz/AoTD), CVPR 2025) — Uses multi-tool agent pipeline to generate reasoning traces, then distills them into a single Video-LLM. Strong on temporal QA (74.3%) but weak on spatial (28.8% VSI-Bench). *Agent is used only at training time to generate data; final model is fine-tuned (requires training).*

**LVAgent** ([arXiv](https://arxiv.org/abs/2503.10200), [code](https://github.com/64327069/LVAgent)) — Multi-agent collaboration where 3+ MLLMs are dynamically selected, debate answers, and filter unreliable agents. First to exceed 80% on all long-video benchmarks. Temporal retrieval only, no spatial understanding. *Separate layer (orchestrator + multiple VLMs), training-free.*

## Benchmarks

**Thinking-in-Space** ([arXiv](https://arxiv.org/abs/2412.14171), [code](https://github.com/vision-x-nyu/thinking-in-space), CVPR 2025) — Introduced VSI-Bench (5K spatial QA from 288 egocentric videos). Critical finding: 71% of spatial errors come from egocentric-allocentric transformation failures, only 8% from perception. CoT/ToT prompting actually hurts spatial performance.

**SpatialScore** ([arXiv](https://arxiv.org/abs/2505.17012), [code](https://github.com/haoningwu3639/SpatialScore)) — Comprehensive benchmark with 30 spatial tasks and 5K samples. Best model reaches 60% vs human 86% (26-point gap). Their tool-augmented SpatialAgent (+7.78) competes with fine-tuning (+10.47) without catastrophic forgetting risk.

## Other

**TRACE** ([arXiv](https://arxiv.org/abs/2603.23404)) — Prompting method that guides VLMs to explicitly generate allocentric representations (room topology, camera trajectory, entity registry) before answering spatial questions. Consistently improves all tested models by ~7.5% on VSI-Bench. Works on video. Notably, generating the representation matters more than having it — the act of structured reasoning helps. *Prompting only (no external tools), training-free.*

## Results Comparison

### VSI-Bench (video spatial QA, 5K samples, human: 79.2%)

| Method | Type | Overall |
| --- | --- | --- |
| RieMind + GPT-4.1 | Agent + GT 3D scene graph | **89.5%** |
| RieMind + GPT-4o | Agent + GT 3D scene graph | 85.2% |
| RieMind + Qwen2.5-VL-7B | Agent + GT 3D scene graph | 64.1% |
| TRACE + Gemini 3 Pro | Prompting | 60.15% |
| VADAR | Agent (image-only subset) | 50.1% |
| Gemini-1.5 Pro | Base VLM | 49.1% |
| GPT-4o | Base VLM | 34.9% |
| AoTD (7B) | Distilled | 28.8% |

### SpatialScore (30 spatial tasks, 5K samples, human: 86.6%)

| Method | Type | Overall |
| --- | --- | --- |
| Gemini-3-Pro | Base VLM | 60.12% |
| GPT-5 | Base VLM | 58.13% |
| Qwen3-VL-8B + SpatialCorpus | Fine-tuned | 54.71% |
| Qwen3-VL-8B + SpatialAgent-ReAct | Agent (training-free) | 53.81% |
| Qwen3-VL-4B + SpatialCorpus | Fine-tuned | 52.99% |
| Qwen3-VL-4B + SpatialAgent-ReAct | Agent (training-free) | 50.30% |
| Qwen3-VL-8B | Base VLM | 45.48% |

**Takeaway:** Agent-based approaches (training-free) achieve +8 points over base VLMs, competitive with fine-tuning (+9-10 points).