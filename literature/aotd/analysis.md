# AoTD — Detailed Analysis

## Core idea

The fundamental insight is that agent-based systems produce high-quality reasoning traces as a byproduct of their execution, but are too slow for deployment. AoTD treats the agent system as a *teacher* — run it offline to generate Chain-of-Thoughts, verify them, then distill into a single end-to-end model that learns to internalize the multi-step reasoning.

This is essentially **knowledge distillation from a compound AI system into a monolithic model**, where the "knowledge" being transferred is not logits or features but structured reasoning traces.

## Architecture of the agent system

The agent decomposes VideoQA into a pipeline of 5 specialist sub-tasks:

| Sub-task | Best model | Metric | Score |
|---|---|---|---|
| Question decomposition | DeepSeek-Coder-Instruct (6.7B) | Acc | 85.7% |
| Object detection | OWL-ViT v2 | IoU | 63.0% |
| Temporal grounding | UniVTG | IoU/Recall | 24.7/35.3 |
| Action recognition | LLaVA-NeXT-Video-DPO (7B) | Top1-Acc | 18.2% |
| Question answering | LLaVA-NeXT-Video-DPO (7B) | Acc | 53.4% |

**Execution flow:** LLM generates Python program → program calls tools sequentially (typically: temporal grounding → object detection → QA) → execution trace recorded → trace converted to natural language CoT.

Key code tools (from Appendix D.1): `Query_Objs`, `Query_Actions`, `Filter_frames_with_act`, `Filter_frames_with_obj`, `trim`, `Find`, `select_answer`, `exist`, `Video_summary`.

## What's interesting for our research

### 1. Agent → distillation pattern
The paper validates the hypothesis that agent-based spatial reasoning can be compressed into a single model. This is directly relevant — if we build an agent that externalizes spatial reasoning (egocentric→allocentric transforms), we could potentially distill that capability back.

### 2. Sub-task performance is surprisingly low
Temporal grounding tops out at 24.7% IoU. Action recognition at 18.2%. These are the current SOTA specialist models, yet they're quite weak. The paper works around this by filtering — only keeping traces that lead to correct answers. This means the agent system itself is unreliable; the value comes from *volume* (run on many samples, keep the ~20% that work).

### 3. Spatial reasoning is still limited
The agent system handles bounding-box-level spatial reasoning (object locations per frame) but doesn't do true allocentric spatial reasoning. It answers "where is X" with pixel coordinates, not with spatial relationships or 3D understanding. VSIBench results (28.8%) are the lowest across benchmarks, confirming spatial-temporal reasoning remains the hardest.

### 4. CoT verification is critical
Without LLM-based filtering, performance drops (Table 7). Just having correct final answers isn't enough — the reasoning path must also be coherent. This suggests naive "scrape agent traces" won't work; quality control on the reasoning chains matters.

### 5. The Python program decomposition approach
Using a code LLM to decompose questions into executable programs (modified ViperGPT approach) is elegant. The program structure naturally creates a reasoning DAG that can be traced and converted to CoTs. This is more structured than free-form agent planning.

## Limitations (from paper + my assessment)

1. **Dependent on specialist model quality** — if the underlying tools improve, the whole system improves, but current tool performance is quite low (especially temporal grounding and action recognition).

2. **Only compositional VideoQA** — focused on datasets like STAR, NExT-QA, AGQA that require multi-step reasoning. Doesn't address single-step or purely perceptual questions.

3. **Low CoT yield** — from 158.6K QA pairs, only 32.3K (~20%) produce usable CoTs after filtering. The agent system fails on ~80% of questions.

4. **No true 3D spatial reasoning** — bounding boxes in 2D frames, no depth estimation or allocentric spatial representations. This is exactly the gap our research targets.

5. **VSIBench performance** — evaluated on VSIBench (the benchmark from "Thinking in Space") but scores are modest (28.8%), reinforcing that spatial reasoning in 3D scenes remains unsolved by this approach.

## Relation to our project

**Similarities:**
- Both use agent-based approaches to externalize reasoning that VLMs struggle with
- Both decompose complex questions into sub-tasks handled by specialist models
- Both target spatial-temporal understanding in video

**Key differences:**
- AoTD focuses on VideoQA (compositional temporal questions); we focus on spatial reasoning (egocentric↔allocentric transforms)
- AoTD's spatial handling is 2D bounding boxes; we aim for 3D spatial representations
- AoTD distills back to an end-to-end model; we're exploring whether the agent layer should remain at inference time
- AoTD's agent uses off-the-shelf tools sequentially; our architecture could use VLMs purely for perception while handling spatial transforms externally

**Potential takeaways:**
- The distillation approach could be applied to our spatial agent — if we build a good agent, we could distill its spatial reasoning back
- Their sub-task evaluation methodology (Table 1) is useful for benchmarking our own specialist components
- CoT filtering is important — we should plan for low yield from agent systems and need quality control
- The code/tool decomposition pattern (ViperGPT-style) is worth considering for our agent architecture

## Code structure

```
code/
├── agent_system/
│   ├── main.py                  # Agent entry point
│   ├── vision_models.py         # Specialist model wrappers
│   ├── video_process.py         # Video loading/frame sampling
│   ├── video_clip.py            # Video clip operations
│   ├── util.py                  # Utilities
│   ├── cot_construction.py      # Run agent on datasets, generate traces
│   ├── cot_transfer.py          # Convert execution traces → NL CoTs
│   ├── cot_filter.py            # LLM-based CoT quality filtering
│   ├── cot.ipynb                # Interactive demo notebook
│   ├── configs/                 # YAML configs (base_config.yaml)
│   ├── prompts/                 # In-context examples for program generation
│   ├── datasets/                # Dataset loaders
│   └── UniVTG/                  # Temporal grounding model
├── distillation/
│   ├── train.py                 # Fine-tuning script
│   ├── train.sh                 # Training launch script
│   ├── configs/                 # Training configs
│   └── datasets/                # Distillation data loading
└── evaluation/
    ├── eval_mvbench.py          # MVBench evaluation
    ├── eval_nextqa.py           # NExT-QA evaluation
    └── scripts/                 # Evaluation launch scripts
```
