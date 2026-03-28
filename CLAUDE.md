# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating spatial reasoning in multimodal LLMs. The research direction: build an agent layer on top of VLMs that externalizes spatial reasoning, using VLMs only for perception (where they excel) while handling egocentric-allocentric transformations externally.

## Repository Structure

```
paper-thinking-in-space/
├── literature/                         # Papers read, analyzed, with authors' code
│   └── thinking-in-space/              # "Thinking in Space" (Yang et al., CVPR 2025 Oral)
│       ├── Thinking in Space.pdf
│       ├── summary.md                  # Paper summary & URLs
│       ├── analysis.md                 # Detailed analysis & findings
│       └── code/                       # Authors' evaluation codebase (lmms-eval fork)
│           ├── evaluate_all_in_one.sh
│           ├── lmms_eval/
│           │   ├── __main__.py         # CLI entry point
│           │   ├── evaluator.py        # Core evaluation loop
│           │   ├── api/                # Registry, base model/task classes, metrics
│           │   ├── models/             # 40+ model implementations
│           │   └── tasks/vsibench/     # VSI-Bench task definition
│           │       ├── vsibench.yaml   # Task config (dataset, generation params)
│           │       └── utils.py        # Benchmark logic: prompts, metrics, aggregation
│           └── data/meta_info/         # 3D scene metadata
├── notes/                              # Research ideas, findings
│   ├── idea-agent-architecture.md      # Agent architecture proposal (from Diwei Su)
│   └── meetings/                       # Meeting/discussion notes (date-prefixed)
├── experiments/                        # Own code and experiments
├── writing/                            # Own paper drafts
└── CLAUDE.md
```

**Convention for adding a new paper:** create `literature/<paper-name>/` with the PDF, summary.md, analysis.md, and optionally `code/` for authors' code.

## Setup & Installation (authors' evaluation code)

```bash
conda create --name vsibench python=3.10
conda activate vsibench
cd literature/thinking-in-space/code
git submodule update --init --recursive
cd transformers && pip install -e . && cd ..
pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
pip install deepspeed
```

Requires a custom dev build of transformers (v4.45.0.dev0) via git submodule — must install that first.

## Running Evaluations

```bash
cd literature/thinking-in-space/code

# Single model
bash evaluate_all_in_one.sh --model llava_one_vision_qwen2_7b_ov_32f

# All available models
bash evaluate_all_in_one.sh --model all

# Quick test with sample limit
bash evaluate_all_in_one.sh --model internvl2_2b_8f --limit 10

# Direct CLI
accelerate launch --num_processes=4 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --output_path logs/
```

API models require `OPENAI_API_KEY` or `GOOGLE_API_KEY` env vars.

## Architecture Notes (authors' code)

- **Model registry** (`lmms_eval/models/__init__.py`): maps model name strings to classes via `AVAILABLE_MODELS` dict. Models are loaded dynamically.
- **Task system**: YAML configs in `tasks/` define dataset source, prompts, and generation params. `utils.py` handles all custom logic per task.
- **VSI-Bench metrics**: MCA tasks use exact-match accuracy; NA tasks use Mean Relative Accuracy (MRA) — a threshold-sweep metric over [0.5, 0.95].
- **Evaluation uses `accelerate`** for multi-GPU distribution. Large models (40B+, 72B) use `device_map=auto` with `num_processes=1`.

## Formatting (authors' code)

- Black with `line-length = 240`
- isort for import sorting
