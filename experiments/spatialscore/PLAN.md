# Plan: SpatialScore Initial Experiments

## Context

After reading the Thinking in Space and SpatialScore papers, the next step (per Diwei Su's plan from 2026-03-25) is to run basic experiments with SpatialScore and study its code. The primary goal is **understanding the codebase and modern VLM tooling**, not reproducing paper results.

**Hardware:** Remote server `morgenshtern` — RTX 3090 (24GB VRAM), CUDA 12.8. Can run 3B-7B models.

## Phases

### Phase 1: Environment Setup
Run `setup.sh` on the remote server to install dependencies, download Qwen2.5-VL-3B model, and download dataset images.

### Phase 2: Code Study
Study key files in the SpatialScore codebase, write annotated notes to `code-walkthrough.md`:
1. `literature/spatialscore/code/test_qwen.py` — main evaluation script (308 lines)
2. `literature/spatialscore/code/utils/util.py` — answer extraction and metrics
3. `literature/spatialscore/code/SpatialAgent/agent.py` — agent loop (94 lines)
4. `literature/spatialscore/code/SpatialAgent/utils/prompt.py` — tool definitions + 7 few-shot demos (~650 lines)
5. `literature/spatialscore/code/SpatialAgent/utils/executor.py` — tool dispatch

### Phase 3: Small-Scale Experiments
1. **Experiment 1:** 50 MMVP samples (single-image, multi-choice) — simplest case to trace code path
2. **Experiment 2:** ~100 diverse samples (10 per source, single-image only) — tests all prompt templates
3. **Experiment 3 (optional):** Compare Qwen2.5-VL-3B vs 7B on the diverse subset

### Phase 4: Analyze & Document
Run `analyze_results.py` to inspect outputs. Write `findings.md` with baseline numbers and observations relevant to our agent architecture research.

## How to Run

```bash
# 1. Setup (on remote server)
bash setup.sh

# 2. Create test datasets
cd literature/spatialscore/code
python ../../../experiments/spatialscore/create_subsets.py

# 3. Run experiment 1 (50 MMVP samples)
CUDA_VISIBLE_DEVICES=0 python test_qwen.py \
    --model_name qwen2_5vl-3b \
    --model_path ~/models/Qwen2.5-VL-3B-Instruct \
    --dataset_json_path ./dataset/SpatialScore_test50.json \
    --output_dir ./eval_results_test
    
CUDA_VISIBLE_DEVICES=0 TORCH_CUDNN_V8_API_DISABLED=1 python test_qwen.py \
    --model_name qwen2_5vl-3b \
    --model_path ~/models/Qwen2.5-VL-3B-Instruct \
    --dataset_json_path ./dataset/SpatialScore_test50.json \
    --output_dir ./eval_results_test

# 4. Run experiment 2 (diverse 100 samples)
CUDA_VISIBLE_DEVICES=0 TORCH_CUDNN_V8_API_DISABLED=1 python test_qwen.py \
    --model_name qwen2_5vl-3b \
    --model_path ~/models/Qwen2.5-VL-3B-Instruct \
    --dataset_json_path ./dataset/SpatialScore_diverse.json \
    --output_dir ./eval_results_diverse

# 5. Analyze results
python ../../../experiments/spatialscore/analyze_results.py ./eval_results_test/qwen2_5vl-3b/
python ../../../experiments/spatialscore/analyze_results.py ./eval_results_diverse/qwen2_5vl-3b/

# 6. Apply patch to authors' SpatialAgent code (needed after fresh clone/setup)
cd literature/spatialscore/code
patch -p1 < ../../../experiments/spatialscore/patches/fix-termination-msg-str.patch

# 7. Run SpatialAgent inference (experiment 1 — 50 MMVP samples)
CUDA_VISIBLE_DEVICES=0 python ../../../experiments/spatialscore/run_agent.py \
    --model_path ~/models/Qwen2.5-VL-3B-Instruct \
    --model_name qwen2_5vl-3b \
    --dataset_json_path ./dataset/SpatialScore_test50.json \
    --output_dir ./eval_results_test_agent \
    --checkpoints_dir ~/checkpoints \
    --max_steps 5

# 8. Run SpatialAgent inference (experiment 2 — diverse samples)
CUDA_VISIBLE_DEVICES=0 python ../../../experiments/spatialscore/run_agent.py \
    --model_path ~/models/Qwen2.5-VL-3B-Instruct \
    --model_name qwen2_5vl-3b \
    --dataset_json_path ./dataset/SpatialScore_diverse.json \
    --output_dir ./eval_results_diverse_agent \
    --checkpoints_dir ~/checkpoints \
    --max_steps 5

# 9. Compare baseline vs agent
python ../../../experiments/spatialscore/analyze_comparison.py \
    --baseline_dir ./eval_results_diverse/qwen2_5vl-3b \
    --agent_dir ./eval_results_diverse_agent/qwen2_5vl-3b \
    --output ../../../experiments/spatialscore/findings.md
```

## Notes

- `--dataset_name` arg in `test_qwen.py` is parsed but never used to filter data. Use filtered JSON files instead.
- If `flash-attn` fails to install, change `attn_implementation="flash_attention_2"` to `"eager"` in `test_qwen.py` line 24.
- SpatialAgent inference code was not released by the authors. We implemented the missing glue layer (actions.py, action_wrappers.py, model_registry.py, qwen_client.py, run_agent.py) in experiments/spatialscore/. All 31 unit tests pass locally.
- The authors' `SpatialAgent/agent.py` requires a patch (`patches/fix-termination-msg-str.patch`) to work with local model clients. The default autogen `_is_termination_msg` assumes messages are dicts, but local clients pass plain strings. The patch must be applied after every fresh `setup.sh` run.
