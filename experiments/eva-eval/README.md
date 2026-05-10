# eva-eval

VSI-Bench evaluation harness for Embodied VideoAgent (Fan et al., ICCV 2025) with swappable LLM/VLM backends.

The Embodied VideoAgent (EVA) paper code lives in a separate repo at
`../../literature/EmbodiedVideoAgent/code/` (gitignored from this repo).
This harness imports the paper's perception stack as a library, adds the
four agent tools that VSI-Bench needs, and runs a sweep over the dataset.

## Setup (Linux, 24 GB+ NVIDIA GPU)

```sh
# 1. Paper code: YOLO-World + SAM2 + CLIP + DINOv2 + ObjectMemory
cd literature/EmbodiedVideoAgent/code
conda env create -f environment.yaml
conda activate e-videoagent
conda install habitat-sim==0.3.0 withbullet -c conda-forge -c aihabitat
pip install -r requirements.txt
pip install -e .

# 2. eva-eval
cd ../../../experiments/eva-eval
pip install -e ".[dev]"

# 3. Inference servers (paper-faithful split: text planner via vllm + VLM via lmdeploy)
# Two separate envs because vllm and lmdeploy pin different torch versions.

# Planner: Qwen2.5-7B-Instruct-AWQ via vllm (~5 GB VRAM)
# vllm 0.6.6.post1 + torch 2.5.1+cu124, transformers pinned <4.50 to avoid
# the hf_hub 1.x / `is_offline_mode` import break.
pip install "vllm==0.6.6.post1" "transformers>=4.45.0,<4.50"
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --port 18000 \
    --gpu-memory-utilization 0.25 \
    --max-model-len 8192 \
    --enforce-eager &

# VLM: InternVL2-8B-AWQ via lmdeploy (~6 GB VRAM)
# vllm 0.6.6's InternVL2 loader chokes on the AWQ wqkv combined weights
# ("KeyError: model.layers.0.attention.wqkv.qweight"); lmdeploy's turbomind
# backend natively supports the InternLM AWQ layout.
# Use a *separate* env: torch 2.4.1+cu121, transformers <4.46, plus timm.
pip install "lmdeploy>=0.6,<0.7" "transformers>=4.41.0,<4.46" timm einops
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
lmdeploy serve api_server OpenGVLab/InternVL2-8B-AWQ \
    --server-port 18001 \
    --model-format awq \
    --backend turbomind \
    --cache-max-entry-count 0.4 &
```

Combined VRAM ~12 GB, leaves headroom for CLIP/embedding lookups during
eval. Tune `--gpu-memory-utilization` (vllm) and `--cache-max-entry-count`
(lmdeploy) if you hit OOM.

For Azure GPT-4o set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`
before invoking any script.

### MASt3R-SfM (Phase 2 preprocessing)

Not on PyPI — clone and put on `PYTHONPATH`. Run in a **separate** env from
`e-videoagent` (different torch version):

```sh
git clone --recursive https://github.com/naver/mast3r ~/mast3r
cd ~/mast3r && pip install -r requirements.txt

# MASt3R's requirements.txt doesn't pin torch, so pip pulls the latest
# (currently cu13) wheels that won't run against drivers <580. Pin to
# CUDA 12.1 wheels which work on driver 525+ and the RTX 3090 stack.
pip install --upgrade --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

mkdir -p checkpoints && cd checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
echo 'export PYTHONPATH=$HOME/mast3r:$PYTHONPATH' >> ~/.bashrc
```

Run preprocessing (default: 1 fps, swin-5 scene graph, 512 px):

```sh
python scripts/01_preprocess.py \
    --video-dir /path/to/vsibench/videos \
    --cache-root cache/vsibench/
```

### Build detection vocabulary (run once)

Generates `config/detection_classes.txt` by merging the EVA paper's
`customized_classes` with frequent terms from VSI-Bench questions.
The output is heuristic — review and edit if needed.

```sh
python scripts/build_vocabulary.py \
    --paper-code-dir ../../literature/EmbodiedVideoAgent/code \
    --output config/detection_classes.txt \
    --min-count 3 --max-classes 200 --include-options
```

### Build object memory (Phase 3)

Per video: load Phase 2 cache → run paper's `ObjectMemory.process_a_frame`
on every frame → pickle the memory state.

```sh
python scripts/02_build_memory.py \
    --cache-root cache/vsibench/ \
    --classes-file config/detection_classes.txt \
    --paper-code-dir ../../literature/EmbodiedVideoAgent/code
```

Per-video output: `cache/vsibench/<video_id>/memory.pkl`. Failures are
logged to `memory_failures.jsonl` and the script exits non-zero so a
re-run can target failed videos.

### Run VSI-Bench (Phase 5)

Dev subset (500 questions, stratified by task type — sanity gate):

```sh
python scripts/03_run_vsibench.py \
    --cache-root cache/vsibench/ \
    --paper-code-dir ../../literature/EmbodiedVideoAgent/code \
    --classes-file config/detection_classes.txt \
    --output results/dev500.jsonl \
    --limit 500
```

Full sweep, per model:

```sh
# default planner = qwen2.5-7b-text, default vlm = internvl2-8b (per config/models.yaml)
python scripts/03_run_vsibench.py \
    --cache-root cache/vsibench/ \
    --paper-code-dir ../../literature/EmbodiedVideoAgent/code \
    --classes-file config/detection_classes.txt \
    --output results/full_qwen.jsonl

# Override planner (e.g., switch to GPT-4o once you've set Azure env vars)
python scripts/03_run_vsibench.py --planner gpt-4o ... --output results/full_gpt4o.jsonl
```

Outputs: `results/<name>.jsonl` (one row per question with prediction and
score) plus `results/<name>.jsonl.summary.json` with aggregated metrics
matching the leaderboard layout (overall + 8 task types, with
`object_rel_direction` rolled up across easy/medium/hard).

## Switching models

Edit `config/models.yaml` — change `default_model` at the top, or pass
`--model NAME` to any script. New backends register under `models:` with
either `backend: openai_compatible` (vLLM, lmdeploy, Ollama, OpenAI proper)
or `backend: azure_openai`.

## Roadmap

| Phase | Status |
|---|---|
| 0 — scaffolding | done |
| 1 — model abstraction | in progress |
| 2 — MASt3R-SfM depth+pose preprocessing | not started |
| 3 — memory build via paper's `ObjectMemory` | not started |
| 4 — tools + ReAct agent | not started |
| 5 — VSI-Bench harness | not started |
| 6 — full sweep | not started |

See `notes/` in the parent repo for the running plan.
