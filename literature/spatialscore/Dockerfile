# SpatialScore — GPU image for run_spatialagent.py (Qwen2.5-VL + SpatialAgent tools).
#
# Build (from this directory):
#   docker build -t spatialscore:latest .
#
# Run (mount model weights + tool checkpoints; run from code/ for dataset paths):
#   docker run -it --gpus all \
#     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     -v ~/models/Qwen2.5-VL-7B-Instruct:/models/Qwen2.5-VL-7B-Instruct:ro \
#     -v ~/checkpoints:/checkpoints:ro \
#     -v ~/github-projects/bit-final-project/literature/spatialscore/eval_results_agent_all:/workspace/literature/spatialscore/code/eval_results_agent_all \
#     spatialscore:latest \
#     python ../run_spatialagent.py \
#       --model_path /models/Qwen2.5-VL-7B-Instruct \
#       --model_name qwen2_5vl-7b \
#       --dataset_json_path ./dataset/SpatialScore_test50.json \
#       --output_dir ./eval_results_agent \
#       --checkpoints_dir /checkpoints \
#       --max_steps 5
#
# Override base image if needed (must match host NVIDIA driver / CUDA):
#   docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime -t spatialscore:latest .

ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OpenCV / PIL / headless display; git for Hugging Face hub optional clones
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/literature/spatialscore

COPY requirements-docker.txt /workspace/literature/spatialscore/requirements-docker.txt

# Base image already includes torch/torchvision; install Python deps only
RUN pip install --no-cache-dir -r /workspace/literature/spatialscore/requirements-docker.txt

# Application code (vendor SpatialAgent under code/)
COPY . /workspace/literature/spatialscore/

WORKDIR /workspace/literature/spatialscore/code

# Default: print help (override command when running)
CMD ["python", "../run_spatialagent.py", "--help"]
