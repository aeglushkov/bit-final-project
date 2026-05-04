#!/usr/bin/env bash
# Start vllm (Qwen2.5-7B planner) on :18000 and lmdeploy (InternVL2-8B VLM) on :18001.
# Both processes are launched in this script's session via setsid -f so they
# survive when this script exits, but their stdout goes to logs/<server>.log.
# Use 04_stop_servers.sh to stop both.
set -e
source "$(dirname "$0")/_env.sh"

LOG_QWEN="$ROOT/logs/server_qwen.log"
LOG_INTERNVL2="$ROOT/logs/server_internvl2.log"
mkdir -p "$ROOT/logs"

echo "============ 03_start_servers ============"
date

# Order matters: vllm first so it gets clean GPU memory, then lmdeploy with
# a small cache budget that fits in what's left. RTX 3090 24 GB:
#   vllm Qwen2.5-7B-AWQ (gpu-mem-util 0.45 of 24 GB)        ≈ 10.8 GB
#   lmdeploy InternVL2-8B-AWQ (cache-max-entry-count 0.2)   ≈  6 GB
#   leaves ~7 GB free for CLIP/embedding lookups during eval
if pgrep -f "vllm serve" >/dev/null; then
    echo "vllm already running:"
    pgrep -af "vllm serve"
else
    echo "==> starting Qwen2.5-7B-Instruct-AWQ on :18000 -> $LOG_QWEN"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid -f bash -c "
        cd '$ROOT'
        '$VLLM_ENV/bin/vllm' serve Qwen/Qwen2.5-7B-Instruct-AWQ \
            --port 18000 \
            --gpu-memory-utilization 0.45 \
            --max-model-len 16384 \
            --enforce-eager \
            > '$LOG_QWEN' 2>&1
    " < /dev/null

    echo "==> waiting for vllm before starting lmdeploy..."
    for i in $(seq 1 60); do
        if curl -fs http://127.0.0.1:18000/v1/models >/dev/null 2>&1; then
            echo "[ready] Qwen on :18000 (after ${i}*5s)"; break
        fi
        if ! pgrep -f "vllm serve" >/dev/null; then
            echo "ERROR: Qwen died before ready"; tail -30 "$LOG_QWEN"; exit 1
        fi
        sleep 5
    done
fi

if pgrep -f "lmdeploy serve api_server" >/dev/null; then
    echo "lmdeploy already running:"
    pgrep -af "lmdeploy serve api_server"
else
    echo "==> starting InternVL2-8B-AWQ on :18001 -> $LOG_INTERNVL2"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid -f bash -c "
        cd '$ROOT'
        '$LMDEPLOY_ENV/bin/lmdeploy' serve api_server OpenGVLab/InternVL2-8B-AWQ \
            --server-port 18001 \
            --model-format awq \
            --backend turbomind \
            --cache-max-entry-count 0.2 \
            > '$LOG_INTERNVL2' 2>&1
    " < /dev/null
fi

echo
echo "==> waiting up to 5 min for InternVL2 to be ready..."
for i in $(seq 1 60); do
    if curl -fs http://127.0.0.1:18001/v1/models >/dev/null 2>&1; then
        echo "[ready] InternVL2 on :18001 (after ${i}*5s)"; break
    fi
    if ! pgrep -f "lmdeploy serve api_server" >/dev/null; then
        echo "ERROR: InternVL2 died before ready"; tail -30 "$LOG_INTERNVL2"; exit 1
    fi
    sleep 5
done

curl -fs http://127.0.0.1:18000/v1/models >/dev/null 2>&1 || { echo "ERROR: Qwen no longer responding"; exit 1; }
curl -fs http://127.0.0.1:18001/v1/models >/dev/null 2>&1 || { echo "ERROR: InternVL2 not responding"; exit 1; }

echo
echo "==> GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo DONE_START_SERVERS
date
