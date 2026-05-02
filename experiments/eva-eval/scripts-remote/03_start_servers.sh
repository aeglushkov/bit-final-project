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
            --gpu-memory-utilization 0.25 \
            --max-model-len 8192 \
            --enforce-eager \
            > '$LOG_QWEN' 2>&1
    " < /dev/null
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
            --cache-max-entry-count 0.4 \
            > '$LOG_INTERNVL2' 2>&1
    " < /dev/null
fi

echo
echo "==> waiting up to 5 min for both to be ready..."
ready_qwen=0; ready_internvl2=0
for i in $(seq 1 60); do
    if [ $ready_qwen -eq 0 ] && curl -fs http://127.0.0.1:18000/v1/models >/dev/null 2>&1; then
        echo "[ready] Qwen on :18000 (after ${i}*5s)"; ready_qwen=1
    fi
    if [ $ready_internvl2 -eq 0 ] && curl -fs http://127.0.0.1:18001/v1/models >/dev/null 2>&1; then
        echo "[ready] InternVL2 on :18001 (after ${i}*5s)"; ready_internvl2=1
    fi
    [ $ready_qwen -eq 1 ] && [ $ready_internvl2 -eq 1 ] && break
    sleep 5
done

if [ $ready_qwen -eq 0 ]; then
    echo "ERROR: Qwen did not become ready"
    tail -30 "$LOG_QWEN"
fi
if [ $ready_internvl2 -eq 0 ]; then
    echo "ERROR: InternVL2 did not become ready"
    tail -30 "$LOG_INTERNVL2"
fi

[ $ready_qwen -eq 1 ] && [ $ready_internvl2 -eq 1 ] || exit 1

echo
echo "==> GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo DONE_START_SERVERS
date
