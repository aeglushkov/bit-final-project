#!/usr/bin/env bash
# Stop both inference servers cleanly.
set -e
source "$(dirname "$0")/_env.sh"

echo "============ 04_stop_servers ============"
echo "==> SIGINT to vllm and lmdeploy"
pkill -SIGINT -f "vllm serve" 2>/dev/null || echo "  no vllm running"
pkill -SIGINT -f "lmdeploy serve api_server" 2>/dev/null || echo "  no lmdeploy running"

# give them up to 30s to shut down
for i in $(seq 1 30); do
    if ! pgrep -f "vllm serve" >/dev/null && ! pgrep -f "lmdeploy serve api_server" >/dev/null; then
        echo "==> both stopped"
        break
    fi
    sleep 1
done

if pgrep -f "vllm serve" >/dev/null; then
    echo "==> vllm still up; SIGKILL"
    pkill -9 -f "vllm serve" || true
fi
if pgrep -f "lmdeploy serve api_server" >/dev/null; then
    echo "==> lmdeploy still up; SIGKILL"
    pkill -9 -f "lmdeploy serve api_server" || true
fi

echo
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo DONE_STOP_SERVERS
