#!/usr/bin/env bash
# Stop the bf16 split stack: morgen's vLLM, neo's lmdeploy (via SSH), and the
# tunnel. Idempotent.
set -e
source "$(dirname "$0")/_env.sh"

echo "============ 04_stop_servers_bf16 ============"

echo "==> SIGINT vllm on morgen"
pkill -SIGINT -f "vllm serve" 2>/dev/null || echo "  no vllm running"

echo "==> SIGINT lmdeploy on neo"
ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
    "pkill -SIGINT -f 'lmdeploy serve api_server' 2>/dev/null || echo '  no lmdeploy running on neo'"

# wait up to 30s
for i in $(seq 1 30); do
    morgen_done=true
    neo_done=true
    pgrep -f "vllm serve" >/dev/null && morgen_done=false
    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
        "pgrep -f 'lmdeploy serve api_server' >/dev/null" 2>/dev/null && neo_done=false
    if $morgen_done && $neo_done; then
        echo "==> both stopped"
        break
    fi
    sleep 1
done

# force-kill if needed
if pgrep -f "vllm serve" >/dev/null; then
    echo "==> vllm still up; SIGKILL"
    pkill -9 -f "vllm serve" || true
fi
ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
    "pgrep -f 'lmdeploy serve api_server' >/dev/null && (echo '==> remote lmdeploy still up; SIGKILL'; pkill -9 -f 'lmdeploy serve api_server') || true"

echo
echo "==> closing tunnel"
"$(dirname "$0")/_tunnel.sh" down

echo
echo "morgen GPU:"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "neo GPU:";    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
                      'nvidia-smi --query-gpu=memory.used --format=csv,noheader' 2>/dev/null
echo DONE_STOP_SERVERS_BF16
