#!/usr/bin/env bash
# Start the full-precision (bf16) split-host stack:
#   morgen:18000   vLLM serving Qwen/Qwen2.5-7B-Instruct      (~20 GB on 3090)
#   neo:18001      lmdeploy serving OpenGVLab/InternVL2-8B    (~22 GB on 5090)
#   morgen -> neo  SSH tunnel exposes neo:18001 as morgen:18001 (transparent)
#
# Eval clients keep using http://localhost:{18000,18001}/v1 unchanged; pick
# the bf16 model entries with `EVA_PLANNER=qwen2.5-7b-text-bf16
# EVA_VLM=internvl2-8b-bf16`.
#
# Prerequisites:
#   - On neo: miniconda env at ~/miniconda3/envs/lmdeploy with lmdeploy + torch
#     cu128 (sm_120 support).
#   - On neo: InternVL2-8B weights at ~/hf-cache/InternVL2-8B.
#   - On morgen: Qwen2.5-7B-Instruct weights at .hf-cache/Qwen2.5-7B-Instruct.
#   - morgen -> neo passwordless SSH (id_ed25519 in ~/.ssh).
#
# Run via launch.sh so it survives SSH drops:
#   ./scripts-remote/launch.sh 03_start_servers_bf16
set -e
source "$(dirname "$0")/_env.sh"

LOG_QWEN="$ROOT/logs/server_qwen_bf16.log"
LOG_INTERNVL2_NEO="${NEO_LOG_DIR}/server_internvl2_bf16.log"

mkdir -p "$ROOT/logs"

echo "============ 03_start_servers_bf16 ============"
date

qwen_up()      { curl -fs http://127.0.0.1:18000/v1/models >/dev/null 2>&1; }
internvl2_up() { curl -fs http://127.0.0.1:18001/v1/models >/dev/null 2>&1; }

# --- morgen: Qwen bf16 on vLLM ----------------------------------------------
# Replaces the AWQ launch in 03_start_servers.sh. bf16 weights are ~14 GB so
# we raise gpu-memory-utilization from 0.45 to 0.85 (~20 GB of the 3090's 24).
if qwen_up; then
    echo "vllm already responding on :18000, skipping start"
else
    echo "==> starting Qwen2.5-7B-Instruct (bf16) on :18000 -> $LOG_QWEN"
    QWEN_LOCAL_DIR="$ROOT/.hf-cache/Qwen2.5-7B-Instruct"
    [ -d "$QWEN_LOCAL_DIR" ] || { echo "ERROR: $QWEN_LOCAL_DIR not found — download weights first"; exit 1; }

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid -f bash -c "
        cd '$ROOT'
        '$VLLM_ENV/bin/vllm' serve '$QWEN_LOCAL_DIR' \
            --served-model-name 'Qwen/Qwen2.5-7B-Instruct' \
            --port 18000 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 16384 \
            --dtype bfloat16 \
            --enforce-eager \
            > '$LOG_QWEN' 2>&1
    " < /dev/null

    echo "==> waiting for Qwen..."
    for i in $(seq 1 60); do
        if qwen_up; then echo "[ready] Qwen on :18000 (after ${i}*5s)"; break; fi
        sleep 5
    done
    qwen_up || { echo "ERROR: Qwen did not become ready"; tail -30 "$LOG_QWEN"; exit 1; }
fi

# --- neo: InternVL2 bf16 on lmdeploy (over SSH) -----------------------------
# Detect remote process by liveness of its loopback port (which doesn't
# answer over SSH but does respond locally to the same curl test).
remote_internvl2_up() {
    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
        'curl -fs http://127.0.0.1:18001/v1/models >/dev/null' 2>/dev/null
}

if remote_internvl2_up; then
    echo "neo lmdeploy already responding on :18001, skipping start"
else
    echo "==> starting InternVL2-8B (bf16) on neo:18001 -> ${NEO_USER}@${NEO_HOST}:${LOG_INTERNVL2_NEO}"
    # --backend pytorch: lmdeploy 0.13's turbomind path has a config-parsing bug
    # for InternVL2's nested llm_config dict (raises AttributeError before the
    # --trust-remote-code flag is honored). Pytorch backend takes the model
    # through transformers' AutoConfig directly and works without that crash.
    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
        "mkdir -p ${NEO_LOG_DIR}
         export HF_HOME=${NEO_HF_HOME}
         export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
         setsid -f bash -c \"
             ${NEO_LMDEPLOY_BIN} serve api_server \\\$HOME/hf-cache/InternVL2-8B \
                 --model-name OpenGVLab/InternVL2-8B \
                 --server-name 127.0.0.1 \
                 --server-port 18001 \
                 --backend pytorch \
                 --dtype bfloat16 \
                 --cache-max-entry-count 0.15 \
                 --session-len 32768 \
                 --trust-remote-code \
                 > ${LOG_INTERNVL2_NEO} 2>&1
         \" < /dev/null"
fi

echo
echo "==> opening morgen -> neo SSH tunnel for :18001"
"$(dirname "$0")/_tunnel.sh" up

echo
echo "==> waiting up to 5 min for InternVL2 to be ready (via tunnel)..."
for i in $(seq 1 60); do
    if internvl2_up; then
        echo "[ready] InternVL2 on :18001 via tunnel (after ${i}*5s)"
        break
    fi
    sleep 5
done
internvl2_up || {
    echo "ERROR: InternVL2 not responding via tunnel; remote log tail:"
    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" "tail -40 ${LOG_INTERNVL2_NEO}"
    exit 1
}

echo
echo "==> served model names (verify no -AWQ suffix):"
curl -s http://127.0.0.1:18000/v1/models | head -200
echo
curl -s http://127.0.0.1:18001/v1/models | head -200

echo
echo "==> GPU usage:"
echo "morgen:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo "neo:";    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
                  'nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'

echo
echo DONE_START_SERVERS_BF16
date
