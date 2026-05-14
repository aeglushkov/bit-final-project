#!/usr/bin/env bash
# Start the SenseNova-SI ("si") split-host stack:
#   morgen:18000   vLLM serving OpenGVLab/InternVL3-8B (vanilla base, text-only use)
#   neo:18001      lmdeploy serving sensenova/SenseNova-SI-1.5-InternVL3-8B
#   morgen -> neo  SSH tunnel exposes neo:18001 as morgen:18001 (transparent)
#
# This is Su's recommended pair from 2026-05-14: vanilla InternVL3-8B as the
# text planner, spatial-intelligence-enhanced SenseNova-SI as the VLM. Eval
# clients keep using http://localhost:{18000,18001}/v1 unchanged; pick the SI
# model entries with `EVA_PLANNER=internvl3-8b-text-bf16
# EVA_VLM=sensenova-si-1.5-internvl3-8b-bf16`.
#
# InternVL3-8B is multimodal but used here as a text-only planner. We pass
# `--limit-mm-per-prompt image=0` so vLLM does not allocate KV slots for vision
# tokens we never send, and `--trust-remote-code` because InternVL3's config
# defines a custom model class.
#
# Prerequisites:
#   - On neo: miniconda env at ~/miniconda3/envs/lmdeploy (same as bf16 stack).
#   - On neo: SenseNova-SI-1.5-InternVL3-8B weights at
#     ~/hf-cache/SenseNova-SI-1.5-InternVL3-8B.
#   - On morgen: InternVL3-8B weights at .hf-cache/InternVL3-8B.
#   - morgen -> neo passwordless SSH (id_ed25519 in ~/.ssh).
#
# Run via launch.sh so it survives SSH drops:
#   ./scripts-remote/launch.sh 03_start_servers_si
set -e
source "$(dirname "$0")/_env.sh"

LOG_PLANNER="$ROOT/logs/server_internvl3_planner_si.log"
LOG_VLM_NEO="${NEO_LOG_DIR}/server_sensenova_si_vlm.log"

mkdir -p "$ROOT/logs"

echo "============ 03_start_servers_si ============"
date

planner_up() { curl -fs http://127.0.0.1:18000/v1/models >/dev/null 2>&1; }
vlm_up()     { curl -fs http://127.0.0.1:18001/v1/models >/dev/null 2>&1; }

# --- morgen: InternVL3-8B (bf16) on vLLM, text-only use --------------------
# Same gpu-memory-utilization 0.85 as the bf16 Qwen launch (~20 GB of the 3090's 24).
# image=0 disables multimodal KV allocation since we only use the planner role.
if planner_up; then
    echo "vllm already responding on :18000, skipping start"
else
    echo "==> starting InternVL3-8B (bf16, text-only) on :18000 -> $LOG_PLANNER"
    PLANNER_LOCAL_DIR="$ROOT/.hf-cache/InternVL3-8B"
    [ -d "$PLANNER_LOCAL_DIR" ] || { echo "ERROR: $PLANNER_LOCAL_DIR not found — download weights first"; exit 1; }

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    setsid -f bash -c "
        cd '$ROOT'
        '$VLLM_ENV/bin/vllm' serve '$PLANNER_LOCAL_DIR' \
            --served-model-name 'OpenGVLab/InternVL3-8B' \
            --port 18000 \
            --gpu-memory-utilization 0.85 \
            --max-model-len 16384 \
            --dtype bfloat16 \
            --enforce-eager \
            --trust-remote-code \
            --limit-mm-per-prompt image=0 \
            > '$LOG_PLANNER' 2>&1
    " < /dev/null

    echo "==> waiting for InternVL3 planner..."
    for i in $(seq 1 60); do
        if planner_up; then echo "[ready] InternVL3 planner on :18000 (after ${i}*5s)"; break; fi
        sleep 5
    done
    planner_up || { echo "ERROR: InternVL3 planner did not become ready"; tail -30 "$LOG_PLANNER"; exit 1; }
fi

# --- neo: SenseNova-SI-1.5-InternVL3-8B (bf16) on lmdeploy (over SSH) -------
# Same pytorch backend as bf16 InternVL2 (turbomind 0.13 chokes on InternVL3's
# nested llm_config). --trust-remote-code is required for SenseNova-SI's custom
# config class.
remote_vlm_up() {
    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
        'curl -fs http://127.0.0.1:18001/v1/models >/dev/null' 2>/dev/null
}

if remote_vlm_up; then
    echo "neo lmdeploy already responding on :18001, skipping start"
else
    echo "==> starting SenseNova-SI-1.5-InternVL3-8B (bf16) on neo:18001 -> ${NEO_USER}@${NEO_HOST}:${LOG_VLM_NEO}"
    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
        "mkdir -p ${NEO_LOG_DIR}
         export HF_HOME=${NEO_HF_HOME}
         export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
         setsid -f bash -c \"
             ${NEO_LMDEPLOY_BIN} serve api_server \\\$HOME/hf-cache/SenseNova-SI-1.5-InternVL3-8B \
                 --model-name sensenova/SenseNova-SI-1.5-InternVL3-8B \
                 --server-name 127.0.0.1 \
                 --server-port 18001 \
                 --backend pytorch \
                 --dtype bfloat16 \
                 --cache-max-entry-count 0.15 \
                 --session-len 32768 \
                 --trust-remote-code \
                 > ${LOG_VLM_NEO} 2>&1
         \" < /dev/null"
fi

echo
echo "==> opening morgen -> neo SSH tunnel for :18001"
"$(dirname "$0")/_tunnel.sh" up

echo
echo "==> waiting up to 5 min for SenseNova-SI VLM to be ready (via tunnel)..."
for i in $(seq 1 60); do
    if vlm_up; then
        echo "[ready] SenseNova-SI VLM on :18001 via tunnel (after ${i}*5s)"
        break
    fi
    sleep 5
done
vlm_up || {
    echo "ERROR: SenseNova-SI VLM not responding via tunnel; remote log tail:"
    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" "tail -40 ${LOG_VLM_NEO}"
    exit 1
}

echo
echo "==> served model names (verify InternVL3 planner + SenseNova-SI VLM):"
curl -s http://127.0.0.1:18000/v1/models | head -200
echo
curl -s http://127.0.0.1:18001/v1/models | head -200

echo
echo "==> GPU usage:"
echo "morgen:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo "neo:";    ssh -o BatchMode=yes "${NEO_USER}@${NEO_HOST}" \
                  'nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'

echo
echo DONE_START_SERVERS_SI
date
