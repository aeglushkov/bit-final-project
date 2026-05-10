#!/usr/bin/env bash
# Create the openeqa-habitat conda env with habitat-sim 0.2.5 (headless), as
# specified in the OpenEQA repo's data/README.md. Conda's solver is slow with
# habitat-sim — typical run is 5-15 min. Idempotent (skips if env exists).
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "============ 13a_install_habitat ============"
date

if [ -d "$OPENEQA_HABITAT_ENV" ]; then
    echo "[skip] $OPENEQA_HABITAT_ENV already exists"
else
    CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
    if [ ! -x "$CONDA_BIN" ]; then
        echo "ERROR: conda not found at $CONDA_BIN. Set CONDA_BIN if needed." >&2
        exit 1
    fi
    "$CONDA_BIN" create -p "$OPENEQA_HABITAT_ENV" python=3.9 -y
    "$CONDA_BIN" install -p "$OPENEQA_HABITAT_ENV" \
        habitat-sim==0.2.5 headless \
        -c conda-forge -c aihabitat -y
fi

# Sanity check
"$OPENEQA_HABITAT_ENV/bin/python" -c "import habitat_sim; print('habitat-sim', habitat_sim.__version__)"

echo "==> done. Next: bash scripts-remote/13b_download_hm3d.sh"
date
