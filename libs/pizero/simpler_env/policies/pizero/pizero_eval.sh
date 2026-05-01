#!/usr/bin/env bash
#
# Self-contained evaluation script for PiZero + (ACG | CFG | WNG | none)
# guidance on SimplerEnv. Run from this directory (.../simpler_env/policies/pizero/):
#
#   bash pizero_eval.sh
#
# Edit the "User config" block to change tasks / guidance / scale / etc.

set -euo pipefail

# ---------------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------------
NUM_GPUS=1
GUIDANCE_TYPE="acg"      # one of: acg | cfg | wng | none
GUIDANCE_SCALE=3.0
GUIDANCE_SKIP_BLOCKS=(7 9 11)
GUIDANCE_NOISE_STD=1.0
N_TRAJS=50
POLICY="pizero"
CHECKPOINT="pretrained/open-pi-zero"
RESULT_ROOT="./results/default/${GUIDANCE_TYPE}"

TASKS=(
    "google_robot_pick_coke_can"
    # "google_robot_move_near"
    # "google_robot_close_drawer"
    # "google_robot_open_drawer"
    # "google_robot_place_apple_in_closed_top_drawer"
    # "widowx_put_eggplant_in_basket"
    # "widowx_spoon_on_towel"
    # "widowx_carrot_on_plate"
    # "widowx_stack_cube"
)

# ---------------------------------------------------------------------------
# Resolve directories relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../pizero/   (policy)
OPEN_PI_ZERO_DIR="$SCRIPT_DIR/open_pi_zero"                      # .../open_pi_zero/
POLICIES_DIR="$(dirname "$SCRIPT_DIR")"                          # .../policies/
SIMPLER_ENV_DIR="$(dirname "$POLICIES_DIR")"                     # .../simpler_env/
PIZERO_LIB_DIR="$(dirname "$SIMPLER_ENV_DIR")"                   # .../libs/pizero/
LIBS_DIR="$(dirname "$PIZERO_LIB_DIR")"                          # .../libs/
ACG_ROOT="$(dirname "$LIBS_DIR")"                                # .../ACG/

echo "[paths]"
echo "  SCRIPT_DIR       (= .../pizero/)        = $SCRIPT_DIR"
echo "  OPEN_PI_ZERO_DIR                         = $OPEN_PI_ZERO_DIR"
echo "  PIZERO_LIB_DIR   (= .../libs/pizero/)    = $PIZERO_LIB_DIR"
echo "  LIBS_DIR                                 = $LIBS_DIR"
echo

# ---------------------------------------------------------------------------
# 1. Activate venv if present
# ---------------------------------------------------------------------------
VENV_DIR="$PIZERO_LIB_DIR/.venv"
if [ -d "$VENV_DIR" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[venv] activating $VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi
echo "[python] $(command -v python)"
python -c "import sys; print('[python]', sys.version.split()[0])"
echo

# ---------------------------------------------------------------------------
# 2. Make sure src/, open_pi_zero/, and pizero/ stay PURE NAMESPACE PACKAGES.
#    Mixing a regular `src/__init__.py` with namespace subpackages breaks
#    Hydra's import_module("src.model.paligemma.siglip") even though
#    `from src.model... import ...` would still work.
# ---------------------------------------------------------------------------
rm -f "$OPEN_PI_ZERO_DIR/__init__.py" \
      "$OPEN_PI_ZERO_DIR/src/__init__.py" \
      "$SCRIPT_DIR/__init__.py"

# ---------------------------------------------------------------------------
# 3. PYTHONPATH so every import resolves regardless of CWD:
#       OPEN_PI_ZERO_DIR -> `from src.model... import ...`  (also for Hydra)
#       SCRIPT_DIR       -> `from pizero_model import ...`,  `simpler_adapter` (Hydra target without prefix)
#       PIZERO_LIB_DIR   -> `simpler_env.policies.pizero...`  (Hydra target)
#       LIBS_DIR         -> `robomimic.robomimic.algo.guidance...`  (guidance imports)
# ---------------------------------------------------------------------------
export PYTHONPATH="$OPEN_PI_ZERO_DIR:$SCRIPT_DIR:$PIZERO_LIB_DIR:$LIBS_DIR:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 4. Default env vars
# ---------------------------------------------------------------------------
: "${VLA_DATA_DIR:=$ACG_ROOT/data}"
: "${VLA_LOG_DIR:=$ACG_ROOT/log}"
: "${TRANSFORMERS_CACHE:=$HOME/.cache/huggingface}"
export VLA_DATA_DIR VLA_LOG_DIR TRANSFORMERS_CACHE
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"

# Useful when Hydra's resolver hides the underlying ImportError chain.
export HYDRA_FULL_ERROR=1

echo "[env]"
echo "  VLA_DATA_DIR        = $VLA_DATA_DIR"
echo "  VLA_LOG_DIR         = $VLA_LOG_DIR"
echo "  TRANSFORMERS_CACHE  = $TRANSFORMERS_CACHE"
echo "  PYTHONPATH (head)   = ${PYTHONPATH%%:*}"
echo

# ---------------------------------------------------------------------------
# 5. Run eval over tasks
# ---------------------------------------------------------------------------
GUIDANCE_FLAGS=()
if [ "$GUIDANCE_TYPE" != "none" ]; then
    GUIDANCE_FLAGS+=(--use_guidance
                     --guidance_type "$GUIDANCE_TYPE"
                     --guidance_scale "$GUIDANCE_SCALE"
                     --guidance_skip_blocks "${GUIDANCE_SKIP_BLOCKS[@]}"
                     --guidance_noise_std "$GUIDANCE_NOISE_STD")
fi

cd "$SCRIPT_DIR"  # pizero_model.py loads 'open_pi_zero/config/eval/bridge.yaml' relative to CWD

for task in "${TASKS[@]}"; do
    echo "============================================================"
    echo "[run] policy=$POLICY  task=$task  guidance=$GUIDANCE_TYPE  scale=$GUIDANCE_SCALE"
    echo "============================================================"
    python parallel_inference.py \
        --policy "$POLICY" \
        --checkpoint "$CHECKPOINT" \
        --task "$task" \
        --num-gpus "$NUM_GPUS" \
        --result-root "$RESULT_ROOT" \
        --n-trajs "$N_TRAJS" \
        "${GUIDANCE_FLAGS[@]}"
done

echo "[done] results under: $SCRIPT_DIR/$RESULT_ROOT"
