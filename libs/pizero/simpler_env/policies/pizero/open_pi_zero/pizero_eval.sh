set -euo pipefail

NUM_GPUS=1
GUIDANCE_TYPE="acg"      # one of: acg | cfg | wng | none
GUIDANCE_SCALE=3.0
GUIDANCE_SKIP_BLOCKS=(7 9 11)
GUIDANCE_NOISE_STD=1.0
N_TRAJS=50
POLICY="pizero"
CHECKPOINT="pretrained/open-pi-zero"
RESULT_ROOT="output_pizero/${GUIDANCE_TYPE}"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../open_pi_zero/
POLICY_DIR="$(dirname "$SCRIPT_DIR")"                            # .../pizero/   (policy)
POLICIES_DIR="$(dirname "$POLICY_DIR")"                          # .../policies/
SIMPLER_ENV_DIR="$(dirname "$POLICIES_DIR")"                     # .../simpler_env/
PIZERO_LIB_DIR="$(dirname "$SIMPLER_ENV_DIR")"                   # .../libs/pizero/
LIBS_DIR="$(dirname "$PIZERO_LIB_DIR")"                          # .../libs/
ACG_ROOT="$(dirname "$LIBS_DIR")"                                # .../ACG/

echo "[paths]"
echo "  SCRIPT_DIR     = $SCRIPT_DIR"
echo "  POLICY_DIR     = $POLICY_DIR"
echo "  PIZERO_LIB_DIR = $PIZERO_LIB_DIR"
echo "  LIBS_DIR       = $LIBS_DIR"
echo

VENV_DIR="$PIZERO_LIB_DIR/.venv"
if [ -d "$VENV_DIR" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[venv] activating $VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi
echo "[python] $(command -v python)"
python -c "import sys; print('[python]', sys.version.split()[0])"
echo

touch "$SCRIPT_DIR/__init__.py" \
      "$SCRIPT_DIR/src/__init__.py" \
      "$POLICY_DIR/__init__.py"

export PYTHONPATH="$SCRIPT_DIR:$POLICY_DIR:$PIZERO_LIB_DIR:$LIBS_DIR:${PYTHONPATH:-}"

: "${VLA_DATA_DIR:=$ACG_ROOT/data}"
: "${VLA_LOG_DIR:=$ACG_ROOT/log}"
: "${TRANSFORMERS_CACHE:=$HOME/.cache/huggingface}"
export VLA_DATA_DIR VLA_LOG_DIR TRANSFORMERS_CACHE
mkdir -p "$VLA_DATA_DIR" "$VLA_LOG_DIR"

echo "[env]"
echo "  VLA_DATA_DIR        = $VLA_DATA_DIR"
echo "  VLA_LOG_DIR         = $VLA_LOG_DIR"
echo "  TRANSFORMERS_CACHE  = $TRANSFORMERS_CACHE"
echo "  PYTHONPATH (head)   = ${PYTHONPATH%%:*}"
echo

GUIDANCE_FLAGS=()
if [ "$GUIDANCE_TYPE" != "none" ]; then
    GUIDANCE_FLAGS+=(--use_guidance
                     --guidance_type "$GUIDANCE_TYPE"
                     --guidance_scale "$GUIDANCE_SCALE"
                     --guidance_skip_blocks "${GUIDANCE_SKIP_BLOCKS[@]}"
                     --guidance_noise_std "$GUIDANCE_NOISE_STD")
fi

cd "$SCRIPT_DIR"  

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
