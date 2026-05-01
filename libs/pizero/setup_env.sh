#!/usr/bin/env bash
#
# Setup script for running parallel_inference.py (PiZero + ACG/CFG/WNG guidance
# on SimplerEnv) using uv.
#
# Steps:
#   1. Install uv if missing.
#   2. Create a Python 3.10 venv (.venv) at the pizero/ root.
#   3. Install open_pi_zero in editable mode (brings torch, transformers,
#      hydra, tf, bitsandbytes, ...).
#   4. Clone & install allenzren/SimplerEnv (with --recurse-submodules) and its
#      ManiSkill2_real2sim submodule.
#   5. Install robomimic (where the guidance modules live) in editable mode.
#   6. Add a .pth file so `from src.model.vla.pizero import ...` resolves
#      (open_pi_zero ships its source under src/, not as a package).
#   7. Print env-vars to set and a sample run command.
#
# Re-run safe: skips already-cloned dirs and uv re-uses cached wheels.

set -euo pipefail

# --- Paths ----------------------------------------------------------------
PIZERO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACG_ROOT="$(cd "$PIZERO_DIR/../.." && pwd)"
LIBS_DIR="$(cd "$PIZERO_DIR/.." && pwd)"
OPEN_PI_ZERO_DIR="$PIZERO_DIR/simpler_env/policies/pizero/open_pi_zero"
ROBOMIMIC_DIR="$LIBS_DIR/robomimic"
SIMPLER_ENV_DIR="$PIZERO_DIR/SimplerEnv"
VENV_DIR="$PIZERO_DIR/.venv"

echo "[1/7] Pizero root:    $PIZERO_DIR"
echo "      ACG root:       $ACG_ROOT"
echo "      open_pi_zero:   $OPEN_PI_ZERO_DIR"
echo "      robomimic:      $ROBOMIMIC_DIR"
echo "      SimplerEnv:     $SIMPLER_ENV_DIR"
echo "      venv:           $VENV_DIR"
echo

# --- 1. uv ----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[2/7] uv not found, installing via the official script..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[2/7] uv already installed: $(uv --version)"
fi

# --- 2. venv --------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[3/7] Creating venv with Python 3.10 at $VENV_DIR..."
  uv venv --python 3.10 "$VENV_DIR"
else
  echo "[3/7] Reusing venv at $VENV_DIR"
fi

# Activate so subsequent `uv pip` installs land in the right venv.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV="$VENV_DIR"

# --- 3. open_pi_zero ------------------------------------------------------
echo "[4/7] Installing open_pi_zero (-e) and its deps..."
uv pip install -e "$OPEN_PI_ZERO_DIR"

# --- 4. SimplerEnv + ManiSkill2_real2sim ----------------------------------
if [ ! -d "$SIMPLER_ENV_DIR" ]; then
  echo "[5/7] Cloning allenzren/SimplerEnv (with submodules)..."
  git clone --recurse-submodules https://github.com/allenzren/SimplerEnv "$SIMPLER_ENV_DIR"
else
  echo "[5/7] SimplerEnv already cloned at $SIMPLER_ENV_DIR; updating submodules..."
  (cd "$SIMPLER_ENV_DIR" && git submodule update --init --recursive)
fi
uv pip install -e "$SIMPLER_ENV_DIR"
if [ -d "$SIMPLER_ENV_DIR/ManiSkill2_real2sim" ]; then
  uv pip install -e "$SIMPLER_ENV_DIR/ManiSkill2_real2sim"
else
  echo "WARN: $SIMPLER_ENV_DIR/ManiSkill2_real2sim not found; submodule may have failed."
fi

# --- 5. robomimic (guidance lives here) -----------------------------------
echo "[6/7] Installing robomimic (-e)..."
uv pip install -e "$ROBOMIMIC_DIR"

# --- 6. Make `src.model.vla.pizero` importable ----------------------------
# open_pi_zero's code is laid out under `src/...` rather than as a real
# Python package, so the guidance modules' `from src.model.vla.pizero import PiZero`
# only works when the open_pi_zero directory is on sys.path. Drop a .pth file
# into site-packages so this is permanent for the venv.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PTH_FILE="$SITE_PACKAGES/open_pi_zero_src.pth"
echo "$OPEN_PI_ZERO_DIR" >"$PTH_FILE"
echo "      wrote $PTH_FILE -> $OPEN_PI_ZERO_DIR"

# --- 7. Done --------------------------------------------------------------
cat <<EOF

[7/7] Environment ready.

To activate later:
  source "$VENV_DIR/bin/activate"

Required env vars (set them before running):
  export VLA_DATA_DIR="\$PWD/data"
  export VLA_LOG_DIR="\$PWD/log"
  export TRANSFORMERS_CACHE="\$HOME/.cache/huggingface"   # PaliGemma weights live here
  # If guidance imports fail, also: export PYTHONPATH="$LIBS_DIR:\$PYTHONPATH"

Download checkpoints (from open_pi_zero README):
  https://huggingface.co/allenzren/open-pi-zero
Place them under: $PIZERO_DIR/pretrained/open-pi-zero/

Run examples (from $PIZERO_DIR):

  # No guidance baseline
  python parallel_inference.py \\
      --num-gpus 1 --policy pizero \\
      --checkpoint pretrained/open-pi-zero \\
      --task google_robot_pick_coke_can \\
      --n-trajs 10

  # ACG
  python parallel_inference.py \\
      --num-gpus 1 --policy pizero \\
      --checkpoint pretrained/open-pi-zero \\
      --task google_robot_pick_coke_can \\
      --n-trajs 10 \\
      --use_guidance --guidance_type acg \\
      --guidance_scale 3.0 --guidance_skip_blocks 7 9 11

  # CFG
  python parallel_inference.py \\
      --num-gpus 1 --policy pizero \\
      --checkpoint pretrained/open-pi-zero \\
      --task google_robot_pick_coke_can \\
      --n-trajs 10 \\
      --use_guidance --guidance_type cfg --guidance_scale 1.5

  # White noise
  python parallel_inference.py \\
      --num-gpus 1 --policy pizero \\
      --checkpoint pretrained/open-pi-zero \\
      --task google_robot_pick_coke_can \\
      --n-trajs 10 \\
      --use_guidance --guidance_type wng \\
      --guidance_scale 1.5 --guidance_skip_blocks 7 9 11 \\
      --guidance_noise_std 1.0
EOF
