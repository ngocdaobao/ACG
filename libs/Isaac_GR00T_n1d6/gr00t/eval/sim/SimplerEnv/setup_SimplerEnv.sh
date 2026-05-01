#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
SIMPLER_REPO="$PROJECT_REPO/external_dependencies/SimplerEnv"
UV_ENV="$SCRIPT_DIR/simpler_uv"

# SAPIEN requires libvulkan.so.1. Find it in conda envs if not already on LD_LIBRARY_PATH.
if ! ldconfig -p 2>/dev/null | grep -q libvulkan && ! echo "${LD_LIBRARY_PATH:-}" | grep -q vulkan; then
  VULKAN_LIB=$(find "$HOME" /pfss -name "libvulkan.so.1" 2>/dev/null | head -1)
  if [ -n "$VULKAN_LIB" ]; then
    export LD_LIBRARY_PATH="$(dirname "$VULKAN_LIB"):${LD_LIBRARY_PATH:-}"
    echo "Using libvulkan from: $(dirname "$VULKAN_LIB")"
  else
    echo "WARNING: libvulkan.so.1 not found. SAPIEN may fail to import." >&2
  fi
fi

if [ -z "$(ls -A "$SIMPLER_REPO" 2>/dev/null)" ]; then
  git clone --recurse-submodules https://github.com/squarefk/SimplerEnv.git "$SIMPLER_REPO"
else
  echo "SimplerEnv already cloned, skipping."
fi

# Numpy pin: cluster uses 1.26.4; SimplerEnv README mentions 1.24.4 for pinocchio IK.
# Override by exporting SIMPLER_NUMPY=1.24.4 if needed.
SIMPLER_NUMPY="${SIMPLER_NUMPY:-1.26.4}"

# python -m pip install -U uv
rm -rf "$UV_ENV"
mkdir -p "$UV_ENV"
uv venv "$UV_ENV/.venv" --python 3.10
source "$UV_ENV/.venv/bin/activate"
uv pip install "setuptools<81"  # >=81 removed top-level pkg_resources that SAPIEN needs

# Core deps (match cluster’s pyproject pattern)
uv pip install \
  gymnasium==0.29.1 \
  json-numpy>=2.1.1 \
  numpy=="$SIMPLER_NUMPY" \
  opencv-python-headless==4.10.0.84 \
  ray==2.48.0

# Install SimplerEnv sources (editable)
# uv pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim" --config-settings editable_mode=compat
# uv pip install -e "$SIMPLER_REPO" --config-settings editable_mode=compat

uv pip install -e "$SIMPLER_REPO/ManiSkill2_real2sim"
uv pip install -e "$SIMPLER_REPO"

# Make your OSS project importable
uv pip install --editable "$PROJECT_REPO" --no-deps

uv pip install tianshou==0.5.1 pydantic av zmq torchvision==0.22.0 transformers==4.51.3

# Sanity check
python - <<'PY'
from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs
register_simpler_envs()
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
print("SimplerEnv import OK")
import gymnasium as gym
env = gym.make("simpler_env_google/google_robot_pick_object")
env.reset()
env.close()
print("Env OK:", type(env))
PY

echo "SimplerEnv ready at: $UV_ENV/.venv/bin/python3"

/user/hominhduy.nguyen01/u24877/.cache/uv/archive-v0/v7Kers2dgs_aDbfY6KAck/glfw/x11                            
/user/hominhduy.nguyen01/u24877/.cache/uv/archive-v0/z0iNT9k-YyYFXYM6R4QXv/OpenGL/EGL/EXT/platform_x11.py      
/user/hominhduy.nguyen01/u24877/.cache/uv/archive-v0/z0iNT9k-YyYFXYM6R4QXv/OpenGL/EGL/KHR/platform_x11.py  
## STUFF I ADDED MANUALLY


