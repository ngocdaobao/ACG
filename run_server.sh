#!/usr/bin/env bash
uv run --active --no-sync --directory libs/Isaac_GR00T_n1d6 \
  python -m gr00t.eval.run_gr00t_server \
  --model-path "nvidia/GR00T-N1.6-3B" \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper \
  --use_guidance \
  --guidance acg
  