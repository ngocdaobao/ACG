uv run scripts/try_checkpoint_in_simpler.py \
    --task google_robot_pick_horizontal_coke_can \
    --checkpoint_path config/eval/bridge_beta_step19296_2024-12-26_22-30_42.pt \
    --recording \
    --use_bf16 \
    --use_torch_compile # first batch will be slow