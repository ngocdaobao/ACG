## Set up
```bash
bash uv_install.sh
bash libs/Isaac_GR00T_n1d6/gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```
## Run server client
Terminal 1: 
```bash
#!/usr/bin/env bash

uv run --active --no-sync --directory libs/Isaac_GR00T_n1d6 \
  python -m gr00t.eval.run_gr00t_server \
  --model-path ACG/fractal_finetune \
  --embodiment-tag OXE_GOOGLE \
  --use-sim-policy-wrapper \
  --use_guidance \
  --guidance acg
  
  ```

  Terminal 2:
  ```bash

export CUDA_VISIBLE_DEVICES=0

TASKS=(
# simpler_env_google/google_robot_close_drawer
# simpler_env_google/google_robot_move_near
# simpler_env_google/google_robot_open_drawer
# simpler_env_google/google_robot_pick_coke_can
# simpler_env_google/google_robot_place_in_closed_drawer
simpler_env_google/google_robot_place_apple_in_closed_top_drawer
)
 

guidance=acg

AH=(1)
EPISODES=50
N_envs=1
SEEDS=(523)

for seed in "${SEEDS[@]}"; do
    echo "Running seed: $seed"
    for action_horizon in "${AH[@]}"; do
        echo "Running action horizon: $action_horizon"
        for TASK in "${TASKS[@]}"; do

            NAME=$(basename "$TASK")
            LOG_DIR="ACG/output/${guidance}/google_simpler_env/nenvs${N_envs}_eps${EPISODES}_ah${action_horizon}/$NAME/seed${seed}"
            VIDEO_DIR="$LOG_DIR/videos"
            mkdir -p "$LOG_DIR"
            mkdir -p "$VIDEO_DIR"

            echo "Running task: $TASK"

            libs/Isaac_GR00T_n1d6/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python libs/Isaac_GR00T_n1d6/gr00t/eval/rollout_policy.py \
                --n_episodes $EPISODES \
                --policy_client_host 127.0.0.1 \
                --policy_client_port 5555 \
                --max_episode_steps=300 \
                --env_name "$TASK" \
                --n_action_steps $action_horizon \
                --n_envs $N_envs \
                --seed $seed \
                --video_dir "$VIDEO_DIR" 

            echo "Finished task: $TASK"
            echo ""
        done
    done
done 

  ```