gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python -u gr00t/eval/rollout_policy.py \
    --n_episodes 1 \
    --n_envs 1 \
    --max_episode_steps 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --env_name simpler_env_widowx/widowx_spoon_on_towel \
    --n_action_steps 4
