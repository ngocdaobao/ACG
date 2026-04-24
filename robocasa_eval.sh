source .venv/bin/activate
# === RoboCasa rollout: without guidance ===
note=""
n_rollouts="24"                   # number of rollout episodes
num_batch_envs="8"                # number of parallel environments
export MAX_NUM_EMBODIMENTS="32"
dataset_name="robocasa_mg100"
config_path="libs/Isaac-GR00T-N1/robomimic_configs/${dataset_name}.json"
model_path="DAVIAN-Robotics/GR00T-N1-2B-tuned-RoboCasa-MG100-FrankaPandaGripper"
seed="123"

bash scripts/base_rollout.sh ${config_path} ${model_path} ${seed} ${n_rollouts} ${num_batch_envs} "${note}"




# python -m robocasa.scripts.download_datasets --ds_types mg_im --tasks PnPCounterToCab