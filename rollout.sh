
export CUDA_VISIBLE_DEVICES=0

TASKS=(
robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env	
robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env	
robocasa_panda_omron/CoffeePressButton_PandaOmron_Env	
robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env	
robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env	
robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env	
robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env	
robocasa_panda_omron/OpenDrawer_PandaOmron_Env	
robocasa_panda_omron/CloseDrawer_PandaOmron_Env	
robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env	
robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env	
robocasa_panda_omron/PnPCounterToCab_PandaOmron_Env	
robocasa_panda_omron/PnPCabToCounter_PandaOmron_Env	
robocasa_panda_omron/PnPCounterToSink_PandaOmron_Env	
robocasa_panda_omron/PnPSinkToCounter_PandaOmron_Env	
robocasa_panda_omron/PnPCounterToMicrowave_PandaOmron_Env	
robocasa_panda_omron/PnPMicrowaveToCounter_PandaOmron_Env	
robocasa_panda_omron/PnPCounterToStove_PandaOmron_Env	
robocasa_panda_omron/PnPStoveToCounter_PandaOmron_Env	
robocasa_panda_omron/TurnOnSinkFaucet_PandaOmron_Env	
robocasa_panda_omron/TurnOffSinkFaucet_PandaOmron_Env	
robocasa_panda_omron/TurnSinkSpout_PandaOmron_Env	
robocasa_panda_omron/TurnOnStove_PandaOmron_Env	
robocasa_panda_omron/TurnOffStove_PandaOmron_Env
)
 



AH=(8)
EPISODES=50
N_envs=1
SEEDS=(123)

for seed in "${SEEDS[@]}"; do
    echo "Running seed: $seed"
    for action_horizon in "${AH[@]}"; do
        echo "Running action horizon: $action_horizon"
        for TASK in "${TASKS[@]}"; do

            NAME=$(basename "$TASK")
            LOG_DIR="ACG/output/cfg/robocasa/nenvs${N_envs}_eps${EPISODES}_ah${action_horizon}/$NAME/seed${seed}"
            VIDEO_DIR="$LOG_DIR/videos"
            mkdir -p "$LOG_DIR"
            mkdir -p "$VIDEO_DIR"

            echo "Running task: $TASK"

            libs/Isaac_GR00T_n1d6/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python libs/Isaac_GR00T_n1d6/gr00t/eval/rollout_policy.py \
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




