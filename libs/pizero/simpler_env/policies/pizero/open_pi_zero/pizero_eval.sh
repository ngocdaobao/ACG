num_gpus=1
guidance_type="acg"
result_root="output_pizero/$guidance_type"

policies=("pizero")
checkpoints=("pretrained/open-pi-zero")
tasks=(
    "google_robot_pick_coke_can"
    # "google_robot_move_near"
    # "google_robot_close_drawer"
    # "google_robot_open_drawer"
    # "google_robot_place_apple_in_closed_top_drawer"
)

export PYTHONPATH="$PWD/libs/pizero:$PYTHONPATH"

for i in "${!policies[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running inference for ${policies[$i]} on $task"

        python parallel_inference.py \
            --policy "${policies[$i]}" \
            --checkpoint "${checkpoints[$i]}" \
            --task "$task" \
            --num-gpus $num_gpus \
            --result-root "$result_root" \
            --use_guidance \
            --guidance_type "$guidance_type" \
            --guidance_scale 3.0 \
            --n-trajs 50
    done
done

