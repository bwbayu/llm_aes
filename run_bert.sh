#!/bin/bash

# List of configurations
batch_sizes=(4)
overlappings=(128 256)
epochs_list=(1)
learning_rates=(1e-5 2e-5)
config_id=0
max_jobs=2

# Function to control job concurrency
function run_job {
    while [ "$(jobs | wc -l)" -ge "$max_jobs" ]; do
        sleep 1  # Wait if the number of background jobs reaches the limit
    done
    python3 main_bert.py --batch_size $1 --overlapping $2 --epochs $3 --learning_rate $4 --config_id $5 &
}

# Loop through all combinations
for batch_size in "${batch_sizes[@]}"; do
    for overlapping in "${overlappings[@]}"; do
        for epochs in "${epochs_list[@]}"; do
            for lr in "${learning_rates[@]}"; do
                echo "Running config_id=$config_id, batch_size=$batch_size, overlapping=$overlapping, epochs=$epochs, lr=$lr"
                run_job $batch_size $overlapping $epochs $lr $config_id
                config_id=$((config_id + 1))
            done
        done
    done
done

# Wait for all processes to finish
wait
echo "All experiments completed!"

# How to run ? bash my_script.sh or sh my_script.sh