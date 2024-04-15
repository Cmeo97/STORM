#!/bin/bash
module load 2023
module load Miniconda3/23.5.2-0
conda init
source activate STORM
declare -a All_pool_layers=(mlp dino-mlp cls-transformer)
declare -a All_states=(z h joint)
declare -a All_env_names=(Boxing Pong)
declare -a All_seeds=(0)




for pool_layer in "${All_pool_layers[@]}"
do
    for state in "${All_states[@]}"
    do
        for env_name in "${All_env_names[@]}"
        do
            for seed in "${All_seeds[@]}"
            do
                sbatch train.sh pool_layer state env_name seed                
            done
        done
    done
done
