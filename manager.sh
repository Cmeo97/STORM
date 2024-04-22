#!/bin/bash

# Declare variables 
declare -a All_pool_layers=(dino-mlp)
declare -a All_states=(joint) # h joint)
declare -a All_env_names=(Boxing)
declare -a All_seeds=(2)
declare -a All_agent_pool_layers=(dino-mlp)
declare -a All_mixer_types=(concat+attn) # concat concat+attn)



for pool_layer in "${All_pool_layers[@]}"
do
    for state in "${All_states[@]}"
    do
        for env_name in "${All_env_names[@]}"
        do
            for seed in "${All_seeds[@]}"
            do
                for agent_pool_layer in "${All_agent_pool_layers[@]}"
                do
                    for mixer in "${All_mixer_types[@]}"
                    do
                        bash train.sh $pool_layer $state $env_name $seed $agent_pool_layer $mixer  
                    done
                done          
            done
        done
    done
done
