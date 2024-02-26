env_name=Boxing
device=4
seed=2
exp_name=${env_name}-life_done-wm_2L512D8H-100k-seed_${seed}
nohup python -u train.py \
    -n ${exp_name} \
    -seed ${seed} \
    -config_path "config_files/STORM_XL.yaml" \
    -env_name "ALE/${env_name}-v5" \
    -trajectory_path "D_TRAJ/${env_name}.pkl" \
    -device cuda:${device} \
    > 'logs/'${exp_name}'.out' 2> 'logs/'${exp_name}'.err'
