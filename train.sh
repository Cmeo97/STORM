#!/bin/bash
#SBATCH -J OC-STORM
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task 18 
#SBATCH --gpus 1 
#SBATCH --mail-type=END
#SBATCH --mail-user=c.meo@tudelft.nl
#SBATCH --mem=60G


#Loading modules

#module load 2023
#module load Miniconda3/23.5.2-0
#conda activate STORM
pool_layer=$1
env_name=Boxing
device=$2
seed=0
state=z
exp_name=${env_name}-life_done-wm_2L512D8H-100k-seed_${seed}_${pool_layer}_state_${state}
nohup python train.py \
BasicSettings.n=${exp_name} \
BasicSettings.Seed=${seed} \
BasicSettings.config_path="config_files/STORM_XL.yaml" \
BasicSettings.env_name="ALE/${env_name}-v5" \
BasicSettings.trajectory_path="D_TRAJ/${env_name}.pkl" \
BasicSettings.device=cuda:${device} \
Models.WorldModel.wm_oc_pool_layer=${pool_layer}
> 'logs/'${exp_name}'.out' 2> 'logs/'${exp_name}'.err'


