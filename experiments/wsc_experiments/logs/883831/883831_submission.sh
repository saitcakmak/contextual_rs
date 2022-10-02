#!/bin/bash

# Parameters
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --error=/home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%A_%a/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=submitit
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%A_%a/%A_%a_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@90
#SBATCH --time=60
#SBATCH --wckey=''

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%A_%a/%A_%a_%t_log.out --error /home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%A_%a/%A_%a_%t_log.err /home/gid-cakmaks/anaconda3/envs/contextual_rs/bin/python -u -m submitit.core._submit /home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%j
