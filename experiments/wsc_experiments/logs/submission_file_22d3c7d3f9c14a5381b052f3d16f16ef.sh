#!/bin/bash

# Parameters
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=4
#SBATCH --error=/home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%A_%a/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=submitit
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%A_%a/%A_%a_0_log.out
#SBATCH --partition=default_gpu
#SBATCH --signal=USR1@90
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%A_%a/%A_%a_%t_log.out --error /home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%A_%a/%A_%a_%t_log.err --unbuffered /home/gid-cakmaks/anaconda3/envs/contextual_rs/bin/python -u -m submitit.core._submit /home/gid-cakmaks/contextual_rs/experiments/wsc_experiments/logs/%j
