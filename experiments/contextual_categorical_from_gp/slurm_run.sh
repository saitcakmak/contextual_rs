#!/bin/bash

seeds=$(seq 0 4)
labels=("LCEGP" "Li" "Gao")
config="config_0"

for seed in $seeds
do
  for label in ${labels[@]}
  do
    sleep 0.1
    sbatch --requeue /home/gid-cakmaks/contextual_rs/experiments
    /contextual_categorical_from_gp/sub.sub $config $label $seed
  done
done
