#!/bin/bash

seeds=$(seq 0 9)
labels=(
  "DSCO_f_2"
  "C-OCBA_f_2"
  "GP-C-OCBA_f_2"
  "GP-C-OCBA_r_8"
  "TS_e_2"
  "TS_f_2"
  "TS+_e_2"
  "TS+_f_2"
  "IKG_f_2"
  "IKG_r_2"
)
config="config_covid"

for seed in $seeds
do
  for label in ${labels[@]}
  do
    sleep 0.5
    sbatch --requeue /home/gid-cakmaks/contextual_rs/experiments/simulation_experiments/sub.sub $config $label $seed -a
  done
  sleep 5
done
