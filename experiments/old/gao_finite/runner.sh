#!/bin/bash

arg_list=(
  "b_3_mean"
  "g_3_mean"
  "h_3_mean"
  "b_4_mean"
  "g_4_mean"
  "h_4_mean"
  "h_5_mean"
)
key_list=(
  "ML_IKG"
  "ML_Gao"
  "ML_Gao_infer_p"
)

for i in {1..1}
do
  for arg in "${arg_list[@]}"
  do
    for key in "${key_list[@]}"
    do
      python submit.py $arg -a $key
      sleep 30
    done
  done
done
