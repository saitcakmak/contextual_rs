#!/bin/bash

arg_list=(
  "b_3_mean"
  "b_3_worst"
  "g_4_mean"
  "g_4_worst"
  "h_4_mean"
  "c_5_mean"
)
key_list=(
#  "ML_IKG"
#  "ML_Gao"
#  "Li"
#  "Gao"
  "LEVI"
)

for i in {1..1}
do
  for arg in "${arg_list[@]}"
  do
#    scancel -u gid-cakmaks
    sleep 0.1
    for key in "${key_list[@]}"
    do
      python submit.py $arg -a $key
#      sleep 300
    done
#    sleep 1200
  done
done
