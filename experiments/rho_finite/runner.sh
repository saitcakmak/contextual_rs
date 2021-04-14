#!/bin/bash

arg_list=("h_4_mean")
key_list=("ML_IKG" "ML_Gao" "ML_Gao_infer_p" "Li" "Gao")

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
