#!/bin/bash

arg_list=(
 "branin"
 "greiwank"
 "hartmann"
)
key_list=(
  "GP-C-OCBA-1.0"
  "random"
  "LEVI"
)

for i in {1..1}
do
  for arg in "${arg_list[@]}"
  do
#    scancel -u gid-cakmaks
    sleep 1
    for key in "${key_list[@]}"
    do
      python submit.py $arg -a $key
#     sleep 300
    done
#   sleep 1800
  done
done
