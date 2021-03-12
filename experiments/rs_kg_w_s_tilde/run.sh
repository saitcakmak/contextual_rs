#!/bin/bash

N=2;

for i in {20..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_7 $i & done

