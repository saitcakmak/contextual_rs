#!/bin/bash

N=1;

for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_1 $i & done

for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_2 $i & done

for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_3 $i & done

for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_4 $i & done
