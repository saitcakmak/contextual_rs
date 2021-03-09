#!/bin/bash

N=2;

#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_0 $i & done

#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_1 $i & done

#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_2 $i & done

#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_3 $i & done

for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_9 $i -a & done
