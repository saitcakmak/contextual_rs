#!/bin/bash

N=2;
#
for i in {0..9}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_13 $i & done
#
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_1 $i & done
#
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_2 $i & done
#
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_3 $i & done
#
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_4 $i & done
#
#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_5 $i & done
#
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_6 $i & done
#
#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_7 $i & done
#
#for i in {0..99}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_8 $i & done

#for i in {0..99}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_9 $i -a & done

#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_10 $i & done

#for i in {0..19}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_11 $i & done

#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_12 $i & done
