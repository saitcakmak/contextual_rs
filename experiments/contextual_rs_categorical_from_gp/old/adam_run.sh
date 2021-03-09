#!/bin/bash

N=2;
#
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_0 $i & done
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_7 $i & done
#for i in {0..19}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_11 $i & done
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_12 $i & done
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_8 $i & done
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_9 $i & done
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_10 $i & done
#
#
#
## These are to run later if others finish first
#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_0 $i & done
#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_7 $i & done
#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_12 $i & done
for i in {0..99}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_8 $i & done
#for i in {0..99}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_9 $i & done
#for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_10 $i & done
#for i in {0..29}; do ((j=j%N)); ((j++==0)) && wait; python add_adam.py config_11 $i & done
