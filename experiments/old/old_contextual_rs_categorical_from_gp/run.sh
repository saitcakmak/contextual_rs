#!/bin/bash

N=2;
#
for i in {0..19}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_0 $i -add & done
for i in {0..19}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_00 $i -add & done
for i in {0..19}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_000 $i -add & done
for i in {0..19}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_4 $i -add & done
for i in {0..19}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_5 $i -add & done
#
