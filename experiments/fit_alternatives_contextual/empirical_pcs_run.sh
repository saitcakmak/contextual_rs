#!/bin/bash

N=2;

for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait;
python add_empirical_pcs.py config_0 $i & done

for i in {0..49}; do ((j=j%N)); ((j++==0)) && wait;
python add_empirical_pcs.py config_1 $i & done


