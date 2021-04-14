#!/bin/bash

N=2;
#
for i in {0..19}; do ((j=j%N)); ((j++==0)) && wait;
python main.py config_b_0 LCEGP $i & done

#
