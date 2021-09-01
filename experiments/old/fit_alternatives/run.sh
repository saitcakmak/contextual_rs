#!/bin/bash

N=2; for i in {0..99}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_1 $i & done

N=2; for i in {0..99}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_2 $i & done

N=2; for i in {0..99}; do ((j=j%N)); ((j++==0)) && wait; python main.py config_3 $i & done

