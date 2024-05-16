#!/bin/bash


# train phase
CUDA_VISIBLE_DEVICES=0 python trainer/uganConsisTrainer.py -p train -f 0

CUDA_VISIBLE_DEVICES=0 python trainer/uganConsisTrainer.py -p test -f 0 -i 000 -wh best
