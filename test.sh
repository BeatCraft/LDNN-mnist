#!/bin/sh

# 0:FC, 1:CNN, 2: weight value mode
config=0
mode=1
batch_size=1000

python3 ./main.py $config $mode $batch_size

