#!/bin/sh

config=0 # 0:FC, 1:CNN
size=1000 # size of batch 

python3.11 ./main_train.py $config $size
