#!/bin/sh

config=0       # 0:FC, 1:CNN
size=100       # not used in test
idx=0

python3 ./main-train.py $config $size $idx
