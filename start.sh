#!/bin/sh

config=0        # 0:FC, 1:CNN
mode=2          # 0:train, 1:test, 2:ec, 3:ac
size=7500       # not used in test
idx=0

python3 ./main.py $config $mode $size $idx
