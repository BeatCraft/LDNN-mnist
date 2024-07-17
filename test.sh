#!/bin/sh

config=0    # 0:FC, 1:CNN, 2:FCNN
log=0
idx=0

python3.11 ./main-test.py $config $idx $log
