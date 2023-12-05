#!/bin/sh

config=2    # 0:FC, 1:CNN, 2:FCNN
log=0
idx=0

python3 ./main-test.py $config $idx $log
