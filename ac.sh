#!/bin/sh

config=0        # 0:FC, 1:CNN
size=1000       # mini batch size
index=0

python3 ./ac.py  $config $size $index

