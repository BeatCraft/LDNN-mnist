#!/bin/sh

config=0        # 0:FC, 1:CNN
size=65250       # mini batch size

python3 ./ce.py  $config $size
