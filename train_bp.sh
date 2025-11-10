#!/bin/sh

# 0:FC, 1:CNN, 2: weight value mode
config=0
# 0:train, 1:test, 2:mini batch train, 3:bp train test
mode=3
iteration=1000
num_attack=4
size=100 # size of batch
bi=2

python3 ./main.py $config $mode $iteration $num_attack $size $bi
