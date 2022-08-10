#!/bin/sh

config=0        # 0:FC, 1:CNN
mode=3          # 0:train, 1:test, 2:ec, 3:ac
size=6250       # not used in test
idx=100

#python3 ./main.py $config $mode $size $idx


for i in `seq 0 100`
do
    python3 ./main.py $config $mode $size $i
done

#for i in 100 250 500 1000 2500 5000 10000 20000 40000
#do
#    python3 ./main.py $config $mode $size $i
#done

