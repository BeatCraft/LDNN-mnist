#!/bin/sh

config=0 # 0:FC, 1:CNN
size=10000 # size of mini batch

start_time=`date +%s`

for i in `seq 0 100`
do
    echo "________"
    echo $i
    echo "________"
    python3 ./mini_train.py $config $size $i

    end_time=`date +%s`
    elapsed_time=$((end_time - start_time))
    echo "elapsed_time"
    echo  $elapsed_time
done
