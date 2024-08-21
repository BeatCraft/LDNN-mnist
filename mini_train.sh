#!/bin/sh

config=0 # 0:FC, 1:CNN
size=100 # size of mini batch

start_time=`date +%s`

for i in `seq 0 10`
do
    echo "________"
    echo $i
    echo "________"
    python3.11 ./mini_train.py $config $size $i

    end_time=`date +%s`
    elapsed_time=$((end_time - start_time))
    echo "elapsed_time"
    echo  $elapsed_time
done
