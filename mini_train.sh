#!/bin/sh

config=0 # 0:FC, 1:CNN
mode=2 #0:train, 1:test, 2:mini batch train
iteration=50
num_attack=64
size=1000 # size of mini batch
num=100 # loop

start_time=`date +%s`

python3 ./main.py $config $mode $iteration $num_attack $size $num

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
