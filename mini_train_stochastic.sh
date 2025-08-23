#!/bin/sh

config=0 # 0:FC, 1:CNN
mode=7 # 0:train, 1:test, 2:mini batch train
iteration=50
num_attack=64
size=200 # size of mini batch
num=100 # loop

start_time=`date +%s`

for i in $(seq 1 100); do
    echo "Number: $i"
    python3 ./main.py $config $mode $iteration $num_attack $size $num
done

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
