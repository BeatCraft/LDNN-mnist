#!/bin/sh

config=0 # 0:FC, 1:CNN
size=1000 # size of batch 
iteration=100000
num_attack=4

start_time=`date +%s`

python3.11 ./main_train.py $config $size $iteration $num_attack

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
