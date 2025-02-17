#!/bin/sh

# 0:FC, 1:CNN, 2: weight value mode
config=0
size=12500 # size of batch 
iteration=1000
num_attack=4

start_time=`date +%s`

python3 ./main_train.py $config $size $iteration $num_attack

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
