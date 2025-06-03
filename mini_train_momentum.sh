#!/bin/sh

config=0 # 0:FC, 1:CNN
# 0:train
# 1:test
# 2:mini batch train
# 3:train with momentum
# 4:mini batch train with momentum
mode=4
iteration=1
num_attack=4
size=1000 # size of mini batch
num=100 # loop


start_time=`date +%s`

python3 ./main.py $config $mode $iteration $num_attack $size $num

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
