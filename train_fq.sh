#!/bin/sh

# 0:FC, 1:CNN, 2: weight value mode
config=0
# 0:train, 1:test, 2:mini batch train, 3:train with momentum
mode=6
iteration=100000
num_attack=4
size=1000 # size of batch


start_time=`date +%s`

python3 ./main.py $config $mode $iteration $num_attack $size

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
