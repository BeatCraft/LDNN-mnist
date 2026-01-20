#!/bin/sh

# 0:FC, 1:CNN
config=1

# 0:train, 1:test, ...
exec_mode=3

# weight mode
# 0:wi(uint8), 1:value(float)
weight_mode=1
# quantaization mode
# 0:32bit, 1:16bit, 2:8bit
qmode=0

# size of batch
size=1000

#
# training parameters
#
iteration=1000
num_attack=4
bi=0 # batch index

start_time=`date +%s`

python3 ./main.py $config $exec_mode $weight_mode $qmode $size $iteration $num_attack $bi

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
