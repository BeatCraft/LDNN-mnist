#!/bin/sh

# 0:FC, 1:CNN
config=0

# 0:train, 1:test, ...
exec_mode=1

# weight mode
# 0:index(uint8), 1:value/float for backpropagation
wmode=0

# quantaization mode
# 0:32bit, 1:16bit, 2:8bit
qmode=0

# size of batch
size=1000


start_time=`date +%s`

python3 ./main.py $config $exec_mode $wmode $qmode $size

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
