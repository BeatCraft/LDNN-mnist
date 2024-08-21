#!/bin/sh

config=0 # 0:FC, 1:CNN
size=1000 # size of batch 

start_time=`date +%s`

python3.11 ./main_train.py $config $size

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
