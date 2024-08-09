#!/bin/sh

start_time=`date +%s`

for j in `seq 0 10`
do

for i in `seq 0 59`
do
    echo $i
    python3.11 ./mini_batch_train.py $i
done

done

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo $elapsed_time
