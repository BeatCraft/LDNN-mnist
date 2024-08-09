#!/bin/sh

start_time=`date +%s`

for i in `seq 0 3000`
do
    echo $i
    python3.11 ./mini_batch_train2.py $i
done

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time

python3.11 mini_batch_test.py
