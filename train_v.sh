#!/bin/sh

# 0:FC, 1:CNN, 2: weight value mode
config=2
size=4 # size of batch
iteration=10
num_attack=4

start_time=`date +%s`

for i in {1..100} ; do
    echo ${i}
    python3 ./train_bp.py $config $size $iteration $num_attack
done

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
