#!/bin/sh

# 0:FC, 1:CNN, 2: weight value mode
config=2
size=1 # size of batch
iteration=1
num_attack=4
bpd=1
ss=0
debug=0

start_time=`date +%s`

#for i in {1..10000} ; do
#    echo ${i}
    python3 ./train_bp.py $config $size $iteration $num_attack $bpd $ss $debug
#done

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time"
echo  $elapsed_time
