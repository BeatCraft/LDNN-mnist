#!/bin/sh

# 0:FC, 1:CNN
config=0

# 0:train, 1:test, ...
exec_mode=4

# weight mode
# 0:wi(uint8), 1:w (float16)

weight_mode=1
# quantaization mode
# 0:32bit, 1:16bit, 2:8bit
qmode=0

# size of batch
size=50

#
# training parameters
#
iteration=1000
num_attack=4
bi=0 # batch index

start_time=`date +%s`

for i in `seq 1 1`
do
    echo "i = $i"
    python3 ./main.py $config $exec_mode $weight_mode $qmode $size $iteration $num_attack $bi
done

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo "elapsed_time:" $elapsed_time
