#!/bin/sh

# 0:FC, 1:CNN
config=0

# 0:train
# 1:test
# 2:train w/minibatch
# 3:train w/backpropagation
# 4:train w/backpropagation w/minibatch
exec_mode=5

# weight mode
# 0:weight index(uint8), 1:value/float for backpropagation
wmode=0

# quantaization mode
# 0:32bit, 1:16bit, 2:8bit
qmode=0

# size of batch
size=60000

#
# training parameters
#
iteration=100
num_attack=4
bi=0 # batch index
undo=0

start_time=`date +%s`

#for i in 1024 1024 1024 1024 1024 512 512 512 512 512 256 256 256 256 256 128 128 128 128 128
for i in 512 512 512 256 256 256 128 128 128
#for i in 128 128 128 64 64 64
do
    num_attack=$i
    python3 ./main.py $config $exec_mode $wmode $qmode $size $iteration $num_attack $bi
done
#for i in `seq 1 1000`
#do
#    echo "i = $i"
#    python3 ./main.py $config $exec_mode $wmode $qmode $size $iteration $num_attack $bi
#done

end_time=`date +%s`
elapsed_time=$((end_time - start_time))
echo  $elapsed_time

hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))
printf "Elapsed time: %02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
