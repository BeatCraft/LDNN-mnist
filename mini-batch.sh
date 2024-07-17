#!/bin/sh

#for i in `seq 0 499`
for j in `seq 0 10`
do

for i in `seq 0 59`
do
    echo $i
    python3.11 ./mini_batch_train.py $i
done

done
