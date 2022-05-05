#!/bin/sh

type_id=0       # 0:OpenCL, 1:CuPy/GDX
platform_id=0   # MBP : platform_id=0 and device_id=1
device_id=1     # tr  : platform_id=1 and device_id=0
config=2        # 0:FC, 1:CNN
mode=0          # 0:train, 1:test
size=1000       # not used in test

python3 ./main.py $type_id $platform_id $device_id $config $mode $size
