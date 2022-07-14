#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
#
from mpi4py import MPI
import cupy as cp
import cupyx

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import util
import core
import dgx
import train
import mpi
import exam
import mnist

sys.setrecursionlimit(10000)

def output(path, msg):
    with open(path, 'a') as f:
        print(msg, file=f)
    #
    
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    config_id = 0 # FC
    type_id = 1
    platform_id = 0
    device_id = 0
    config = 1
    mode = 1
    
    if type_id==0:
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif type_id==1:
        my_gpu = dgx.Dgx(1)
    else:
        print("error : undefined type =%d" % (type_id))
        return 0
    #
    
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    batch_image = util.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
    batch_label = util.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
    batch_size = mnist.TEST_BATCH_SIZE
    batch_offset = 0
    
    if config==0:
        idx = 0
        path = "./wi/wi-fc-%04d.csv" % (idx)
        r = mnist.setup_dnn(my_gpu, config, path)
    else:
        return 0
    #
    
    ac = exam.classification(r, data_size, num_class, batch_size, batch_image, batch_label, 1000)
    print(ac)
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
