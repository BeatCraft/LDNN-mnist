#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
#
#from mpi4py import MPI
#import cupy as cp
#import cupyx

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
if sys.platform.startswith('darwin'):
    import opencl
else:
    if plat.ID==1:
        import opencl
    elif plat.ID==2:
        import dgx
    #
#
import util
import core
import train
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
    print(argc)
    if argc!=3:
        return 0
    #
    
    config_id = int(argvs[1])
    batch_size = int(argvs[2]) # 1000 - 10000?
    
    if plat.ID==0: # MBP
        platform_id = 0
        device_id = 1
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif  plat.ID==1: # tr
        platform_id = 1
        device_id = 0
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif plat.ID==2: # nvidia
        #cp.cuda.Device(0).use()
        my_gpu = dgx.Dgx(1)
    else:
        print("error : undefined platform")
        return 0
    #
    #config_id = 0 # FC
    
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    
    train_data_batch = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_label_batch = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    batch_offset = 0
    
    if config_id==0: # fc
        r = mnist.setup_dnn(my_gpu, config_id, "./wi-fc.csv")
    else:
        return 0
    #
    r.prepare(batch_size, data_size, num_class)
    if config_id==0:
        ce = 0.0
        n = int(mnist.TRAIN_BATCH_SIZE / batch_size)
        for i in range(n):
            r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
            ce += r.evaluate()
            batch_offset += batch_size
        #
        ce = ce / float(n)
        log = "%d, %f" % (0, ce)
        output("./log.csv", log)
        r.save_as("./wi/wi-fc-0000.csv")
        
    #
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
