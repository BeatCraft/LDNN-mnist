#! /usr/bin/python
# -*- coding: utf-8 -*-
#

import os
import sys
import time
import random
import numpy as np

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
import util
import core
#import train
import exam
import dhandle
    
if sys.platform.startswith('darwin'):
    import opencl
else:
    if plat.ID==1:
        import opencl
    elif plat.ID==2:
        import dgx
    #
#

import mnist
        
def main():
    argvs = sys.argv
    argc = len(argvs)
    
    config = 1 # CNN
    mode = 0
    batch_size_full = mnist.TRAIN_BATCH_SIZE
    batch_size = 100
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    batch_offset = 0
    
    dh = dhandle.DataHandler(data_size, num_class)
    dh.load("./work/2-incorrect_data.pickle", "./work/2-incorrect_label.pickle")
    dh.shuffle()
    
    head, tail = dh.divide(10)
    print(len(head), len(tail))
    
    h1d = dh.flatten(head)
    t1d = dh.flatten(tail)
    print(len(h1d), len(t1d))
    
    train_batch_data, train_batch_label = dh.makeBatch(h1d)
    test_batch_data, test_batch_label = dh.makeBatch(t1d)
    
    dh.saveBatch(train_batch_data, train_batch_label, "./work/3-train_data.pickle", "./work/3-train_label.pickle")
    dh.saveBatch(test_batch_data, test_batch_label, "./work/3-test_data.pickle", "./work/3-test_label.pickle")
    
    if plat.ID==0:   # MBP
        my_gpu = opencl.OpenCL(0, 1)
        my_gpu.set_kernel_code()
    elif plat.ID==1: # TR
        my_gpu = opencl.OpenCL(1, 0)
        my_gpu.set_kernel_code()
    elif plat.ID==2: # nvidia
        my_gpu = dgx.Dgx(1)
    else:
        print("error : undefined platform")
        return 0
    #
    r = mnist.setup_dnn(my_gpu, config, "./work/3-wi-cnn.csv")
    if r:
        pass
    else:
        return 0
    #

    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_batch_data, train_batch_label, batch_size, 0)    
    dhandle.train_loop(r)
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
