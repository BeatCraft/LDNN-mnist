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
#import exam
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
    #batch_size = 100
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    batch_offset = 0
    
    test_batch_data = util.pickle_load("./work/1-test_data.pickle")
    test_batch_label = util.pickle_load("./work/1-test_label.pickle")
    batch_size = len(test_batch_label)
    
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
    r = mnist.setup_dnn(my_gpu, config, "./work/1-wi-cnn.csv")
    if r:
        pass
    else:
        return 0
    #
    
    correct, incorrect = dhandle.test_loop(r, test_batch_data, test_batch_label, batch_size, data_size, num_class, batch_offset)
    
    print(len(correct), len(incorrect))
    
    dh = dhandle.DataHandler(data_size, num_class)
    dh.load("./work/1-test_data.pickle", "./work/1-test_label.pickle")
    
    correct_data, correct_label = dh.extract(correct)
    dh.saveBatch(correct_data, correct_label, "./work/1-correct_data.pickle", "./work/1-correct_label.pickle")
    
    incorrect_data, incorrect_label = dh.extract(incorrect)
    dh.saveBatch(incorrect_data, incorrect_label, "./work/1-incorrect_data.pickle", "./work/1-incorrect_label.pickle")
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
