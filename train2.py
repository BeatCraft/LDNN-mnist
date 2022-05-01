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
import train
#import exam
#import dhandle
    
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
    #batch_size_full = mnist.TRAIN_BATCH_SIZE
    batch_size = 200
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    batch_offset = 0
    
    batch_data0 = util.pickle_load("./work/0-train_data.pickle")
    batch_label = util.pickle_load("./work/0-train_label.pickle")
    
    batch_data1 = util.pickle_load("./work/1-train_data.pickle")
    batch_label1 = util.pickle_load("./work/1-train_label.pickle")
    
    batch_data = np.concatenate([batch_data0, batch_data1])
    print(batch_data.shape)
    batch_label += batch_label1
    print(len(batch_label))
    #return 0
    
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
    r = mnist.setup_dnn(my_gpu, config, "./work/wi-cnn.csv")
    if r:
        pass
    else:
        return 0
    #

    t = train.Train(r)
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, batch_data, batch_label, batch_size, batch_offset)
    
    fc_w_list = t.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
    cnn_w_list = t.make_w_list([core.LAYER_TYPE_CONV_4])
    
    for idx in range(50):
        t.mode_w = 1
        r.propagate()
        for i in range(1, 5): # FC
            layer = r.get_layer_at(i)
            layer.lock = True
        #
        t.loop_k(fc_w_list, "fc", idx, 1, 50)
            
        t.mode_w = 2
        for i in range(1, 5): #CNN
            layer = r.get_layer_at(i)
            layer.lock = False
        #
        t.loop_k(cnn_w_list, "cnn", idx, 1, 20)
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
