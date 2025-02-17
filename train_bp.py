#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
import util
import core
import train
import batch

import mnist

    
def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    if argc==8:
        pass
    else:
        print("error in sh")
        return 0
    #
    config = int(argvs[1])
    batch_size = int(argvs[2])
    iteration = int(argvs[3])
    num_attack = int(argvs[4])
    bp_debug = int(argvs[5])
    save_switch = int(argvs[6])
    debug = int(argvs[7])
        
    batch_offset = 0
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    
    scale = True
    print("config=%d" % (config))

    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config)
    if r==None:
        return 0
    #
    
    #
    # batch
    #
    b = batch.Batch(data_size, type, num_class)
    b.load_data(mnist.TRAIN_IMAGE_BATCH_PATH)
    b.load_label(mnist.TRAIN_LABEL_BATCH_PATH)
    b.prepare_batch(scale)
    data_array, label_array = b.get_batch(batch_size)
    #print(data_array)
    #b.prepare_mini_batch(batch_size)
    #data_array, label_array = b.get_mini_batch(0)
    
    #
    # train
    #
    t = train.Train(r)
    r.prepare(batch_size, data_size, num_class)
    r.direct_set_data(data_array)
    r.direct_set_label(label_array)

    w_list = t.make_w_list([core.LAYER_TYPE_CONV, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
    t.w_list = w_list
    
    ce = r.evaluate(0)
    print(ce)
    #return 0
    
    for i in range(iteration):
        r.backpropagate(ce, bp_debug, debug)
        #break
        #r.update_weight()
        ce2 = r.evaluate(0)
        print("[%04d]" %(i), ce, ">", ce2)
        ce = ce2
        if ce<0.000001:
            break
        #
    #
    #return 0
    
    hl = r.get_layer_at(1)
    print("hidden 1", hl._output_array[0])
    #print("hidden 1 w", hl._weight_matrix[0])
        
    hl = r.get_layer_at(2)
    print("hidden 2 o", hl._output_array[0])
    #print("hidden 2 w", hl._weight_matrix[0])
    
    hl = r.get_layer_at(3)
    print("output", hl._output_array[0])
    print("output, softmax", hl._softmax_array)
    
    if save_switch:
        r.save_as("./w.csv", 1)
    #
    return 0
    
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
    
