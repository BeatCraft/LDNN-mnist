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
    print(argvs)
    print(argc)
    if argc==4:
        pass
    else:
        print("error in sh")
        return 0
    #
    config = int(argvs[1])
    batch_size = int(argvs[2])
    idx = int(argvs[3])

    batch_offset = 0
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
        
    print("config=%d" % (config))
    print("batch_size=%d" % (batch_size))
    
    if config==0:
        wpath = "./wi-fc.csv"
    elif config==1:
        wpath = "./wi-cnn.csv"
    else:
        return 0
    #
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config, wpath)
    if r:
        pass
    else:
        return 0
    #
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    
    train_batch_image = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_batch_label = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    
    t = train.Train(r)
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_batch_image, train_batch_label, batch_size, 0)

    w_list = t.make_w_list([core.LAYER_TYPE_CONV, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
    
    if config==0: # all
        #for i in range(1):
        #    t.logathic_loop(i, w_list, "all")
        #
        temperature = 100.0
        total = 10000
        debug = 0
        t.loop_sa(idx, w_list, "all", temperature, total, debug)
    elif config==1:
        temperature = 100.0
        total = 10000
        debug = 0
        t.loop_sa(idx, w_list, "all", temperature, total, debug)
    #
    r.save_as(wpath)
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
