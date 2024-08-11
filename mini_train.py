#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import pickle
import numpy as np
import csv
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
import core
import train
import batch

# tool
sys.path.append(os.path.join(os.path.dirname(__file__), '../ptool/'))
import tool

import mnist

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)

    if argc!=2:
        print("error : need batch offset index")
        return 0
    #
    config = int(argvs[1])
    #if config==0:
    #    wpath = "./wi-fc.csv"
    #elif config==1:
    #    wpath = "./wi-cnn.csv"
    #else:
    #    return 0
    #

    iteration = 100
    type = 0 # classificattion
    scale = True
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    batch_size = mnist.TRAIN_BATCH_SIZE
    mini_batch_size = 100
    mini_batch_num = int(batch_size / mini_batch_size)
    print(batch_size, mini_batch_size, mini_batch_num)
    
    #
    # batch
    #
    b = batch.Batch(data_size, type, num_class)
    b.load_data(mnist.TRAIN_IMAGE_BATCH_PATH)
    b.load_label(mnist.TRAIN_LABEL_BATCH_PATH)
    b.prepare_batch(scale)
    b.prepare_mini_batch(mini_batch_size)
    
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config)
    if r==None:
        return 0
    #
    
    #r = core.Roster()
    #r.set_gpu(my_gpu)
    
    #if config==0:
    #    mnist.setup_fc(r, data_size)
    #elif config==1:
    #    mnist.setup_cnn(r, data_size)
    #
    #r.set_path(wpath)
    #r.set_scale_input(1)
    #r.load()
    #r.update_weight()
    
    r.prepare(batch_size, data_size, num_class)
    
    t = train.Train(r)
    t.w_list = t.make_w_list()
    
    #
    # random mini-batch
    #
    for n in range(mini_batch_num):
        data_array, label_array = b.get_mini_batch(n)
        r.reset()
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        ce = r.evaluate()
        t.main_simple_loop(n, ce, iteration, 4)
    #
    return 0

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
