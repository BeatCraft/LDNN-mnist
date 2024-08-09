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
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
#import util
import plat
import core
import train
#import opencl

import mnist

# tool
sys.path.append(os.path.join(os.path.dirname(__file__), '../ptool/'))
import tool

BATCH_DATA_ARRAY = tool.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
BATCH_LABEL_LSIT = tool.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
LABEL_ARRAY = np.array(BATCH_LABEL_LSIT, np.float32)

    
def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)

    if argc!=2:
        print("error : need batch offset index")
        return 0
    #

    batch_offset_index = int(argvs[1])
    print("batch_offset_index", batch_offset_index)
    
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    batch_size = mnist.TRAIN_BATCH_SIZE
    mini_batch_size = 1000
    batch_num = int(batch_size/ mini_batch_size)
    print(batch_size, mini_batch_size, batch_num)
    
    wpath = "./wi-cnn.csv"
    my_gpu = plat.getGpu()
    
    r = core.Roster()
    r.set_gpu(my_gpu)
    mnist.setup_cnn(r, data_size)
    
    r.set_path(wpath)
    r.set_scale_input(1)
    r.load()
    r.update_weight()
    r.prepare(mini_batch_size, data_size, num_class)
    
    #w_list = make_w_list_for_mini_batch(r, batch_num)
    #print(len(w_list), len(w_list[0]))
    
    # data
    batch_offset = mini_batch_size * batch_offset_index
    data_array = BATCH_DATA_ARRAY[batch_offset:(batch_offset + mini_batch_size)]
    for i in range(mini_batch_size): # scale to 0.0 - 1.0
        data_array[i] = data_array[i] / 255.0
    #
    r.direct_set_data(data_array)
    
    # label
    labels = LABEL_ARRAY[batch_offset:(batch_offset + mini_batch_size)]
    label_array = np.zeros((mini_batch_size, num_class), dtype=np.float32)
    for j in range(mini_batch_size):
        labels[j]
        k = int(labels[j])
        label_array[j][k] = 1.0
    #
    r.direct_set_label(label_array)
    
    ce = r.evaluate()
    print(ce)
    #return 0
    
    t = train.Train(r)
    t.w_list = t.make_w_list()
    #w_list[idx] #t.make_w_list()
    t.main_simple_loop(batch_offset_index, ce, 10000, 4)
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
