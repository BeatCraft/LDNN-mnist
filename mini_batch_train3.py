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
    batch_size = 100
    batch_num = int(mnist.TRAIN_BATCH_SIZE / batch_size)
    print(mnist.TRAIN_BATCH_SIZE, batch_size, batch_num)
    
    wpath = "./wi.csv"
    my_gpu = plat.getGpu()
    
    r = core.Roster()
    r.set_gpu(my_gpu)
    mnist.setup_fc(r, mnist.IMAGE_SIZE)
    
    r.set_path(wpath)
    r.set_scale_input(1)
    r.load()
    r.update_weight()
    r.prepare(batch_size, data_size, num_class)
    t = train.Train(r)
    t.w_list = t.make_w_list()
    #
    # random mini-batch
    #
    data_array = np.zeros((batch_size, data_size), dtype=np.float32)
    label_array = np.zeros((batch_size, num_class), dtype=np.float32)
    idx_list = []
    for i in range(mnist.TRAIN_BATCH_SIZE):
        idx_list.append(i)
    #
    
    for n in range(batch_num):
        #print(len(idx_list))
        label_array = np.zeros((batch_size, num_class), dtype=np.float32)
        mini_idx_list = random.sample(idx_list, batch_size)
        j = 0
        for i in mini_idx_list:
            idx = random.randrange(mnist.TRAIN_BATCH_SIZE)
            data_array[j] = BATCH_DATA_ARRAY[idx] / 255.0
            # scale to 0.0 - 1.0
            k = int(LABEL_ARRAY[idx])
            label_array[j][k] = 1.0
            j = j + 1
        #
        r.reset()
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        ce = r.evaluate()
        #print(ce)
        t.main_simple_loop(n, ce, 100, 4)
        #
        for i in mini_idx_list:
            idx_list.remove(i)
        #
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
