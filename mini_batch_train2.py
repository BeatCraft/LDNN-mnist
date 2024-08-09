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

def make_w_list_for_mini_batch(r, num, type_list=None):
    #r = self._r
    if type_list is None:
        type_list = [core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT]
    #
    w_list  = []
    for i in range(num):
        mini_list = []
        w_list.append(mini_list)
    #
    
    c = r.count_layers()
    for li in range(1, c):
        layer = r.get_layer_at(li)
        type = layer.get_type()
        ret = type in type_list
        if ret is False:
            continue
        #
        
        for ni in range(layer._num_node):
            for ii in range(layer._num_input):
                #bi = ii / num
                q, bi = divmod(ii, num)
                w_list[bi].append(layer.getWeight(ni, ii))
                #w_list.append(layer.getWeight(ni, ii))
            #
        #
    #
    return w_list
        
def mini_train(r, w_list, size, idx, ce, loop_max=1000, attack_num=4, debug=0):
    # set a mini batch
    print("batch_size:", size)
    batch_size = size
    batch_offset = batch_size * idx
    data_array = BATCH_DATA_ARRAY[batch_offset:(batch_offset + batch_size)]
    for i in range(size): # scale to 0.0 - 1.0
        data_array[i] = data_array[i] / 255.0
    #
    r.direct_set_data(data_array)
    label_array = LABEL_ARRAY[batch_offset:(batch_offset + batch_size)]
    r.direct_set_label(label_array)
    
    ce = r.evaluate(1)
    print(ce)
    return 0
    
    
    t = train.Train(r)
    t.w_list = w_list[idx] #t.make_w_list()
    #print("t.w_list", len(t.w_list))
    attack_num = 2
    ce = t.mini_batch_loop(idx, ce, loop_max, attack_num, debug)
    return ce
    
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
    batch_size = 1000
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
    
    #
    # random mini-batch
    #
    data_array = label_array = np.zeros((batch_size, data_size), dtype=np.float32)
    label_array = np.zeros((batch_size, num_class), dtype=np.float32)
    for i in range(batch_size):
        idx = random.randrange(mnist.TRAIN_BATCH_SIZE)
        data_array[i] = BATCH_DATA_ARRAY[idx] / 255.0 # scale to 0.0 - 1.0
        k = int(LABEL_ARRAY[idx])
        label_array[i][k] = 1.0
    #
    
    r.direct_set_data(data_array)
    r.direct_set_label(label_array)
    
    ce = r.evaluate()
        
    t = train.Train(r)
    t.w_list = t.make_w_list()
    t.main_simple_loop(batch_offset_index, ce, 100, 4)
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
