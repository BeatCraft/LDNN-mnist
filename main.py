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
import exam
import train

sys.path.append(os.path.join(os.path.dirname(__file__), '../ptool'))
import batch

import mnist

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    if argc<3:
        print("error", argc)
    #
    config = int(argvs[1])
    mode = int(argvs[2])
    print("config=%d, mode=%d" % (config, mode))
    if mode==0: # train
        if argc!=6:
            print("error", argc)
            return 0
        #
        iteration = int(argvs[3])
        num_attack = int(argvs[4])
        batch_size = int(argvs[5])
        print("train")
    elif mode==1: # test
        if argc!=4:
            print("error", argc)
            return 0
        #
        batch_size = int(argvs[3])
        print("test")
    elif mode==2: # mini batch train
        if argc!=7:
            print("error", argc)
            return 0
        #
        iteration = int(argvs[3])
        num_attack = int(argvs[4])
        batch_size = int(argvs[5])
        loop = int(argvs[6])
        print("mini batch train")
    else:
        print("mode error")
    #
    
    #
    # batch
    #
    type = 0 # classification
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    quantize = 2
    b = batch.Batch(data_size, type, num_class, mode, quantize)
    b.train_data_path = mnist.TRAIN_IMAGE_BATCH_PATH
    b.train_label_path = mnist.TRAIN_LABEL_BATCH_PATH
    b.test_data_path = mnist.TEST_IMAGE_BATCH_PATH
    b.test_label_path = mnist.TEST_LABEL_BATCH_PATH
    b.load_mode()
    b.scale()
    b.quantize()
    b.label_list_to_one_hot_vector()
    
    #batch_offset = 0
    #
    # gpu
    #
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config, batch_size)
    if r==None:
        return 0
    #
    
    if mode==0: # train
        data_array, label_array = b.get_batch(batch_size, 0)
        t = train.Train(r)
        t.w_list = t.make_w_list()
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        ce = r.evaluate()
        t.main_simple_loop(0, 0, ce, iteration, num_attack)
    elif mode==1: # test
        debug = 0
        single = 0
        ac = exam.classification(r, b, 1000, debug, single)
        print(ac)
    elif mode==2: # mini batch train
        t = train.Train(r)
        t.w_list = t.make_w_list()
        #
        # mini-batch
        #
        b.prepare_mini_batch(batch_size)
        #b.mini_batch_size = batch_size
        #mini_batch_num = int(b.batch_size / b.mini_batch_size)
        
        for l in range(loop):
            for n in range(b.mini_batch_num):
                data_array, label_array = b.get_mini_batch(n*batch_size)
                r.reset()
                r.direct_set_data(data_array)
                r.direct_set_label(label_array)
                ce = r.evaluate()
                t.main_simple_loop(l, n, ce, iteration, num_attack)
            #
            b.shuffle_mini_batch()
        #
    else:
        print("mode error")
    #
    
    return 0
    
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)

