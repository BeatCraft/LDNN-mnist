#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import random
import csv

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
    if argc<5:
        print("error", argc)
    #
    config = int(argvs[1])
    exec_mode = int(argvs[2])
    wmode = int(argvs[3]) # index or float
    qmode = 0
    batch_size = int(argvs[4])
    if argc==7:
        iteration = int(argvs[5])
        num_attack = int(argvs[6])
        batch_index = int(argvs[27])
    #

    #
    # batch
    #
    type = mnist.MODEL_TYPE
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    b = batch.Batch(data_size, type, num_class)
    if exec_mode==0: # train
        b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
    else: # test
        b.setDataPath(mnist.TEST_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TEST_LABEL_BATCH_PATH)
    #
    b.loadDataAndLebel()
        
    #
    # gpu
    #
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config, wmode, qmode, batch_size)
    if r==None:
        return 0
    #
    
    start_time = time.time()
        
    if exec_mode==0: # train
        t = train.Train(r)
        t.w_list = t.make_w_list()
        
        (data_array, label_list, label_array) = b.get_batch(batch_size, batch_index)
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)

        ce = r.evaluate(0)
        #print("CE:", ce)
        loop_max = 100
        for i in range(iteration):
            ce, hit_rate = t.main_challenge_loop(ce, loop_max, attack_num, True)
        #
        #num_attack_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        #for na in num_attack_list:
        #    loop_cnt = 0
        #    while 1:
        #        ce, hit_rate = t.main_simple_loop(0, 0, ce, 100, na)
        #        if hit_rate<0.05 or loop_cnt>32 or ce<0.000001:
        #            break
        #        #
        #        loop_cnt += 1
        #    #
        #
    elif exec_mode==1: # test
        debug = 0
        single = 0
        ac = exam.classification(r, b, 1000, debug, single)
        print(ac)
    elif exec_mode==2: # mini batch train
        ce_list = []
        #sum_ce = 0.0
        mini_batch_num = int(b.batch_size / batch_size)
        for n in range(mini_batch_num):
            (data_array, label_list, label_array) = b.get_batch(batch_size, n*batch_size)
            r.reset()
            r.direct_set_data(data_array)
            r.direct_set_label(label_array)
        
            ce = r.evaluate()
            #print(ce)
            ce_list.append((n, ce))
        #
        sorted_data = sorted(ce_list, key=lambda x: x[1], reverse=False)
        
        t = train.Train(r)
        t.w_list = t.make_w_list()
        
        #for i in range(10):
        for sd in sorted_data:
            bi = sd[0]
            (data_array, label_list, label_array) = b.get_batch(batch_size, bi)
            r.reset()
            r.direct_set_data(data_array)
            r.direct_set_label(label_array)
            
            ce = r.evaluate(0)
            #ce, hit_rate = t.main_simple_loop(0, 0, ce, 100, 64)
            loop_max = 100
            ce, hit_rate = t.main_challenge_loop(ce, loop_max, attack_num, True)
        #
        r.save()
    elif exec_mode==3: # bp test
        t = train.Train(r)
        t.w_list = t.make_w_list()
        
        r.set_backpropagation(True)
        debug = 0
        
        (data_array, label_list, label_array) = b.get_batch(batch_size, batch_index)
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        for i in range(10):
            ce = r.evaluate(debug)
            print(i, "CE:", ce)
            r.bp(debug)
            r.update_weight()
        #
        ce  = r.evaluate(debug)
        print("CE:", ce)
        r.save(wmode)
        #r.save_as("./w-fc.csv", 1)
        return 0
                
        mini_batch_num = int(b.batch_size / batch_size)
        for n in range(mini_batch_num):
            (data_array, label_list, label_array) = b.get_batch(batch_size, n*batch_size)
            r.reset()
            r.direct_set_data(data_array)
            r.direct_set_label(label_array)
        
            ce = r.evaluate(debug)
            print(n, "CE:", ce)
            r.bp(debug)
            r.update_weight()
        #
        return 0
    #
    
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print(("time = %s" % (t)))
    
    return 0
    
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)

