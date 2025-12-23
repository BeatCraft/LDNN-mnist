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
    wmode = int(argvs[3]) # wi or value
    qmode = int(argvs[4])
    batch_size = int(argvs[5])
    print("config:", config)
    print("exec_mode:", exec_mode)
    print("wmode:", wmode)
    print("qmode:", qmode)
    print("batch_size:", batch_size)
    if argc==9:
        iteration = int(argvs[6])
        num_attack = int(argvs[7])
        attack_num = int(argvs[7])
        batch_index = int(argvs[8])
        print("iteration:", iteration)
        print("num_attack:", num_attack)
        print("batch_index:", batch_index)
    #

    #
    # batch
    #
    type = mnist.MODEL_TYPE
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    b = batch.Batch(data_size, type, num_class, True, qmode)
    #print("scale_mode:", b.scale_mode)
        
    if exec_mode==1: # test
        b.setDataPath(mnist.TEST_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TEST_LABEL_BATCH_PATH)
    else: # train
        b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
    #
    b.loadDataAndLebel()
    #print("scale_mode:", b.scale_mode)
    
    #
    # gpu
    #
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config, exec_mode, wmode, qmode, batch_size)
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
        r.save(wmode)
    elif exec_mode==1: # test
        debug = 0
        single = 0
        
        #(data_array, label_list, label_array) = b.get_batch(batch_size, 1000)
        #r.direct_set_data(data_array)
        #r.direct_set_label(label_array)
        #r.propagate(1)
        #ce = r.get_cross_entropy(1)
        #print("CE:", ce)
        #print(data_array[0])
        #return 0
        
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
    elif exec_mode==3: # train bp
        print("exec_mode:", exec_mode)
        t = train.Train(r)
        #t.w_list = t.make_w_list()
        
        r.set_backpropagation(True, 0.01)
        (data_array, label_list, label_array) = b.get_batch(batch_size, batch_index)
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        
        debug = 0
        ce = r.evaluate(debug)
        print("CE:", ce)
        #r.bp(debug)
        #return 0
        
        debug = 0
        for i in range(100): # 65
            ce = r.evaluate(debug)
            print(i, "CE:", ce)
            r.bp(debug)
            r.update_weight()
        #
        r.save(wmode)
    elif exec_mode==4: # train bp with mini batch
        debug = 0
        t = train.Train(r)
        r.set_backpropagation(True, 0.01)
        mini_batch_num = int(b.batch_size / batch_size)

        ce = 10.0
        ce_alt = 0.0
        cnt = 0
        
        (data_array, label_list, label_array) = b.get_batch(batch_size, 0)
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        ce = r.evaluate(debug)
        
        for n in range(mini_batch_num):
            if n>0:
                r.reset()
                (data_array, label_list, label_array) = b.get_batch(batch_size, n*batch_size)
                r.direct_set_data(data_array)
                r.direct_set_label(label_array)
            #
            for i in range(10):
                ce_alt = r.evaluate(debug)
                print("(%d/%d)" % (cnt, mini_batch_num), n, i, "CE:", ce_alt)
                r.bp(debug)
                r.update_weight()
            #
            ce = ce_alt

            cnt += 1
        #
        r.save(wmode)
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

