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
    
    exec_mode = 0 # train:0, test:1
    batch_quantize_mode = 0 # sacle only:0, fixed float:1, index:2
    mode_q = 0 # weight only:0, full:1
    
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
        exec_mode = 1
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
    elif mode==3: # train with momoentum
        if argc!=6:
            print("error", argc)
            return 0
        #
        iteration = int(argvs[3])
        num_attack = int(argvs[4])
        batch_size = int(argvs[5])
        print("train with momoentum")
    elif mode==4: # mini batch train with momentum
        if argc!=7:
            print("error", argc)
            return 0
        #
        iteration = int(argvs[3])
        num_attack = int(argvs[4])
        batch_size = int(argvs[5])
        loop = int(argvs[6])
        print("mini batch train")
    elif mode==5: # test for full quantozation
        if argc!=4:
            print("error", argc)
            return 0
        #
        batch_size = int(argvs[3])
        batch_quantize_mode = 2
        mode_q = 1
        exec_mode = 1
        print("test for full quantozation")
    elif mode==6: # train
        if argc!=6:
            print("error", argc)
            return 0
        #
        iteration = int(argvs[3])
        num_attack = int(argvs[4])
        batch_size = int(argvs[5])
        batch_quantize_mode = 2
        mode_q = 1
        print("full quantization train")
    else:
        print("mode error")
    #
    
    #
    # batch
    #
    type = 0 # classification
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    # batch_quantize_mode
    # 0 : float, from -1.0 to 1.0
    # 1 : fixed values of float
    # 2 : index to float table
    b = batch.Batch(data_size, type, num_class, exec_mode, batch_quantize_mode)
    b.train_data_path = mnist.TRAIN_IMAGE_BATCH_PATH
    b.train_label_path = mnist.TRAIN_LABEL_BATCH_PATH
    b.test_data_path = mnist.TEST_IMAGE_BATCH_PATH
    b.test_label_path = mnist.TEST_LABEL_BATCH_PATH
    b.load_mode()
    b.scale()
    b.quantize()
    b.label_list_to_one_hot_vector()
    
    #
    # gpu
    #
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config, mode_q, batch_size)
    if r==None:
        return 0
    #
    
    start_time = time.time()
    if mode==0: # train
        data_array, label_array = b.get_batch(batch_size, 0)
        t = train.Train(r)
        t.w_list = t.make_w_list()
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)

        ce = r.evaluate(0)

        num_attack_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        na_idx = 0
        for na in num_attack_list:
            loop_cnt = 0
            while 1:
                ce, hit_rate = t.main_simple_loop(0, 0, ce, 100, na)
                if hit_rate<0.05 or loop_cnt>32 or ce<0.000001:
                    break
                #
                loop_cnt += 1
            #
        #
        
        #for i in range(128):
        #    na = num_attack_list[na_idx]
        #    ce, hit_rate = t.main_simple_loop(0, 0, ce, 100, na)
        #    if hit_rate<0.05:
        #        na_idx += 1
        #    #
        #
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
    elif mode==3: # train with momentum
        data_array, label_array = b.get_batch(batch_size, 4000)
        t = train.Train(r)
        t.w_list = t.make_w_list()
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        
        for idx in range(iteration):
            #t.momentum_loop(idx, 0, 10, num_attack)
            #num_attack2 = 4
            #t.auto_momentum_loop(idx, 0, 10, num_attack2)
            #r.save()
            attack_num = 64
            attack_list = t.momentum_challenge(idx, 0, 100, attack_num)
        #
        r.save()
    elif mode==4: # mini batch train with momentum
        t = train.Train(r)
        t.w_list = t.make_w_list()
        b.prepare_mini_batch(batch_size)
        for l in range(loop):
            for n in range(b.mini_batch_num):
                data_array, label_array = b.get_mini_batch(n*batch_size)
                r.reset()
                r.direct_set_data(data_array)
                r.direct_set_label(label_array)
                
                for i in range(1):
                    attack_list = t.momentum_challenge(l, n, iteration, num_attack)
                    ret = t.auto_momentum_challenge(l, n, iteration, attack_list, num_attack)
                #
                
                #t.momentum_loop(l, n, iteration, num_attack)
                #num_attack2 = 4
                #t.auto_momentum_loop(l, n, iteration, num_attack2)
                r.save()
            #
            #b.shuffle_mini_batch()
        #
    
    elif mode==5: # full quantization test
        debug = 0
        single = 0
        ac = exam.classification(r, b, 1000, debug, single)
        print(ac)
        
    elif mode==6:
        data_array, label_array = b.get_batch(batch_size, 0)
        t = train.Train(r)
        t.w_list = t.make_w_list()
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        
        
        ce = r.evaluate(0)
        t.main_simple_loop(0, 0, ce, iteration, num_attack)
        
        #iteration = 5
        #for idx in range(10000):
        #    num_attack = 4
        #    t.momentum_loop(idx, 0, iteration, num_attack)
        #    num_attack2 = 4
        #    t.auto_momentum_loop(idx, 0, iteration, num_attack2)
        #    r.save()
        #
    else:
        print("main()::mode error")
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

