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

def exec_train_slope(b, my_gpu, r, wmode, batch_size, attack_num, iteration):
    print("exec_train_slope()", batch_size)
    r.set_backpropagation(True, 0.005)
        
    b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
    b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
    print(b.loadDataAndLebel())
    
    (data_array, label_list, label_array) = b.get_batch(batch_size, 0)
    r.direct_set_data(data_array)
    r.direct_set_label(label_array)
    
    t = train.Train(r)
    t.w_list = t.make_w_list()
    
    ce = r.evaluate(0)
    #print(ce)
    print("+++", ce)
    r.slope(0)
    
    #for w in t.w_list:
    #    li = w.li
    #    ni = w.ni
    #    ii = w.ii
    #    l = r.get_layer_at(li)
    #    slope = l.dW[ii][ni]
    #    w.slope = slope
    #
    
    #for n in range(10):
    cnt = 0
    while cnt<attack_num:
        k = random.randint(0, len(t.w_list)-1)
        w = t.w_list[k]
        li = w.li
        ni = w.ni
        ii = w.ii
        l = r.get_layer_at(li)
        wi = w.wi
        #print(li, ni, ii, l.dW[ii][ni], core.WEIGHT_SET[wi])
        
        if l.dW[ii][ni]==0.0:
            pass
        elif l.dW[ii][ni]<0.0: # ++
            if wi==core.WEIGHT_INDEX_MAX:
                pass
            else:
                l.set_weight_index(w.ni, w.ii, wi+1)
                #cnt += 1
            #
        elif l.dW[ii][ni]>0.0: # --
            if wi==core.WEIGHT_INDEX_MIN:
                pass
            else:
                l.set_weight_index(w.ni, w.ii, wi-1)
                #cnt += 1
            #
        #
        cnt += 1
    #
    r.update_weight()
    ce = r.evaluate(0)
    print("+++", ce)
    r.save(wmode)
        
    return
    
    print("CE:", ce)
    loop_max = 100
    for i in range(iteration):
        ce, hit_rate = t.main_challenge_loop(ce, loop_max, attack_num, True)
    #
    r.save(wmode)

def exec_train_bp_mini(b, my_gpu, r, wmode, batch_size):
    print("exec_train_bp_mini()", batch_size)
    
    debug = 0
    t = train.Train(r)
    r.set_backpropagation(True, 0.005)
    mini_batch_num = int(b.batch_size / batch_size)
        
    for n in range(iteration):
        k = random.randint(0, mini_batch_num-1)
        print("loop", n, k, mini_batch_num)
        (data_array, label_list, label_array) = b.get_batch(batch_size, k*batch_size)
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        
        debug = 0
        for i in range(10):
            ce = r.evaluate(debug)
            print(n, i, "CE:", ce)
            r.bp(debug)
            r.update_weight()
        #
        r.reset()
    #
    r.save(wmode)
        
def exec_train_bp(b, my_gpu, r, wmode, batch_size):
    print("exec_train_mini()", batch_size)
    
    t = train.Train(r)

    r.set_backpropagation(True, 0.01)
    (data_array, label_list, label_array) = b.get_batch(batch_size, batch_index)
    r.direct_set_data(data_array)
    r.direct_set_label(label_array)
        
    debug = 0
    for i in range(100):
        ce = r.evaluate(debug)
        print(i, "CE:", ce)
        r.bp(debug)
        r.update_weight()
    #
    r.save(wmode)
        
def exec_train_mini(b, my_gpu, r, wmode, batch_size, attack_num, iteration):
    print("exec_train_mini()", batch_size)
    b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
    b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
    print(b.loadDataAndLebel())
    
    t = train.Train(r)        
    t.w_list = t.make_w_list()
    
    mini_batch_num = int(mnist.TRAIN_BATCH_SIZE / batch_size)
    idx_list = list(range(mini_batch_num))
    random.shuffle(idx_list)
    for n in idx_list:
        (data_array, label_list, label_array) = b.get_batch(batch_size, n*batch_size)
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        ce = r.evaluate()
        loop_max = 100
        for i in range(iteration):
            ce, hit_rate = t.main_challenge_loop(ce, loop_max, attack_num, True)
        #
        r.reset()
    #
    r.save(wmode)
    
def exec_train(b, my_gpu, r, wmode, batch_size, attack_num, iteration):
    print("exec_train()", batch_size)
    b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
    b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
    print(b.loadDataAndLebel())
    
    (data_array, label_list, label_array) = b.get_batch(batch_size, 0)
    r.direct_set_data(data_array)
    r.direct_set_label(label_array)
    
    t = train.Train(r)
    t.w_list = t.make_w_list()
    
    ce = r.evaluate(0)
    #return
    
    print("CE:", ce)
    loop_max = 100
    for i in range(iteration):
        ce, hit_rate = t.main_challenge_loop(ce, loop_max, attack_num, True)
    #
    r.save(wmode)

def exec_test(b, my_gpu, r):
    b.setDataPath(mnist.TEST_IMAGE_BATCH_PATH)
    b.setLabelPath(mnist.TEST_LABEL_BATCH_PATH)
    b.loadDataAndLebel()
    
    debug = 0
    single = 0
    ac = exam.classification(r, b, 1000, debug, single)
    print(ac)
    
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

    #
    # gpu
    #
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config, exec_mode, wmode, qmode, batch_size)
    if r==None:
        return 0
    #
    
    start_time = time.time()
    print("exec_mode:", exec_mode)
    
    if exec_mode==0: # train
        exec_train(b, my_gpu, r, wmode, batch_size, attack_num, iteration)
    elif exec_mode==1: # test
        exec_test(b, my_gpu, r)
    elif exec_mode==2: # train mini
        exec_train_mini(b, my_gpu, r, wmode, batch_size, attack_num, iteration)
    elif exec_mode==3: # train bp
        exec_train_bp(b, my_gpu, r, wmode, batch_size)
    elif exec_mode==5: # train slope
        exec_train_slope(b, my_gpu, r, wmode, batch_size, attack_num, iteration)
    else:
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

