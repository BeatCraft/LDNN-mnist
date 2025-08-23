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
import batch3

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
    elif mode==2 or mode==7: # mini batch train
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
        mode_q = 1
        print("full quantization train")
    else:
        print("mode error")
    #
    
    #
    # batch
    #
    mtype = 0 # classification
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    b = batch3.Batch(data_size, mtype, num_class)
    if exec_mode==0: # train
        #b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
        #b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
        pass
    else: # test
        b.setDataPath(mnist.TEST_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TEST_LABEL_BATCH_PATH)
    #
    #b.loadDataAndLebel()
    (data_array, label_list, label_array) = b.load_compressed("./batch/compressd/")
    batch_size = len(label_list)
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
        t = train.Train(r)
        t.w_list = t.make_w_list()
        
        #(data_array, label_list, label_array) = b.get_batch(batch_size, 0)
        print(data_array[0].shape)
        print( type(data_array[0]) )
        print(label_array.shape)
        
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)

        ce = r.evaluate(0)
        print(ce)
        #return 0
        
        num_attack_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
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
    elif mode==1: # test
        debug = 0
        single = 0
        ac = exam.classification(r, b, 1000, debug, single)
        print(ac)
    elif mode==2: # mini batch train
        return 0
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
        
    elif mode==7: # stochastic mini batch train
        t = train.Train(r)
        t.w_list = t.make_w_list()
        #b.prepare_mini_batch(batch_size)

        mini_batch_num = int(b.batch_size / batch_size)
        ce_list = []
        sum_ce = 0.0
        #ce_list = []
        for n in range(mini_batch_num):
            (data_array, label_list, label_array) = b.get_batch(batch_size, n*batch_size)
            r.reset()
            r.direct_set_data(data_array)
            r.direct_set_label(label_array)
        
            ce = r.evaluate()
            #ce_list.append((n, ce))
            ce_list.append(ce)
            #print(ce)
            sum_ce += ce
        #
        std_sample = np.std(ce_list, ddof=1)
        mean_value = np.mean(ce_list)
        median_value = np.median(ce_list)
        print(std_sample, mean_value, median_value)
        
        #closest_index = min(range(mini_batch_num), key=lambda i: abs(ce_list[i] - mean_value))
        #print(closest_index, ce_list[closest_index])
        
        closest_indices = sorted(
            range(mini_batch_num),
            key=lambda i: abs(ce_list[i] - mean_value)
        )[:3]
        
        print(closest_indices)
        
        
        min_index = ce_list.index(min(ce_list))
        max_index = ce_list.index(max(ce_list))
        print(min_index, max_index)
        #print(max_index, ce_list[max_index])
        
        closest_indices.append(min_index)
        closest_indices.append(max_index)
        print(closest_indices)
        
        (data_array, lavel_list, label_array) = b.get_batch_multi(batch_size, closest_indices)
        
        #return 0
                
        
        #(data_array, lavel_list, label_array) = b.get_batch_multi(batch_size, [closest_index*batch_size, max_index*batch_size])
        r.reset()
        
        #
        # realloc memory
        #
        r.prepare(batch_size*5, mnist.IMAGE_SIZE, mnist.NUM_CLASS)
        r.load()
        r.update_weight()
    
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        
        ce = r.evaluate()
        print(ce)
        ce, hit_rate = t.main_simple_loop(0, 0, ce, 100, 4)
        r.save()
        #
        return 0
        
        
        for i in range(1000): # epoc
            ce_list = []
            sum_ce = 0.0
            for n in range(mini_batch_num):
                #(data_array, label_list, label_array) = b.get_mini_batch(n*batch_size)
                (data_array, label_list, label_array) = b.get_batch(batch_size, n*batch_size)
                r.reset()
                r.direct_set_data(data_array)
                r.direct_set_label(label_array)
        
                ce = r.evaluate()
                ce_list.append((n, ce))
                sum_ce += ce
            #
            avg_ce = sum_ce/mini_batch_num
            sorted_data = sorted(ce_list, key=lambda x: x[1], reverse=True)
            dif = (sorted_data[0][1] - sorted_data[-1][1]) / sorted_data[-1][1]
            print("***", i, "*** average ce:", avg_ce, "(", sorted_data[-1][1], "-", sorted_data[0][1], ")", "***", dif, "***")
            with open("./log.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                line = [i, sorted_data[-1][1], sorted_data[0][1], avg_ce]
                writer.writerow(line)
            #
            
            #if i % 10 == 0:
            #    item = sorted_data[-1]
            #else:
            #    item = sorted_data[0]
            #
            item = sorted_data[0]
            #if i % 2 == 0:
            #    item = sorted_data[0]
            #else:
            #    item = sorted_data[-1]
            #
            #for item in sorted_data[:int(batch_size*0.01)]:
            #for item in sorted_data[:10]:
            ce = avg_ce
            if item:
                n = item[0]
                ce = item[1]
                #print(i, n, ce)
            
                #data_array, label_array = b.get_mini_batch(n*batch_size)
                data_array, lavel_list, label_array = b.get_batch(batch_size, n*batch_size)
                r.reset()
                r.direct_set_data(data_array)
                r.direct_set_label(label_array)
                
                #ce, hit_rate = t.main_simple_loop(0, 0, ce, 128, 4)
                # adaptive control
                rate = 0.01 # learning rate
                if ce>2.0:
                    rate = 0.1
                    num_attack = 256
                elif ce>1.5:
                    rate = 0.1
                    num_attack = 128
                elif ce>1.0:
                    rate = 0.1
                    num_attack = 64
                elif ce>0.5:
                    rate = 0.1
                    num_attack = 32
                elif ce>0.3:
                    rate = 0.1
                    num_attack = 16
                elif ce>0.2:
                    rate = 0.1
                    num_attack = 8
                elif ce>0.1:
                    rate = 0.1
                    num_attack = 4
                else:
                    rate = 0.01
                    num_attack = 4
                #
                
                ce, hit_rate = t.main_challenge_loop(ce, rate, 512, num_attack, False)
            #
            #print(i, ce)
            r.save()
            #b.shuffle_mini_batch()
        #
        
        #ce, hit_rate = t.main_simple_loop(0, 0, ce, 16, 4)
        #        # simple challenge
        #        sum_hit_rate += hit_rate
        #    #
        #    ave_hit_rate = sum_hit_rate / b.mini_batch_num
        #    print("*** ave_hit_rate:", ave_hit_rate)
        #    b.shuffle_mini_batch()
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

