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

def check_max_grad(layer):
    # layer is MaxLayer
    grad_metal = layer.grad.copy()

    B = layer._batch_size
    ch = layer._ch
    out_w = layer._x
    out_h = layer._y
    in_w = out_w * 2
    in_h = out_h * 2

    delta = layer.delta.reshape(B, ch, out_h, out_w)
    mask = layer._mask_array.reshape(B, ch, in_h, in_w)

    grad_np = np.zeros((B, ch, in_h, in_w), dtype=np.float32)

    for b in range(B):
        for c in range(ch):
            for y in range(out_h):
                for x in range(out_w):
                    d = delta[b, c, y, x]
                    grad_np[b, c, y*2:y*2+2, x*2:x*2+2] = (
                        mask[b, c, y*2:y*2+2, x*2:x*2+2] * d
                    )

    grad_np = grad_np.reshape(B, ch, in_h * in_w)

    print("=== Max grad check ===")
    print("max abs diff:", np.max(np.abs(grad_metal - grad_np)))
    print("mean abs diff:", np.mean(np.abs(grad_metal - grad_np)))
    print("sign match:", np.mean(np.sign(grad_metal) == np.sign(grad_np)))
    
def train_slope(b, my_gpu, r, wmode, batch_size, attack_num, t, ce, n, undo=False):
    r.slope(0)
    cnt = 0
    attack_list = []
    while cnt<attack_num:
        k = random.randint(0, len(t.w_list)-1)
        w = t.w_list[k]
        li = w.li
        ni = w.ni
        ii = w.ii
        l = r.get_layer_at(li)
        wi = w.wi
        type = w.type
        if type==core.LAYER_TYPE_CONV:
            kmax = core.CNN_WEIGHT_INDEX_MAX
            kmin = core.CNN_WEIGHT_INDEX_MIN
        else:
            kmax = core.WEIGHT_INDEX_MAX
            kmin = core.WEIGHT_INDEX_MIN
        #
        if l.dW[ii][ni]==0.0: # no slope
            pass
        elif l.dW[ii][ni]<0.0: # ++
            if wi==kmax:
                pass
            else:
                w.wi_alt = w.wi
                w.wi = wi + 1
                l.set_weight_index(w.ni, w.ii, wi+1) # attack
                attack_list.append(w)
            #
        elif l.dW[ii][ni]>0.0: # --
            if wi==kmin:
                pass
            else:
                w.wi_alt = w.wi
                w.wi = wi - 1
                l.set_weight_index(w.ni, w.ii, wi-1) # attack
                attack_list.append(w)
            #
        #
        cnt += 1
    # while

    r.update_weight()
    ce_alt = r.evaluate(0)
    if ce_alt>ce: # undo
        if undo:
            print("[%d](%d/%d)" % (n, cnt, attack_num), ce, "(", ce_alt, "), UNDO")
            for w in attack_list:
                li = w.li
                l = r.get_layer_at(w.li)
                w.wi = w.wi_alt
                l.set_weight_index(w.ni, w.ii, w.wi)
            #
            r.update_weight()
        else:
            print("[%d](%d/%d)" % (n, cnt, attack_num), ce, "-->", ce_alt)
            ce = ce_alt
        #
    else:
        print("[%d](%d/%d)" % (n, cnt, attack_num), ce, "->", ce_alt)
        ce = ce_alt
    #
    return ce

def exec_train_slope(b, my_gpu, r, wmode, batch_size, attack_num, iteration, undo=False):
    print("exec_train_slope()", batch_size)
    r.set_backpropagation(True, 0.005)

    (data_array, label_list, label_array) = b.get_batch(batch_size, 0)
    r.direct_set_data(data_array)
    r.direct_set_label(label_array)
    
    t = train.Train(r)
    t.w_list = t.make_w_list()
    
    ce = r.evaluate(0)
    ce_alt = 100.0
    print("CE :", ce)
    
    for n in range(iteration):
        ce = train_slope(b, my_gpu, r, wmode, batch_size, attack_num, t, ce, n, undo)
    #
    r.save(wmode)
    return

def exec_train_slope_mini2(b, my_gpu, r, wmode, batch_size, attack_num, iteration, undo=False):
    print("exec_train_slope()", batch_size)
    r.set_backpropagation(True, 0.005)
    mini_batch_num = int(b.batch_size / batch_size)
    
    bidx_list = list(range(mini_batch_num))
    random.shuffle(bidx_list)
    
    t = train.Train(r)
    t.w_lists = t.make_w_list_by_layer()

    ce_alt = 100.0
    
    for bidx in bidx_list:
    #for bidx in range(1):
        #bidx = 0
        #print(bidx)
        (data_array, label_list, label_array) = b.get_batch(batch_size, bidx*batch_size)
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        ce = r.evaluate(0)
        #
        ce = t.train_slope2(b, my_gpu, r, wmode, batch_size, attack_num, ce, bidx, undo)
        r.reset()
        #
    #
    r.save(wmode)
    return
    
def exec_train_slope_mini(b, my_gpu, r, wmode, batch_size, attack_num, iteration, undo=False):
    print("exec_train_slope()", batch_size)
    r.set_backpropagation(True, 0.005)
    mini_batch_num = int(b.batch_size / batch_size)
    
    bidx_list = list(range(mini_batch_num))
    random.shuffle(bidx_list)
    
    t = train.Train(r)
    t.w_list = t.make_w_list()
    #t.w_list = t.make_w_list([core.LAYER_TYPE_CONV])
    #t.w_list = t.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])

    ce_alt = 100.0
    """
    (data_array, label_list, label_array) = b.get_batch(batch_size, 0*batch_size)
    r.direct_set_data(data_array)
    r.direct_set_label(label_array)
    ce = r.evaluate(0)
    r.slope(0)

    for li in range(r.count_layers()):
        layer = r.get_layer_at(li)
        if layer.get_type() == core.LAYER_TYPE_CONV:
            layer.check_conv_grad(0)
        elif layer.get_type() == core.LAYER_TYPE_MAX:
            check_max_grad(layer)
        #
    #
    return 0
    """
    
    for bidx in bidx_list:
        (data_array, label_list, label_array) = b.get_batch(batch_size, bidx*batch_size)
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        ce = r.evaluate(0)
        #
        ce = t.train_slope(b, my_gpu, r, wmode, batch_size, attack_num, ce, bidx, undo)
        r.reset()
        #
        
    #
    r.save(wmode)
    return

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
    r.set_backpropagation(True, 0.01)
        
    t = train.Train(r)
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
    
    (data_array, label_list, label_array) = b.get_batch(batch_size, 0)
    r.direct_set_data(data_array)
    r.direct_set_label(label_array)
    
    t = train.Train(r)
    t.w_list = t.make_w_list()
    
    ce = r.evaluate(0)
    print("CE:", ce)
    loop_max = 100
    for i in range(iteration):
        ce, hit_rate = t.main_challenge_loop(ce, loop_max, attack_num, True)
    #
    r.save(wmode)

def exec_test(b, my_gpu, r):
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
        print("attack_num:", attack_num)
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
        b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
        print(b.loadDataAndLebel())
        #exec_train(b, my_gpu, r, wmode, batch_size, attack_num, iteration)
        #exec_train_slope(b, my_gpu, r, wmode, batch_size, attack_num, iteration)
        exec_train_slope(b, my_gpu, r, wmode, batch_size, attack_num, iteration, False)
    elif exec_mode==1: # test
        b.setDataPath(mnist.TEST_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TEST_LABEL_BATCH_PATH)
        b.loadDataAndLebel()
        exec_test(b, my_gpu, r)
    elif exec_mode==2: # train mini
        b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
        print(b.loadDataAndLebel())
        undo = True
        exec_train_slope_mini2(b, my_gpu, r, wmode, batch_size, attack_num, iteration, undo)
    elif exec_mode==3: # train bp
        print("temporaly, disabled")
        pass
        #b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
        #b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
        #print(b.loadDataAndLebel())
        #exec_train_bp(b, my_gpu, r, wmode, batch_size)
    elif exec_mode==5: # train slope
        b.setDataPath(mnist.TRAIN_IMAGE_BATCH_PATH)
        b.setLabelPath(mnist.TRAIN_LABEL_BATCH_PATH)
        print(b.loadDataAndLebel())
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
