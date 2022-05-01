#! /usr/bin/python
# -*- coding: utf-8 -*-
#

import os
import sys
import time
import random
#from stat import *
import numpy as np
#from PIL import Image

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
import util
import core
import train
import exam
    
if sys.platform.startswith('darwin'):
    import opencl
else:
    if plat.ID==1:
        import opencl
    elif plat.ID==2:
        import dgx
    #
#

import mnist

sys.setrecursionlimit(10000)

def train(r, batch_data, batch_label, batch_size, data_size, num_class, batch_offset):
    t = train.Train(r)
    #r.prepare(batch_size, data_size, num_class)
    #r.set_batch(data_size, num_class, batch_data, batch_label, batch_size, batch_offset)
    
    fc_w_list = t.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
    cnn_w_list = t.make_w_list([core.LAYER_TYPE_CONV_4])
            
    for idx in range(50):
        t.mode_w = 1
        r.propagate()
        for i in range(1, 5): # FC
            layer = r.get_layer_at(i)
            layer.lock = True
        #
        t.loop_k(fc_w_list, "fc", idx, 1, 50)
                
        t.mode_w = 2
        for i in range(1, 5): #CNN
            layer = r.get_layer_at(i)
            layer.lock = False
        #
        t.loop_k(cnn_w_list, "cnn", idx, 1, 20)
    #

def test(r, batch_data, batch_label, batch_size, data_size, num_class, batch_offset):
    r.prepare(100, data_size, num_class)
    r.set_batch(data_size, num_class, batch_data, batch_label, 100, 0)

    exam.classification(r, data_size, num_class, batch_size, batch_data, batch_label, 100)
    #for i in range(batch_size):
    #    idx = i + batch_offset
    #
    #
    #
    #pass


class DataEntry(object):
    def __init__(self, i, data, sise, label):
        self.index = int(i)
        self.data = data
        self.sise = sise
        self.label = label
        
class DataHandler(object):
    def __init__(self, data_size, num_class):
        self.data_size = data_size
        self.num_class = num_class
        self.container = []
        for i in range(num_class):
            self.container.append([])
        #
        
    def load(self, batch_size, data_path, label_path):
        self.batch_size = batch_size
        self.data_path = data_path
        self.label_path = label_path
        self.base_data = util.pickle_load(data_path)
        self.base_label = util.pickle_load(label_path)
        
        for i in range(self.batch_size):
            label = self.base_label[i]
            data = self.base_data[i]
            self.container[label].append(DataEntry(i, data, self.data_size, label))
        #
        
    def shuffle(self):
        #print("DataHandler::shuffle()", self.num_class)
        for i in range(self.num_class):
            random.shuffle(self.container[i])
        #
        
    def head(self, size=10): # size per class
        h = []
        for i in range(self.num_class):
            #print(self.container[i])
            h.append(self.container[i][:size])
            #print(self.container[i])
        #
        return h
        
    def extract(self, elist):
        p = []
        for e in elist:
            p.append(self.container[e.index])
        #
    def flatten(self, list2d):
        list1d = []
        for l in list2d:
            for i in l:
                list1d.append(i)
            #
        #
        return list1d
    
    def makeBatch(self, blist):
        print(blist)
        batch_size = len(blist)
        print("batch_size =", batch_size)
        batch_data = np.zeros((batch_size, self.data_size), dtype=np.float32)
        batch_label = []
        
        for i in range(batch_size):
            d = blist[i]
            batch_data[i] = d.data
            batch_label.append(d.label)
        #
        return batch_data, batch_label
        
def main():
    argvs = sys.argv
    argc = len(argvs)
    
    config = 1 # CNN
    mode = 0
    batch_size_full = mnist.TRAIN_BATCH_SIZE
    batch_size = 100
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    batch_offset = 0
    
    dh = DataHandler(data_size, num_class)
    dh.load(mnist.TRAIN_BATCH_SIZE, mnist.TRAIN_IMAGE_BATCH_PATH, mnist.TRAIN_LABEL_BATCH_PATH)
    
    dh.shuffle()
    p100 = dh.head(1)
    p1001d = dh.flatten(p100)
    print(len(p1001d))
    #print(p1001d)
    
    batch_data, batch_label = dh.makeBatch(p1001d)
    print(batch_data[0])
    print(batch_label)
    
    
    
    
    return 0
    
    if plat.ID==0:   # MBP
        my_gpu = opencl.OpenCL(0, 1)
        my_gpu.set_kernel_code()
    elif plat.ID==1: # TR
        my_gpu = opencl.OpenCL(1, 0)
        my_gpu.set_kernel_code()
    elif plat.ID==2: # nvidia
        my_gpu = dgx.Dgx(1)
    else:
        print("error : undefined platform")
        return 0
    #
    r = mnist.setup_dnn(my_gpu, config, "./wi-cnn-0.csv")
    if r:
        pass
    else:
        return 0
    #

    batch_data_full = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    batch_label_full = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    #print(type(batch_data_full))
    
    #[start:stop:step]
    start = 0
    stop = batch_size #100
    batch_data = batch_data_full[start:stop]
    print(len(batch_data))
    batch_label = batch_label_full[start:stop]
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, batch_data, batch_label, batch_size, 0)
    r.propagate(0)
    answers = r.get_answer()
    print(answers)
    
    start = stop
    stop = batch_size_full #100
    batch_data = batch_data_full[start:stop]
    batch_label = batch_label_full[start:stop]
    
    n = int(len(batch_label)/batch_size)
    print(n)
    
    for i in range(n):
        start = i*batch_size
        stop = start + batch_size
        mini_data = batch_data_full[start:stop]
        mini_label = batch_label_full[start:stop]
        r.set_batch(data_size, num_class, mini_data, mini_label, batch_size, 0)
        r.propagate(0)
        answers = r.get_answer()
        correct = []
        for j in range(batch_size):
            if mini_label[j]==answers[j]:
                correct.append(j)
            #
        #
        print(len(correct))
        #print(answers)
    #
    
    return 0
    
    data_list = []
    for i in range(batch_size):
        data_list.append(DataEntry(batch_data_full[i], data_size, batch_label_full[i]))
    #
    r.prepare(1, data_size, num_class)
    for idx in range(len(data_list)):
        #r.set_batch(data_size, num_class, [d.data], [d.label,], 1, 0)
        r.set_batch(data_size, num_class, [data_list[idx].data], [data_list[idx].label,], 1, 0)
        r.propagate(0)
        answers = r.get_answer()
        print(idx, answers, data_list[idx].label)
        #print(d.label)
    #
    #exam.classification(r, data_size, num_class, 1, data_list[0].data, [data_list[0].label,], 1)
    return 0
    
    batch_data = np.zeros((batch_size, data_size), dtype=np.float32)
    batch_label = []
    for i in range(batch_size):
        batch_data = batch_data_full[i]
        batch_label.append(batch_label_full[i])
    #
    
    #r.prepare(batch_size, data_size, num_class)
    #r.set_batch(data_size, num_class, batch_data, batch_label, batch_size, batch_offset)
    #train(r, batch_data, batch_label, batch_size, data_size, num_class, batch_offset)
    batch_offset = batch_size
    batch_size = 100 #mnist.TRAIN_BATCH_SIZE - batch_size
    test(r, batch_data, batch_label, batch_size, data_size, num_class, batch_offset)
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
