#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import random
import numpy as np

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import util
import core
import train

sys.setrecursionlimit(10000)

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
    
    #def load(self, batch_size, data_path, label_path):
    def load(self, data_path, label_path):
        #self.batch_size = batch_size
        self.data_path = data_path
        self.label_path = label_path
        self.base_data = util.pickle_load(data_path)
        self.base_label = util.pickle_load(label_path)
        self.batch_size = len(self.base_label)
        
        for i in range(self.batch_size):
            label = self.base_label[i]
            data = self.base_data[i]
            self.container[label].append(DataEntry(i, data, self.data_size, label))
        #
        
    def shuffle(self):
        for i in range(self.num_class):
            random.shuffle(self.container[i])
        #
        
    def head(self, size=10): # size per class
        h = []
        for i in range(self.num_class):
            h.append(self.container[i][:size])
        #
        return h
        
    def divide(self, size=10): # size per class
        h = []
        for i in range(self.num_class):
            h.append(self.container[i][:size])
        #
        t = []
        for i in range(self.num_class):
            t.append(self.container[i][size:])
        #
        return h, t
        
    def flatten(self, list2d):
        list1d = []
        for l in list2d:
            for i in l:
                list1d.append(i)
            #
        #
        return list1d
    
    def extract(self, elist):
        batch_size = len(elist)
        print("batch_size =", batch_size)
        
        batch_data = np.zeros((batch_size, self.data_size), dtype=np.float32)
        batch_label = []
        
        for i in range(batch_size):
            idx = elist[i]
            batch_data[i] = self.base_data[idx]
            batch_label.append(self.base_label[idx])
        #
        return batch_data, batch_label
        
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
        
    def saveBatch(self, batch_data, batch_label, data_path, label_path):
        util.pickle_save(data_path, batch_data)
        util.pickle_save(label_path, batch_label)

#def train_loop(r, batch_data, batch_label, batch_size, data_size, num_class, batch_offset):
def train_loop(r):
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

def test_loop(r, batch_data, batch_label, batch_size, data_size, num_class, batch_offset):
    n = 100
    it, left = divmod(batch_size, n)
    print(it, left)
    r.prepare(n, data_size, num_class)
    correct = []
    incorrect = []
    for i in range(it):
        start = i*n
        end = start + n
        #print(i, start, end)
        data = batch_data[start:end]
        #print(data.shape)
        label = batch_label[start:end]
        
        r.set_batch(data_size, num_class, data, label, n, 0)
        r.propagate(0)
        answers = r.get_answer()
        #print(answers)
        for j in range(len(answers)):
            #print(answers[j], batch_label[start+j])
            if answers[j] == batch_label[start+j]:
                correct.append(start+j)
            else:
                incorrect.append(start+j)
            #
        #
    #
    #print(len(correct))
    return correct, incorrect
        
    
    #r.set_batch(data_size, num_class, batch_data, batch_label, 100, 0)
    #exam.classification(r, data_size, num_class, batch_size, batch_data, batch_label, 100)
