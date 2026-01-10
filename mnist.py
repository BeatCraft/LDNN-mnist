#! /usr/bin/python
# -*- coding: utf-8 -*-
#

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))

import util
import core

IMAGE_HEADER_SIZE = 16
LABEL_HEADER_SIZE  = 8
IMAGE_SIZE = 784
NUM_CLASS = 10

MODEL_TYPE = 0 # classification

DATA_BASE_PATH = "./data/"
TRAIN_IMAGE_PATH = DATA_BASE_PATH + "train-images-idx3-ubyte"
TRAIN_LABEL_PATH = DATA_BASE_PATH + "train-labels-idx1-ubyte"
TEST_IMAGE_PATH = DATA_BASE_PATH + "t10k-images-idx3-ubyte"
TEST_LABEL_PATH = DATA_BASE_PATH + "t10k-labels-idx1-ubyte"

BATCH_BASE_PATH = "./batch/"

TRAIN_BATCH_SIZE = 60000
TRAIN_IMAGE_BATCH_PATH = BATCH_BASE_PATH + "train_image_batch.pickle"
TRAIN_LABEL_BATCH_PATH = BATCH_BASE_PATH + "train_label_batch.pickle"

TEST_BATCH_SIZE = 10000
TEST_IMAGE_BATCH_PATH = BATCH_BASE_PATH + "test_image_batch.pickle"
TEST_LABEL_BATCH_PATH = BATCH_BASE_PATH + "test_label_batch.pickle"

def setup_cnn(r, size):
    print("setup_fc(%d)" % (size))

    c = r.count_layers()
    input = core.InputLayer(c, size, size, None, r._gpu)
    r.layers.append(input)
    
    # cnn
    c = r.count_layers()
    cnn_1 = core.Conv_4_Layer(c, 28, 28, 3, 8, input, r._gpu)
    r.layers.append(cnn_1)
    
    # max
    c = r.count_layers()
    max_1 = core.MaxLayer(c, 8, 28, 28, cnn_1, r._gpu)
    r.layers.append(max_1)
    
    # fc
    c = r.count_layers()
    hidden_1 = core.HiddenLayer(c, 14*14*8, 256, max_1, r._gpu)
    r.layers.append(hidden_1)
    
    # fc
    c = r.count_layers()
    hidden_2 = core.HiddenLayer(c, 256, 256, hidden_1, r._gpu)
    r.layers.append(hidden_2)
    
    # output
    c = r.count_layers()
    smax = True
    output = core.OutputLayer(c, 256, 10, hidden_2, r._gpu, smax)
    r.layers.append(output)

def setup_fc(r, size):
    print("setup_fc(%d)" % (size))

    c = r.count_layers()
    input = core.InputLayer(c, size, size, None, r._gpu)
    r.layers.append(input)
    
    # hidden 1
    c = r.count_layers()
    hidden_1 = core.HiddenLayer(c, size, 256, input, r._gpu)
    r.layers.append(hidden_1)
        
    # hidden 2
    c = r.count_layers()
    hidden_2 = core.HiddenLayer(c, 256, 256, hidden_1, r._gpu)
    r.layers.append(hidden_2)
    
    # output
    c = r.count_layers()
    smax = True
    output = core.OutputLayer(c, 256, 10, hidden_2, r._gpu, smax)
    r.layers.append(output)

def setup_dnn(my_gpu, config, exe_mode, wmode=0, qmode=0, batch_size=0):
    if config==0: # FC
        if exe_mode==0: # train
            if wmode==0:
                wpath = "./wi-fc.csv"
            elif wmode==1:
                wpath = "./w-fc.csv"
            #
        elif exe_mode==1: # test
            if wmode==0:
                wpath = "./wi-fc.csv"
            elif wmode==1:
                wpath = "./w-fc.csv"
            #
        elif exe_mode==3 or exe_mode==4: # train bp
            wpath = "./w-fc.csv"
        else:
            wpath = "./wi-fc.csv"
        #
    elif config==1: # CNN
        if exe_mode==1: # test
            if wmode==0:
                wpath = "./wi-cnn.csv"
            elif wmode==1:
                print("not yet")
                return None
                wpath = "./w-cnn.csv"
            #
        else:
            print("not yet")
            return None
        #
    else:
        print("not yet")
        return None
    #
    
    r = core.Roster()
    r.set_gpu(my_gpu)
    if config==0: # fc with wi
        setup_fc(r, IMAGE_SIZE) # 28*28
        # 0:even, 5:std, 7: latest dev.
        r.wi_mode = 7
    elif config==1: # cnn with wi
        print("config CNN", config)
        setup_cnn(r, IMAGE_SIZE) # 28*28
        r.wi_mode = 7
    else:
        print("error config", config)
        return None
    #
    
    r._batch_size = batch_size
    r.set_path(wpath)
    #r.set_scale_input(1)
    r.set_qmode(qmode) # 0:32bit, 1:16bit
    print("batch_size", batch_size)
    r.prepare(batch_size, IMAGE_SIZE, NUM_CLASS)
    
    r.load(wpath, wmode)
    r.update_weight()
    return r
    
