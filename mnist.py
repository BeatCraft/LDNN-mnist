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
    output = core.OutputLayer(c, 256, 10, hidden_2, r._gpu)
    r.layers.append(output)

def setup_dnn(my_gpu, config, mode_q=0, batch_size=0):
    if config==0:
        wpath = "./wi-fc.csv"
    elif config==1:
        wpath = "./wi-cnn.csv"
    elif config==2:
        wpath = "./w.csv"
    else:
        return None
    #
    
    r = core.Roster()
    r.set_gpu(my_gpu)
    if config==0: # fc with wi
        setup_fc(r, IMAGE_SIZE) # 28*28
        # 0:even, 5:std, 7:3bit
        r.wi_mode = 0
        #r.wi_mode = 5
        #r.wi_mode = 7
    elif config==1: # cnn
        print("error, no cnn")
        return None
    elif config==2: # fc with float value
        r.wi_mode = 6
    else:
        print("error config", config)
    #
    
    r._batch_size = batch_size
    r.set_path(wpath)
    #r.set_scale_input(1)
    r.set_mode_q(mode_q) # 0:old, 1:new
    print("batch_size", batch_size)
    r.prepare(batch_size, IMAGE_SIZE, NUM_CLASS)
    
    r.load()
    r.update_weight()
    return r
    
