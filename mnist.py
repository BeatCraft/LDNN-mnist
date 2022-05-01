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

# fpr MPI
#MINI_BATCH_SIZE = [7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500]
#MINI_BATCH_START = [0, 7500, 15000, 22500, 30000, 37500, 45000, 52500]
#MINI_BATCH_SIZE = [6250, 6250, 6250, 6250, 6250, 6250, 6250, 6250, 6250, 6250]
#MINI_BATCH_START = [0, 6250, 12500, 18750, 25000, 31250, 37500, 43750]
MINI_BATCH_SIZE = [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]
MINI_BATCH_START = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500]

def setup_cnn(r, size):
    print("setup_cnn(%d)" % (size))
    
    c = r.count_layers()
    input = core.InputLayer(c, size, size, None, r._gpu)
    r.layers.append(input)
    
    c = r.count_layers()
    cnn_1 = core.Conv_4_Layer(c, 28, 28, 1, 8, input, r._gpu)
    r.layers.append(cnn_1)
                
    c = r.count_layers()
    max_1 = core.MaxLayer(c, 8, 28, 28, cnn_1, r._gpu)
    r.layers.append(max_1)
    
    c = r.count_layers()
    cnn_2 = core.Conv_4_Layer(c, 14, 14, 8, 16, max_1, r._gpu)
    r.layers.append(cnn_2)
    
    c = r.count_layers()
    max_2 = core.MaxLayer(c, 16, 14, 14, cnn_2, r._gpu)
    r.layers.append(max_2)
    
    c = r.count_layers()
    hidden_1 = core.HiddenLayer(c, 7*7*16, 128, max_2, r._gpu)
    r.layers.append(hidden_1)

    c = r.count_layers()
    hidden_2 = core.HiddenLayer(c, 128, 128, hidden_1, r._gpu)
    r.layers.append(hidden_2)

    c = r.count_layers()
    output = core.OutputLayer(c, 128, 10, hidden_2, r._gpu)
    r.layers.append(output)
    
def setup_fc(r, size):
    print("setup_fc(%d)" % (size))
    
    c = r.count_layers()
    input = core.InputLayer(c, size, size, None, r._gpu)
    r.layers.append(input)
    # 1 : hidden : 28 x 28 x 1 = 784
    c = r.count_layers()
    hidden_1 = core.HiddenLayer(c, size, 64, input, r._gpu)
    r.layers.append(hidden_1)
    # 2 : hidden : 64
    c = r.count_layers()
    hidden_2 = core.HiddenLayer(c, 64, 64, hidden_1, r._gpu)
    r.layers.append(hidden_2)
    # 3 : hidden : 64
    c = r.count_layers()
    hidden_3 = core.HiddenLayer(c, 64, 64, hidden_2, r._gpu)
    r.layers.append(hidden_3)
    # 3 : output
    c = r.count_layers()
    output = core.OutputLayer(c, 64, 10, hidden_3, r._gpu)
    r.layers.append(output)

def setup_dnn(my_gpu, config, path):
    r = core.Roster()
    r.set_gpu(my_gpu)
        
    if config==0: # fc
        setup_fc(r, IMAGE_SIZE) # 28*28
        r.set_path(path)
    elif config==1: # cnn
        setup_cnn(r, IMAGE_SIZE) # 28*28
        r.set_path(path)
    #
    
    r.set_scale_input(1)
    r.load()
    r.update_weight()
    return r

