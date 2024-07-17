#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import pickle
import numpy as np
import csv
import random
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
#import util
import plat
import core
import train
import exam
#import opencl

import mnist

# tool
sys.path.append(os.path.join(os.path.dirname(__file__), '../ptool/'))
import tool

BATCH_DATA_ARRAY = tool.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
BATCH_LABEL_LSIT = tool.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
LABEL_ARRAY = np.array(BATCH_LABEL_LSIT, np.float32)
    
def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    batch_size = mnist.TEST_BATCH_SIZE
    mini_batch_size = 1000
    batch_offset_index = 0
    batch_num = int(batch_size / mini_batch_size)

    wpath = "./wi-cnn.csv"
    my_gpu = plat.getGpu()
    
    r = core.Roster()
    r.set_gpu(my_gpu)
    mnist.setup_cnn(r, data_size)
    
    r.set_path(wpath)
    r.set_scale_input(1)
    r.load()
    r.update_weight()
    r.prepare(mini_batch_size, data_size, num_class)

    batch_size = mnist.TEST_BATCH_SIZE
    batch_image = tool.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
    batch_label = tool.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
    ac = exam.classification(r, data_size, num_class, batch_size, batch_image, batch_label, 1000)
    print(ac)
    return 0

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
