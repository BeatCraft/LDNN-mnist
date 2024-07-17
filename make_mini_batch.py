#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import numpy as np
import pickle

import mnist

# tool
sys.path.append(os.path.join(os.path.dirname(__file__), '../ptool/'))
import tool

def main():
    argvs = sys.argv
    argc = len(argvs)

    data_size = mnist.IMAGE_SIZE
    class_num = mnist.NUM_CLASS
    
    # image
    batch_image = tool.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH) # numpy array
    # label
    batch_label = tool.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH) # list

    print(batch_image.shape)
    mini_batch_size = 1000
    batch_size = batch_image.shape[0]
    batch_num = int(batch_size / mini_batch_size)
    print(batch_num, mini_batch_size, batch_image.shape[1])
    
    batch_image = batch_image.reshape(batch_num, mini_batch_size, batch_image.shape[1])
    print(batch_image.shape)
    
    for i in range(batch_num):
        dpath = "./mini_batch/%06d_data.pickle" % (i)
        tool.pickle_save(dpath, batch_image[i])
        
        lpath = "./mini_batch/%06d_label.pickle" % (i)
        lb = batch_label[mini_batch_size * i : mini_batch_size * (i + 1)]
        tool.pickle_save(lpath, lb)
    #
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
