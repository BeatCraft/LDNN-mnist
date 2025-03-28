#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import numpy as np

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
import core
import exam
import util
import batch

import mnist

sys.setrecursionlimit(10000)

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    if argc!=2:
        print("error : need batch offset index")
        return 0
    #
    config = int(argvs[1])

    batch_size = mnist.TEST_BATCH_SIZE # 10000
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    scale = True
    mini_batch_size = 1000
    
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config, mini_batch_size)
    if r==None:
        return 0
    #
    
    b = batch.Batch(data_size, type, num_class)
    b.quantize = True
    #b.quantize = False
    b.load_data(mnist.TEST_IMAGE_BATCH_PATH)
    b.load_label(mnist.TEST_LABEL_BATCH_PATH)
    b.prepare_batch(scale)
    b.prepare_mini_batch(mini_batch_size)
    
    debug = 0
    single = 0
    ac = exam.classification(r, b, 1000, debug, single)
    print(ac)
    
    #r.save_as("./w.csv", 1)
    #r.save_as("./wi-fc.csv", 0)
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
