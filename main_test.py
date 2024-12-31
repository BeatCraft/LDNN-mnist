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
    batch_image = util.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
    batch_label = util.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
    
    my_gpu = plat.getGpu()
    r = mnist.setup_dnn(my_gpu, config)
    if r==None:
        return 0
    #
    
    ac = exam.classification(r, data_size, num_class, batch_size, batch_image, batch_label, 1000)
    print(ac)
    
    #r.save_as("./w.csv", 1)
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
