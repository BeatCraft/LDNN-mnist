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
import util
import core
import train
import exam
import mnist

sys.setrecursionlimit(10000)
       
def save_array_to_png(array, w, h, path): # uint8, 1d, RGB
    data_in = np.reshape(array, (3, w*h))
    r = data_in[0]
    g = data_in[1]
    b = data_in[2]
    
    data = np.zeros((w*h, 3), dtype=np.uint8)
    
    for i in range(w*h):
        data[i][0] = r[i]
        data[i][1] = g[i]
        data[i][2] = b[i]
    #
    data = np.reshape(data, (h, w, 3)) # (2048, 1536, 4)
    pimg = Image.fromarray(data)
    pimg.save(path)

def save_array_gray_to_png(array, w, h, path): # uint8, 1d
    data = np.reshape(array, (h, w))
    pimg = Image.fromarray(data)
    pimg.save(path)

def output(path, msg):
    with open(path, 'a') as f:
        print(msg, file=f)
    #

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    if argc==4:
        pass
    else:
        print("error in sh")
        return 0
    #
    
    config = int(argvs[1])
    idx = int(argvs[2])
    log = int(argvs[3])
    
    batch_offset = 0
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    
    my_gpu = plat.getGpu()
    if config==0:
        r = mnist.setup_dnn(my_gpu, config, "./wi-fc.csv")
    elif config==1:
        r = mnist.setup_dnn(my_gpu, config, "./wi-cnn.csv")
    else:
        return 0
    #
    if r:
        pass
    else:
        return 0
    #
    
    batch_size = mnist.TEST_BATCH_SIZE
    batch_image = util.pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
    batch_label = util.pickle_load(mnist.TEST_LABEL_BATCH_PATH)
        
    ac = exam.classification(r, data_size, num_class, batch_size, batch_image, batch_label, 1000)
    if log==1:
        msg = "%d, %f" % (idx, ac)
        output("./ac.csv", msg)
    #
    print(ac)
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
