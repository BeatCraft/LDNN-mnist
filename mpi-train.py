#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
#
from mpi4py import MPI
#import cupy as cp
#import cupyx

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import plat
import util
import core
import train
import work
import exam
import mnist

sys.setrecursionlimit(10000)

def output(path, msg):
    with open(path, 'a') as f:
        print(msg, file=f)
    #
    
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    start_time = time.time()
    #
    com = MPI.COMM_WORLD
    rank = com.Get_rank()
    size = com.Get_size()
    print("(%d, %d)" % (rank, size))

    #config = 0 # FC
    config = 1 # CNN all
    #
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    train_image_batch = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_label_batch = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    batch_offset = mnist.MINI_BATCH_START[rank]
    batch_size = mnist.MINI_BATCH_SIZE[rank]
    
    my_gpu = plat.getGpu(rank)
    r = mnist.setup_dnn(my_gpu, config, "./wi-fc.csv")
    if config==0:
        r = mnist.setup_dnn(my_gpu, config, "./wi-fc.csv")
    elif config==1:
        r = mnist.setup_dnn(my_gpu, config, "./wi-cnn.csv")
    else:
        return 0
    #
    
    r.prepare(batch_size, data_size, num_class)
    tr = train.Train(r, com, rank, size)
    r.set_batch(data_size, num_class, train_image_batch, train_label_batch, batch_size, batch_offset)
    
    if rank==0:
        w_list = tr.make_w_list([core.LAYER_TYPE_CONV, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
    else:
        w_list = []
    #
    w_list = com.bcast(w_list, root=0)
            
    if config==0:
        #ce = wk.loop_sa5(0, w_list, "all", 100, 200, 1.50, 1) # 2.00, 1.50, 1.25, 1.10
        ce = tr.loop_sa_20(0, w_list, 0)
        #for i in range(100):
        #    ce = wk.loop_sa5(i, w_list, "all")
        #    if rank==0:
        #        log = "%d, %s" % (i+1, '{:.10g}'.format(ce))
        #        output("./log.csv", log)
        #        spath = "./wi/wi-fc-%04d.csv" % (i+1)
        #        r.save_as(spath)
        #    #
    elif config==1:
        idx = 0
        temperature = 200
        total = 10000
        debug = 0
        asw = 1
        tr.loop_sa(idx, w_list, "all", temperature, total, debug, asw)
    else:
        return 0
    #

    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print(("[%d] time = %s" % (rank, t)))

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
