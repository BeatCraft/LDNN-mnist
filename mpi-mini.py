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
#if sys.platform.startswith('darwin'):
#    import opencl
#else:
#    if plat.ID==1:
#        import opencl
#    elif plat.ID==2:
#        import dgx
#        import cupy as cp
#        import cupyx
#    #
#
import util
import core
#import dgx
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

    config = 0 # FC
    #config = 1 # CNN all
    #config = 2 # CNN separate
    #
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    train_image_batch = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_label_batch = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    batch_offset = 0 #mnist.MINI_BATCH_START[rank]
    batch_size = 60000 #mnist.MINI_BATCH_SIZE[rank]
    mini_batch_size = 120
    node_size = 15
    
    my_gpu = plat.getGpu(rank)
    if config==0:
        r = mnist.setup_dnn(my_gpu, config, "./wi-fc.csv")
    elif config==1:
        r = mnist.setup_dnn(my_gpu, config, "./wi-cnn.csv")
    else:
        return 0
    #
    r.prepare(node_size, data_size, num_class)
    tr = train.Train(r, com, rank, size)
    
    if rank==0:
        w_list = tr.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
    else:
        w_list = []
    #
    w_list = com.bcast(w_list, root=0)

    n = int(batch_size/mini_batch_size)
    for i in range(n):
        offset = i*mini_batch_size + node_size*rank
        r.set_batch(data_size, num_class, train_image_batch, train_label_batch, node_size, offset)
        #
        tr.loop_sa(i, w_list, "fc")
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
