#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
#
from mpi4py import MPI
import cupy as cp
import cupyx

#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '../ldnn'))
import util
import core
import dgx
import train
import mpi
import mnist

sys.setrecursionlimit(10000)

def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    com = MPI.COMM_WORLD
    rank = com.Get_rank()
    size = com.Get_size()
    print("(%d, %d)" % (rank, size))

    #config_id = 0 # FC
    config_id = 1 # CNN
    mode = 0
    #
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    train_data_batch = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_label_batch = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    batch_offset = mnist.MINI_BATCH_START[rank]
    batch_size = mnist.MINI_BATCH_SIZE[rank]
        
    #processor_name = MPI.Get_processor_name()
    cp.cuda.Device(rank).use()
    my_gpu = dgx.Dgx(rank)
    
    r = mnist.setup_dnn(my_gpu, config_id)
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
    
    wk = mpi.worker(com, rank, size, r)
    if config_id==0:
        if rank==0:
            w_list = wk.train.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        else:
            w_list = []
        #
        w_list = com.bcast(w_list, root=0)
            
        wk.mode_w = 0 # 0:normal, 1:fc, 2:cnn, 3:single cnn, 4:regression
        wk.mode_e = 0 # 0:ce, 1:mse
        for idx in range(1000):
            wk.loop_k(w_list, "all", idx, 1)
        #
    else:
        if mode==0: # all
            if rank==0:
                w_list = wk.train.make_w_list([core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
            else:
                w_list = []
            #
            w_list = com.bcast(w_list, root=0)

            wk.mode_w = 0 # 0:normal, 1:fc, 2:cnn, 3:single cnn, 4:regression
            wk.mode_e = 0 # 0:ce, 1:mse
            for idx in range(10000):
                wk.loop_k(w_list, "all", idx, 1)
            #
        else: # separate
            if rank==0:
                fc_w_list = wk.train.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
            else:
                fc_w_list = []
            #
            fc_w_list = com.bcast(fc_w_list, root=0)

            if rank==0:
                cnn_w_list = wk.train.make_w_list([core.LAYER_TYPE_CONV_4])
            else:
                cnn_w_list = []
            #
            cnn_w_list = com.bcast(cnn_w_list, root=0)
            
            for idx in range(1000):
                wk.mode_w = 1
                r.propagate()
                for i in range(1, 5): # FC
                    layer = r.get_layer_at(i)
                    layer.lock = True
                #
                wk.loop_k(fc_w_list, "fc", idx, 1)

                wk.mode_w = 2
                for i in range(1, 5): #CNN
                    layer = r.get_layer_at(i)
                    layer.lock = False
                #
                wk.loop_k(cnn_w_list, "cnn", idx, 1)
            #
        #
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
