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

    # 0 : FC, 1 : CNN
    config_id = 1
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
    #r.set_scale_input(1)
    #r.set_path("./wi.csv")
    #r.load()
    #r.update_weight()
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
    
    wk = mpi.worker(com, rank, size, r)

    if config_id==0:
        wk.mode_w = 0 # 0:normal, 1:fc, 2:cnn, 3:single cnn, 4:regression
        wk.mode_e = 0 # 0:ce, 1:mse
        wk.loop(300)
    else:
        wk.mode_w = 1 # 0:normal, 1:fc, 2:cnn, 3:single cnn, 4:regression
        wk.mode_e = 0 # 0:ce, 1:mse 
        wk.mse_idx = 4

        wk.mode_w = 0
        wk.mode_e = 0
        wk.loop(300)
        return 0

        for i in range(1000):
            print(i)
            if i % 2 == 1: # CNN
                wk.mode_w = 2
                for i in range(1, 5):
                    layer = r.get_layer_at(i)
                    layer.lock = False
                #
                wk.loop(1, 10)
            else: # FC
                wk.mode_w = 1
                r.propagate()
                for i in range(1, 5):
                    layer = r.get_layer_at(i)
                    layer.lock = True
                #
                wk.loop(1, 50)
            #
        #
    #   

    #, "./wi.csv", data_size, num_class, train_data_batch, train_label_batch, batch_offset, batch_size)
    #wk = mpi.worker(com, rank, size, config_id, "./wi.csv", data_size, num_class, train_data_batch, train_label_batch, batch_offset, batch_size)
    #wk = worker(com, rank, size, package_id, config_id, "./wi.csv")
    #ce = wk.evaluate()
    #print("[%d] %f" %(rank, ce))
    #return 0
    #wk.loop(300)

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
