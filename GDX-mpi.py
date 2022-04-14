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
import gdx
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
    config_id = 0
    #
    
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    train_data_batch = util.pickle_load(mnist.TRAIN_IMAGE_BATCH_PATH)
    train_label_batch = util.pickle_load(mnist.TRAIN_LABEL_BATCH_PATH)
    batch_offset = mnist.MINI_BATCH_START[rank]
    batch_size = mnist.MINI_BATCH_SIZE[rank]
        
    #processor_name = MPI.Get_processor_name()
    
    cp.cuda.Device(rank).use()
    my_gpu = gdx.Gdx(rank)
    
    r = mnist.setup_dnn(my_gpu, config_id)
    r.set_scale_input(1)
    r.set_path("./wi.csv")
    r.load()
    r.update_weight()
    r.prepare(batch_size, data_size, num_class)
    r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
    
    wk = mpi.worker(com, rank, size, r)
    #, "./wi.csv", data_size, num_class, train_data_batch, train_label_batch, batch_offset, batch_size)
    #wk = mpi.worker(com, rank, size, config_id, "./wi.csv", data_size, num_class, train_data_batch, train_label_batch, batch_offset, batch_size)
    #wk = worker(com, rank, size, package_id, config_id, "./wi.csv")
    #ce = wk.evaluate()
    #print("[%d] %f" %(rank, ce))
    #return 0
    wk.loop(50)

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
